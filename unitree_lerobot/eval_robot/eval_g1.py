"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import cv2
import torch
import logging
import traceback
from functools import partial

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    AsyncChunks,
    limit_command,
    predict_action,
    predict_chunk,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    image_client = None
    try:
        # --- Setup Phase ---
        if cfg.ee.lower() in ("", "none"):  # arms only (e.g. the G1 with rubber hands)
            cfg.ee = ""
        image_client, camera_config = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        # The controller starts by streaming an all-zero arm target. Hold the measured pose instead, so the
        # arms stay where they are until the start is confirmed below.
        arm_ctrl.ctrl_dual_arm(arm_ctrl.get_current_dual_arm_q(), np.zeros(arm_dof))

        # --waist=true: the policy has the waist yaw as the LAST value of its state and action.
        state_dim = int(policy.config.input_features["observation.state"].shape[0])
        base_dim = arm_dof + (2 * ee_dof if cfg.ee else 0)
        has_waist = bool(cfg.waist)
        if state_dim != base_dim + int(has_waist):
            hint = ""
            if state_dim == base_dim + 1:
                hint = " -> this policy has the waist joint: add --waist=true"
            elif state_dim == base_dim:
                hint = " -> this policy has no waist joint: remove --waist=true"
            raise ValueError(
                f"policy state has {state_dim} values, but arm={cfg.arm} ee='{cfg.ee or 'none'}' "
                f"waist={has_waist} gives {base_dim + int(has_waist)}{hint}"
            )
        if has_waist and cfg.motion:
            logger_mp.warning("motion mode: the motion controller owns the waist, the policy's waist action is ignored")
        logger_mp.info(
            f"State/action layout: {arm_dof} arm + {base_dim - arm_dof} hand + {int(has_waist)} waist yaw | "
            f"limits: max_vel {cfg.max_vel} rad/s, max_lead {cfg.max_lead} rad (0 = off)"
        )
        # Cameras: the policy decides which views are used (the ones it was trained with); the image server's
        # config says which ones exist. Views the policy does not have are simply not fed to it.
        image_keys = list(policy.config.image_features)
        prefix = "observation.images."
        available = []
        if camera_config["head_camera"]["enable_zmq"]:
            available.append(prefix + "cam_left_high")
            if camera_config["head_camera"].get("binocular", True):
                available.append(prefix + "cam_right_high")
        for side in ("left", "right"):
            if camera_config[f"{side}_wrist_camera"]["enable_zmq"]:
                available.append(f"{prefix}cam_{side}_wrist")
        available = [cfg.rename_map.get(k, k) for k in available]
        logger_mp.info(
            f"Cameras used by the policy: {[k[len(prefix):] for k in image_keys]} | "
            f"sent by the image server: {[k[len(prefix):] for k in available]}"
        )
        if not_sent := [k[len(prefix):] for k in image_keys if k not in available]:
            raise ValueError(
                f"the policy needs the camera(s) {not_sent}, but the image server does not send them "
                f"(enable them in the image server's camera config, or use --rename_map if only the name differs)"
            )
        image_shapes = {k: tuple(policy.config.image_features[k].shape[-2:]) for k in image_keys}  # (H, W)
        device = get_safe_torch_device(policy.config.device)
        actor = None
        if cfg.replan == "async":
            actor = AsyncChunks(
                lambda obs: predict_chunk(
                    obs, policy, device, preprocessor, postprocessor, policy.config.use_amp, step["task"],
                    use_dataset=cfg.use_dataset, robot_type=None,
                )
            )

        # Get initial pose from the first step of the dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        prev_cmd = None
        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            logger_mp.info("Initializing robot to starting pose...")
            if cfg.max_vel > 0:
                # Go to the start pose under the same limits as the policy: at most max_vel rad/s per joint,
                # never more than max_lead rad ahead of the measured joints.
                def measure():
                    waist_q = [arm_ctrl.get_current_waist_yaw_q()] if has_waist else []
                    return np.concatenate((arm_ctrl.get_current_dual_arm_q(), waist_q))

                init_waist = [float(step["observation.state"][-1])] if has_waist else []
                init_target = np.concatenate((init_arm_pose.astype(float), init_waist))
                prev_cmd = measure()
                distance = float(np.max(np.abs(init_target - prev_cmd)))
                logger_mp.info(f"Start move: {distance:.2f} rad at up to {cfg.max_vel} rad/s")
                deadline = time.perf_counter() + 3.0 * distance / cfg.max_vel + 3.0
                while np.max(np.abs(init_target - prev_cmd)) > 1e-3 and time.perf_counter() < deadline:
                    tick_start = time.perf_counter()
                    prev_cmd = limit_command(init_target, prev_cmd, measure(), cfg.max_vel, cfg.max_lead, cfg.frequency)
                    arm_ctrl.ctrl_dual_arm(prev_cmd[:arm_dof], arm_ik.solve_tau(prev_cmd[:arm_dof]))
                    if has_waist and not cfg.motion:
                        arm_ctrl.ctrl_waist_yaw(float(prev_cmd[arm_dof]))
                    time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - tick_start)))
                time.sleep(0.5)  # let the arms settle
            else:  # limits off: Unitree's original jump to the start pose
                tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
                robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
                time.sleep(1.0)  # Give time for the robot to move
            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                # 1. Get Observations
                observation, current_arm_q = process_images_and_observations(image_client, camera_config, arm_ctrl)
                missing = [k for k in image_keys if k not in {cfg.rename_map.get(o, o) for o in observation}]
                if missing or current_arm_q is None:
                    logger_mp.warning(f"Waiting for {missing or 'the arm state'}")
                    time.sleep(0.1)
                    continue
                # the policy gets its images at the size it was trained with
                for name in list(observation):
                    height_width = image_shapes.get(cfg.rename_map.get(name, name))
                    if height_width and tuple(observation[name].shape[:2]) != height_width:
                        resized = cv2.resize(
                            observation[name].numpy(), (height_width[1], height_width[0]), interpolation=cv2.INTER_AREA
                        )
                        observation[name] = torch.from_numpy(resized)
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                waist_state = np.array([arm_ctrl.get_current_waist_yaw_q()]) if has_waist else np.array([])
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state, waist_state), axis=0)
                ).float()
                observation["observation.state"] = state_tensor
                # 2. Get Action from Policy
                if actor is not None:  # never waits for the network (except for the very first action)
                    action_np = np.asarray(actor(observation))
                else:
                    action = predict_action(
                        observation,
                        policy,
                        device,
                        preprocessor,
                        postprocessor,
                        policy.config.use_amp,
                        step["task"],
                        use_dataset=cfg.use_dataset,
                        robot_type=None,
                    )
                    action_np = action.cpu().numpy()
                # 3. Execute Action
                # arms (+ waist) go through the safety limits; the hands are sent as predicted
                measured = np.concatenate((current_arm_q, waist_state))
                target = np.concatenate((action_np[:arm_dof], action_np[-1:] if has_waist else []))
                if prev_cmd is None:
                    prev_cmd = measured.copy()
                prev_cmd = limit_command(target, prev_cmd, measured, cfg.max_vel, cfg.max_lead, cfg.frequency)
                arm_action = prev_cmd[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)
                if has_waist and not cfg.motion:
                    arm_ctrl.ctrl_waist_yaw(float(prev_cmd[arm_dof]))

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except Exception as e:
        logger_mp.error(f"An error occurred: {e}\n{traceback.format_exc()}")
    finally:
        if image_client is not None:
            image_client.close()


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")
    if cfg.threads:
        torch.set_num_threads(cfg.threads)

    dataset = LeRobotDataset(repo_id=cfg.repo_id, root=cfg.root or None)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()
    # A dataset recorded with one head view called cam_head: feed it the (left) head image.
    if "observation.images.cam_head" in policy.config.image_features and not cfg.rename_map:
        cfg.rename_map = {"observation.images.cam_left_high": "observation.images.cam_head"}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
