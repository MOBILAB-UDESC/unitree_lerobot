import threading
import time
import numpy as np
import torch
from typing import Any
from contextlib import nullcontext
from copy import copy
import logging
from dataclasses import dataclass, field
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def extract_observation(step: dict):
    observation = {}

    for key, value in step.items():
        if key.startswith("observation.images."):
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in [1, 3]:
                value = np.transpose(value, (2, 0, 1))
            observation[key] = value

        elif key == "observation.state":
            observation[key] = value

    return observation


def predict_action(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    use_dataset: bool | None = False,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            if not use_dataset:
                # Skip non-tensor observations (like task strings)
                if not hasattr(observation[name], "unsqueeze"):
                    continue
                if "images" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()

            observation[name] = observation[name].unsqueeze(0).to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        observation = preprocessor(observation)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        action = postprocessor(action)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def predict_chunk(
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None = None,
    use_dataset: bool | None = False,
    robot_type: str | None = None,
):
    """Like predict_action, but returns the whole predicted chunk (chunk_size, action_dim) as numpy and does
    not touch the policy's action queue. Same numbers as calling predict_action chunk_size times."""
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        for name in observation:
            if not use_dataset:
                if not hasattr(observation[name], "unsqueeze"):
                    continue
                if "images" in name:
                    observation[name] = observation[name].type(torch.float32) / 255
                    observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0).to(device)
        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""
        actions = policy.predict_action_chunk(preprocessor(observation))  # (1, chunk, dim), normalised
        return np.stack(
            [postprocessor(actions[:, i]).squeeze(0).float().cpu().numpy() for i in range(actions.shape[1])]
        )


class AsyncChunks:
    """actor(observation) -> action, without stopping the control loop for inference.

    A worker thread computes the next chunk from the newest observation while the current chunk keeps
    playing. A finished chunk replaces the current one and is entered at the tick that has passed since its
    observation, so the actions stay aligned in time. Only the first action of a run waits for the network.
    `plan(observation)` returns a (chunk, dim) array, e.g. a partial of predict_chunk.
    """

    def __init__(self, plan):
        self.plan = plan
        self.lock = threading.Lock()
        self.generation = 0
        self.closed = False
        self.worker = None
        self.reset()

    def reset(self):
        with self.lock:
            self.generation += 1  # a plan still being computed belongs to the old run: drop it
            self.plans = 0
            self.current, self.i, self.tick, self.ready, self.busy = None, 0, 0, None, False

    def _work(self, observation, tick, generation):
        try:
            chunk = self.plan(observation)
        except Exception as e:  # keep playing the current chunk; the next tick starts a new plan
            logger_mp.error(f"[AsyncChunks] planning failed: {e}")
            chunk = None
        with self.lock:
            if generation == self.generation:
                self.busy = False
                if chunk is not None:
                    self.ready = (chunk, tick)
                    self.plans += 1

    def __call__(self, observation):
        if self.current is None:
            self.current, self.i = self.plan(observation), 0
        with self.lock:
            if self.ready is not None:
                (chunk, tick), self.ready = self.ready, None
                if self.tick - tick < len(chunk):
                    self.current, self.i = chunk, self.tick - tick
            start = not self.busy and not self.closed
            if start:
                self.busy = True
                args = (dict(observation), self.tick, self.generation)
        if start:
            self.worker = threading.Thread(target=self._work, args=args, daemon=True)
            self.worker.start()
        action = self.current[min(self.i, len(self.current) - 1)]
        self.i += 1
        self.tick += 1
        return action


    def close(self, timeout: float = 15.0):
        """Stop planning and wait for the planning thread. Without this, a Ctrl-C usually finds that thread
        in the middle of a torch op, and the process aborts when Python exits it there ("terminate called
        without an active exception", exit code 134)."""
        with self.lock:
            self.closed = True
            worker = self.worker
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():
                logger_mp.warning(f"[AsyncChunks] the planning thread is still running after {timeout:.0f} s")


def limit_command(target, prev, measured, max_vel: float, max_lead: float, frequency: float):
    """Safety limits for one control tick: keep the command within max_lead rad of the measured joints, then
    let it change by at most max_vel rad/s against the previous command. A limit of 0 is off."""
    target = np.asarray(target, dtype=float)
    if max_lead > 0:
        target = np.clip(target, measured - max_lead, measured + max_lead)
    if max_vel > 0:
        step = max_vel / frequency
        target = prev + np.clip(target - prev, -step, step)
    return target


def ramp_to(target, read_measured, send, max_vel: float, frequency: float, tolerance: float = 0.05,
            timeout: float = 20.0, sleep=time.sleep):
    """Move a joint command from the measured pose to `target`, changing it by at most max_vel rad/s,
    one send(command) per tick. Returns the last command once the command is at the target and the
    measured joints are within `tolerance` rad of it, or after `timeout` seconds (with a warning)."""
    target = np.asarray(target, dtype=float)
    cmd = np.asarray(read_measured(), dtype=float)
    step = max_vel / frequency
    for _ in range(max(1, int(timeout * frequency))):
        cmd = cmd + np.clip(target - cmd, -step, step)
        send(cmd)
        sleep(1.0 / frequency)
        if np.max(np.abs(cmd - target)) < 1e-9 and np.max(np.abs(np.asarray(read_measured()) - target)) < tolerance:
            return cmd
    worst = float(np.max(np.abs(np.asarray(read_measured()) - target)))
    logger_mp.warning(f"ramp_to: not at the start pose after {timeout:.0f} s ({worst:.2f} rad away); continuing")
    return cmd


def release_arms(arm_ctrl, arm_ik, max_vel: float, frequency: float, release_seconds: float = 2.0,
                 sleep=time.sleep):
    """End of a live run in motion mode (rt/arm_sdk): move the arm command back, at most max_vel rad/s
    (0.5 if 0), to the pose the arms had when the controller started, then hand the arms back to the walking
    controller by lowering the arm_sdk weight from 1 to 0 over release_seconds."""

    def send(q):
        arm_ctrl.ctrl_dual_arm(q, arm_ik.solve_tau(q))

    ramp_to(arm_ctrl.initial_arm_q, arm_ctrl.get_current_dual_arm_q, send, max_vel if max_vel > 0 else 0.5,
            frequency, sleep=sleep)
    steps = max(1, int(release_seconds * frequency))
    for k in range(1, steps + 1):
        arm_ctrl.set_arm_sdk_weight(1.0 - k / steps)
        sleep(1.0 / frequency)


def format_start_pose(names, start, measured) -> str:
    """One line per joint: where the dataset's first frame has it, where the robot has it now, how far it must move."""
    lines = ["Start pose (the dataset's first frame) against the robot now, rad:"]
    for name, goal, now in zip(names, start, measured):
        lines.append(f"  {name:<22} start {goal:+.2f}   now {now:+.2f}   move {goal - now:+.2f}")
    return "\n".join(lines)


def reset_policy(policy: PreTrainedPolicy):
    policy.reset()


def cleanup_resources(image_info: dict[str, Any]):
    """Safely close and unlink shared memory resources."""
    logger_mp.info("Cleaning up shared memory resources.")
    for shm in image_info["shm_resources"]:
        if shm:
            shm.close()
            shm.unlink()


def to_list(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().ravel().tolist()
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def to_scalar(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return float(x.detach().cpu().ravel()[0].item())
    if isinstance(x, np.ndarray):
        return float(x.ravel()[0])
    if isinstance(x, (list, tuple)):
        return float(x[0])
    return float(x)


@dataclass
class EvalRealConfig:
    repo_id: str
    policy: PreTrainedConfig | None = None

    root: str = ""
    episodes: int = 0
    frequency: float = 30.0

    # Basic control parameters
    arm: str = "G1_29"  # G1_29, G1_23
    ee: str = "dex3"  # dex3, dex1, inspire1, brainco

    # Mode flags
    motion: bool = False
    headless: bool = False
    visualization: bool = False
    send_real_robot: bool = False
    use_dataset: bool = False

    # image server address (the robot's PC2); 127.0.0.1 for a simulator on this PC
    image_host: str = "192.168.123.164"
    # talk to a simulator (DDS domain 1) instead of the real robot
    sim: bool = False
    # "async": compute the next action chunk in the background while the current one plays (the control loop
    # never stops for inference); "chunk": stock behaviour, play n_action_steps and then wait for the network
    replan: str = "async"
    # PyTorch CPU threads (0 = leave as is)
    threads: int = 0
    # the policy also has the waist yaw joint, as the LAST value of its state and action (arms + hands + waist)
    waist: bool = False
    # safety limits on the arm (and waist) commands, 0 = off:
    # max_vel  [rad/s] how fast a joint command may change
    # max_lead [rad]   how far a joint command may be ahead of the measured joint
    max_vel: float = 1.0
    max_lead: float = 0.25
    # read the joints and cameras and run the policy, but send nothing to the robot (no arm, hand or
    # mobile controller is created); logs how far the policy's actions are from the measured joints
    dry_run: bool = False

    rename_map: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            logging.warning(
                "No pretrained path was provided, evaluated policy will be built from scratch (random weights)."
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]
