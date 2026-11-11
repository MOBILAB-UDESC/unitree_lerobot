"""One arms-only interface to a Unitree G1, identical for the MuJoCo sim and the real robot.

Everything goes through the same channels xr_teleoperate uses: DDS rt/lowcmd + rt/lowstate (via
xr_teleoperate's own G1_29_ArmController, so gains, held legs, CRC and the real-robot velocity clip
match teleop exactly) and the teleimager image server. The sim additionally publishes rt/sim_state
(object poses, success) and accepts rt/reset_pose/cmd.

    from g1lab.robot_io import G1Arms
    robot = G1Arms("sim")                 # or "real" (DDS domain 0, image server on PC2)
    obs = robot.observe()                 # {"state": (14,), "images": {cam_head, cam_left_wrist, cam_right_wrist}}
    robot.command(q14)                    # arm joint targets, streamed at 250 Hz by the controller thread
    robot.move_to(q14, duration=2.0)      # smooth interpolated move
    q14 = robot.ik(left_T, right_T)       # 4x4 wrist poses (robot frame) -> joints, xr_teleoperate's IK

Joint order (= dataset `observation.state` / `action`): left shoulder pitch/roll/yaw, elbow,
wrist roll/pitch/yaw, then the same for the right arm. With `waist=True` a 15th value, waist yaw (G1
motor 12), is appended to state / command / last_command (the 15-joint "ArmsWaist" dataset format); the
legs stay held. ik()/fk() then work in the pelvis frame with the torso turned by the waist.
"""

import json
import os
import sys
import threading
import time
from contextlib import contextmanager

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THIRD_PARTY = os.environ.get("G1_THIRD_PARTY", os.path.join(REPO, "third_party"))
XR_TELEOP = os.path.join(THIRD_PARTY, "xr_teleoperate")
for p in (XR_TELEOP, os.path.join(XR_TELEOP, "teleop")):
    if p not in sys.path:
        sys.path.insert(0, p)

TARGETS = {
    "sim": dict(domain=1, iface="lo", image_host="127.0.0.1", simulation=True),
    "real": dict(domain=0, iface=None, image_host="192.168.123.164", simulation=False),
}
CAMERA_KEYS = {"cam_head": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}
JOINT_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
]

WAIST_YAW_INDEX = 12      # G1 motor index of the waist yaw joint
WAIST_YAW_LIMIT = 1.0     # rad; the joint allows ~2.6, keep the policy's range modest

_dds_initialized = None


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def init_dds(domain, iface):
    """ChannelFactoryInitialize may only run once per process."""
    global _dds_initialized
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    if _dds_initialized is None:
        ChannelFactoryInitialize(domain, iface)
        _dds_initialized = (domain, iface)
    elif _dds_initialized != (domain, iface):
        raise RuntimeError(f"DDS already initialised as {_dds_initialized}, not {(domain, iface)}")


@contextmanager
def _in_teleop_dir():
    """xr_teleoperate's IK loads its URDF relative to the teleop/ directory."""
    old = os.getcwd()
    os.chdir(os.path.join(XR_TELEOP, "teleop"))
    try:
        yield
    finally:
        os.chdir(old)


class G1Arms:
    def __init__(self, target="sim", iface=None, image_host=None, motion=False, cameras=True,
                 domain=None, image_port=60000, waist=False):
        """domain / image_port override the target defaults (e.g. a second sim started with
        --domain 3 --port-offset 1000 -> domain=3, image_port=61000)."""
        if target not in TARGETS:
            raise ValueError(f"target must be one of {list(TARGETS)}")
        cfg = dict(TARGETS[target])
        if iface:
            cfg["iface"] = iface
        if image_host:
            cfg["image_host"] = image_host
        if domain is not None:
            cfg["domain"] = domain
        self.target, self.cfg, self.sim = target, cfg, cfg["simulation"]
        init_dds(cfg["domain"], cfg["iface"])

        import logging_mp
        logging_mp.basicConfig(level=logging_mp.WARNING)
        from teleop.robot_control.robot_arm import G1_29_ArmController

        self.arm = G1_29_ArmController(motion_mode=motion, simulation_mode=self.sim)
        # The controller starts streaming zero arm targets; hold the measured pose right away so the
        # arms don't jerk toward zero on connect (on the real robot that is only limited by the 30 rad/s clip).
        self._q_cmd = self.arm.get_current_dual_arm_q().copy()
        self._tau_cmd = np.zeros(14)
        self.arm.ctrl_dual_arm(self._q_cmd, self._tau_cmd)
        self._ik = None
        self.waist = waist
        self._waist_cmd = self.waist_q()

        self.images_enabled = cameras
        if cameras:
            from teleimager.image_client import ImageClient
            self.img = ImageClient(host=cfg["image_host"], request_port=image_port, request_bgr=True)
            self.cam_config = self.img.get_cam_config()

        self._sim_state = None
        if self.sim:
            from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
            self._String = String_
            self._reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_)
            self._reset_pub.Init()
            self._state_sub = ChannelSubscriber("rt/sim_state", String_)
            self._state_sub.Init(self._on_sim_state, 10)

    # ------------------------------------------------------------------ state
    def _on_sim_state(self, msg):
        try:
            self._sim_state = json.loads(msg.data)
        except ValueError:
            pass

    def waist_q(self):
        """Measured waist yaw (rad)."""
        return float(self.arm.lowstate_buffer.GetData().motor_state[WAIST_YAW_INDEX].q)

    def state(self):
        """Measured arm joint positions (14,), plus waist yaw (15,) with waist=True."""
        q = self.arm.get_current_dual_arm_q()
        return (np.append(q, self.waist_q()) if self.waist else q).astype(np.float32)

    def last_command(self):
        """The joint targets most recently sent (14, or 15 with waist) - this is the dataset `action`."""
        return (np.append(self._q_cmd, self._waist_cmd) if self.waist else self._q_cmd).astype(np.float32)

    def images(self):
        """RGB uint8 images keyed like the dataset (missing cameras are left out)."""
        out = {}
        if not self.images_enabled:
            return out
        getters = {"cam_head": self.img.get_head_frame, "cam_left_wrist": self.img.get_left_wrist_frame,
                   "cam_right_wrist": self.img.get_right_wrist_frame}
        for key, topic in CAMERA_KEYS.items():
            cfg = self.cam_config.get(f"{topic}_camera", {})
            if not cfg.get("enable_zmq"):
                continue
            frame = getters[key]()
            bgr = getattr(frame, "bgr", None) if frame is not None else None
            if bgr is None:
                continue
            if cfg.get("binocular"):  # stereo head: the dataset uses one 640-wide view
                bgr = bgr[:, : bgr.shape[1] // 2]
            out[key] = bgr[:, :, ::-1].copy()
        return out

    def observe(self):
        return {"state": self.state(), "images": self.images()}

    def sim_state(self):
        """Object poses + success from the sim (None on the real robot or before the first message)."""
        return self._sim_state

    def success(self):
        return bool(self._sim_state and self._sim_state.get("success"))

    # ---------------------------------------------------------------- command
    def command(self, q, tau=None):
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size == 15:
            if not self.waist:
                raise ValueError("15-joint command needs G1Arms(waist=True)")
            self.command_waist(q[14])
            q = q[:14]
        q = q.reshape(14)
        self._q_cmd = q.copy()
        self._tau_cmd = np.zeros(14) if tau is None else np.asarray(tau, dtype=np.float64).reshape(14)
        self.arm.ctrl_dual_arm(self._q_cmd, self._tau_cmd)

    def command_waist(self, yaw):
        """Waist yaw target (rad), held by xr_teleoperate's controller with its leg/waist gains."""
        if not self.waist:
            raise RuntimeError("waist control needs G1Arms(waist=True)")
        self._waist_cmd = float(np.clip(yaw, -WAIST_YAW_LIMIT, WAIST_YAW_LIMIT))
        self.arm.msg.motor_cmd[WAIST_YAW_INDEX].q = self._waist_cmd

    def move_to(self, q, duration=2.0, rate=30.0, on_step=None):
        """Cosine-interpolated move from the current command to q (14, or 15 with the waist);
        on_step(i) runs each tick (e.g. recording)."""
        q1 = np.asarray(q, dtype=np.float64).reshape(-1)
        q0 = self.last_command().astype(np.float64)[: q1.size]
        n = max(1, int(duration * rate))
        t_next = time.perf_counter()
        for i in range(1, n + 1):
            s = 0.5 - 0.5 * np.cos(np.pi * i / n)
            self.command(q0 + (q1 - q0) * s)
            if on_step:
                on_step(i)
            t_next += 1.0 / rate
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def go_home(self):
        """xr_teleoperate's own return-to-zero (and arm_sdk weight ramp-down in motion mode)."""
        self.arm.ctrl_dual_arm_go_home()
        self._q_cmd = np.zeros(14)

    # -------------------------------------------------------------- kinematics
    def _ik_solver(self):
        if self._ik is None:
            from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
            with _in_teleop_dir():
                self._ik = G1_29_ArmIK()
        return self._ik

    def ik(self, left_T, right_T, seed_q=None, waist=None):
        """4x4 wrist poses (pelvis frame: x fwd, z up) -> arm (q14, tau14), for the torso turned by `waist`
        (default: the current waist command; 0 without waist control). xr_teleoperate's IK model has the
        waist locked at 0, so the targets are rotated into the turned torso's frame."""
        ik = self._ik_solver()
        yaw = (self._waist_cmd if self.waist else 0.0) if waist is None else float(waist)
        if yaw:
            left_T, right_T = rot_z(-yaw) @ np.asarray(left_T), rot_z(-yaw) @ np.asarray(right_T)
        q = self._q_cmd if seed_q is None else np.asarray(seed_q)
        # xr_teleoperate's IK smooths its output with a moving-average filter (built for 30 Hz streams);
        # for a one-shot target, re-solve until the filter has settled on the answer.
        for _ in range(8):
            q, tau = ik.solve_ik(np.asarray(left_T), np.asarray(right_T), q, np.zeros(14))
        return q, tau

    def fk(self, q=None, waist=None):
        """Wrist poses (L_ee, R_ee) in the pelvis frame for arm joints q (default: measured) and waist yaw
        (default: measured with waist control, else 0), as 4x4 arrays."""
        import pinocchio as pin
        ik = self._ik_solver()
        model, data = ik.reduced_robot.model, ik.reduced_robot.data
        q = self.arm.get_current_dual_arm_q() if q is None else np.asarray(q)[:14]
        yaw = (self.waist_q() if self.waist else 0.0) if waist is None else float(waist)
        pin.framesForwardKinematics(model, data, q)
        Rw = rot_z(yaw)
        return (Rw @ data.oMf[model.getFrameId("L_ee")].homogeneous, Rw @ data.oMf[model.getFrameId("R_ee")].homogeneous)

    # --------------------------------------------------------------------- sim
    def reset_sim(self, category=1):
        """Ask the sim to reset (and re-randomise, if it runs with --randomize) the task objects."""
        if not self.sim:
            raise RuntimeError("reset_sim() only exists in simulation")
        self._reset_pub.Write(self._String(data=str(category)))
