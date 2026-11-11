"""MCP server that lets a Claude Code session see and drive a Unitree G1's arms step by step
("Claude as the policy"), in the MuJoCo sim or on the real robot, through ``g1lab.robot_io.G1Arms``.

    scripts/run_mcp.sh                                   # sim on DDS domain 1, images on :60000
    scripts/run_mcp.sh --domain 4 --image-port 62000     # a second sim (--domain 4 --port-offset 2000)
    scripts/run_mcp.sh --target real --allow-real        # the real robot

Tools: observe, move_hands, squeeze, move_joints, go_home, reset_scene, wait, start_episode,
stop_episode, get_status. See g1lab/MCP.md.

Robot frame (= xr_teleoperate's IK frame): pelvis-centred, x forward, y left, z up, metres.
Rotations are roll/pitch/yaw in degrees, R = Rz(yaw) @ Ry(pitch) @ Rx(roll). With rpy = (0, 0, 0)
the wrist frames point their x axis forward (fingers forward) and the palms face each other
(left palm toward -y, right palm toward +y): the pose used to squeeze a box between the hands.

stdout is the MCP wire. The unitree / teleimager / ipopt libraries print to fd 1, so ``main()``
moves fd 1 onto stderr for the whole process and speaks MCP over a private duplicate of it.
"""

import argparse
import asyncio
import atexit
import io
import logging
import os
import signal
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

log = logging.getLogger("g1lab.mcp")

CAMERAS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
JOINT_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
]
# URDF joint limits of the IK model (rad), used to clamp joint targets.
JOINT_LOWER = np.array([-3.09, -1.59, -2.62, -1.05, -1.97, -1.61, -1.61,
                        -3.09, -2.25, -2.62, -1.05, -1.97, -1.61, -1.61])
JOINT_UPPER = np.array([2.67, 2.25, 2.62, 2.09, 1.97, 1.61, 1.61,
                        2.67, 1.59, 2.62, 2.09, 1.97, 1.61, 1.61])
# Arm rest pose of the lab sim (03_real2sim/lab_scene.py ARM_START = mean first frame of the real episodes).
REST_POSE = np.array([0.188, -0.055, -0.177, -0.164, 0.069, -0.289, 0.079,
                      0.155, 0.026, 0.031, -0.247, -0.160, -0.074, -0.143])

DEFAULT_WORKSPACE = ((0.10, 0.65), (-0.5, 0.5), (-0.30, 0.45))  # x, y, z bounds for hand targets
LIMITS = {  # per target: max joint change per tool call (rad), max joint speed (rad/s), gravity ff scale
    # xr_teleoperate sends the IK's full gravity torque; in the MuJoCo sim that overshoots by ~1-1.5 cm
    # (upwards), while 0.5 tracks within ~0.5 cm, so the sim defaults to 0.5.
    "sim": dict(max_joint_step=2.0, max_joint_vel=2.0, gravity_scale=0.5),
    "real": dict(max_joint_step=0.8, max_joint_vel=0.8, gravity_scale=1.0),
}
IK_MAX_POS_ERR = 0.02  # m: reject Cartesian targets the IK cannot reach closer than this
MOVE_RATE = 50.0       # Hz of the interpolated command stream
RECORD_FPS = 30

# Lab sim geometry (sim/scene_lab.xml, unitree_mujoco g1_29dof.xml): the world frame has the pelvis at
# the x/y origin facing +x, and the fixed pelvis 0.793 m above the floor. robot z = world z - 0.793.
PELVIS_Z_SIM = 0.793
WRAP_HALF_HEIGHT = 0.133  # wrapped_box body origin is its bottom face; its centre is 0.133 m above
# Rubber hand geometry relative to the L_ee / R_ee wrist frames (from the hand meshes):
PALM_FORWARD = 0.06  # palm centre is ~6 cm ahead of the wrist frame along its x axis
PALM_INSET = 0.02    # palm surface is ~2 cm from the wrist frame towards the palm (+-y)
MIN_SQUEEZE_WIDTH = 0.05
WRAP_HALF_WIDTH = 0.113
# The arms only reach palm centres ~0.40 m forward at box height, less than the box centre (~0.43-0.48),
# so by default squeeze grips the box's near half: palm centres this far behind its near face.
GRIP_DEPTH = 0.04


# ----------------------------------------------------------------------------- pure helpers
def rpy_to_R(rpy_deg):
    r, p, y = np.radians(np.asarray(rpy_deg, dtype=float).reshape(3))
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_rpy(R):
    R = np.asarray(R, dtype=float)
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 1 - 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: put everything into yaw
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.degrees([roll, pitch, yaw])


def quat_to_R(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def make_T(xyz, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def rot_err_deg(Ra, Rb):
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def clamp_workspace(xyz, workspace=DEFAULT_WORKSPACE):
    """Clamp a hand target into the workspace box. Returns (clamped xyz, list of clamped axes)."""
    xyz = np.asarray(xyz, dtype=float).reshape(3)
    out, clamped = xyz.copy(), []
    for i, (axis, (lo, hi)) in enumerate(zip("xyz", workspace)):
        out[i] = min(max(xyz[i], lo), hi)
        if abs(out[i] - xyz[i]) > 1e-9:
            clamped.append(f"{axis} {xyz[i]:+.3f}->{out[i]:+.3f}")
    return out, clamped


def squeeze_targets(width_m, center_xyz):
    """Wrist (L_ee, R_ee) 4x4 targets with the palms facing each other, palm surfaces width_m apart,
    palm centres symmetric about center_xyz (robot frame)."""
    c = np.asarray(center_xyz, dtype=float).reshape(3)
    half = width_m / 2.0 + PALM_INSET
    left = c + np.array([-PALM_FORWARD, half, 0.0])
    right = c + np.array([-PALM_FORWARD, -half, 0.0])
    return make_T(left, np.eye(3)), make_T(right, np.eye(3))


def _norm_joint(name):
    n = str(name).replace("_", "").replace("joint", "").lower()
    return n[1:] if n.startswith("k") and n[1:5] in ("left", "righ") else n


_JOINT_INDEX = {_norm_joint(n): i for i, n in enumerate(JOINT_NAMES)}


def joints_from_input(q, base):
    """A 14-vector, or a dict of named joints (e.g. {"kLeftElbow": 0.5} or {"left_elbow": 0.5}) applied
    on top of ``base``."""
    if isinstance(q, dict):
        out = np.array(base, dtype=float).copy()
        unknown = []
        for k, v in q.items():
            i = _JOINT_INDEX.get(_norm_joint(k))
            if i is None:
                unknown.append(k)
            else:
                out[i] = float(v)
        if unknown:
            raise ValueError(f"unknown joint name(s) {unknown}; valid: {JOINT_NAMES}")
        return out
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.shape != (14,):
        raise ValueError(f"need 14 joint values (left 7 then right 7), got {arr.size}")
    return arr


def fmt_vec(v, fmt="{:+.3f}"):
    return "(" + ", ".join(fmt.format(float(x)) for x in v) + ")"


def fmt_pose(T):
    return f"xyz {fmt_vec(T[:3, 3])} m  rpy {fmt_vec(R_to_rpy(T[:3, :3]), '{:+.1f}')} deg"


def encode_jpeg(rgb, max_width=480, quality=80):
    import cv2

    img = np.asarray(rgb)
    h, w = img.shape[:2]
    if max_width and w > max_width:
        img = cv2.resize(img, (int(max_width), int(round(h * max_width / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


try:  # raising the SDK's ToolError is what makes the message reach the model (mcp 2.x hides others)
    from mcp.server.mcpserver.exceptions import ToolError as _ToolErrorBase
except ImportError:
    try:
        from mcp.server.fastmcp.exceptions import ToolError as _ToolErrorBase
    except ImportError:
        _ToolErrorBase = RuntimeError


class ToolFailure(_ToolErrorBase):
    """A tool call that was refused or failed; the message is shown to the model."""


# ----------------------------------------------------------------------------- recorder
class Recorder:
    """Records LeRobot frames at 30 Hz in a background thread while the tools move the robot.
    The writer is opened per episode and closed (finalized) after it, so an interrupted server never
    leaves an unreadable dataset behind."""

    def __init__(self, session, root, repo_id):
        self.session, self.root, self.repo_id = session, root, repo_id
        self.writer = None
        self.thread = None
        self.stop_evt = threading.Event()
        self.task = None
        self.frames = 0
        self.dropped = 0
        self.t0 = None
        self.error = None
        self.lock = threading.Lock()

    @property
    def active(self):
        return self.writer is not None

    def start(self, task):
        if self.active:
            raise ToolFailure(f"already recording an episode ({self.task!r}); call stop_episode first")
        robot = self.session.robot()
        try:
            from g1lab.lerobot_writer import LeRobotWriter
        except Exception as e:  # missing file, or lerobot not installed in this env
            raise ToolFailure(
                f"LeRobot recording unavailable: cannot import g1lab.lerobot_writer ({type(e).__name__}: {e}). "
                "It needs lerobot 0.4.1 (the `g1lab` conda env has it: start the server with "
                "G1_MCP_ENV=g1lab scripts/run_mcp.sh ...).") from e
        try:
            writer = LeRobotWriter(self.root, self.repo_id, task, fps=RECORD_FPS)
            writer.start_episode(task)
        except Exception as e:
            raise ToolFailure(f"could not open dataset {self.root} ({type(e).__name__}: {e})") from e
        self.writer, self.task = writer, task
        self.frames = self.dropped = 0
        self.error = None
        self.stop_evt.clear()
        self.t0 = time.monotonic()
        self.thread = threading.Thread(target=self._loop, args=(robot,), name="mcp-recorder", daemon=True)
        self.thread.start()

    def _loop(self, robot):
        period = 1.0 / RECORD_FPS
        t_next = time.perf_counter()
        while not self.stop_evt.is_set():
            try:
                obs = robot.observe()
                images = obs["images"]
                if set(images) != set(CAMERAS) or any(images[c].shape != (480, 640, 3) for c in CAMERAS):
                    self.dropped += 1
                else:
                    self.writer.add_frame(obs["state"], robot.last_command(), images)
                    self.frames += 1
            except Exception as e:  # keep the server alive; report it in stop_episode/get_status
                self.error = f"{type(e).__name__}: {e}"
                log.exception("recorder stopped")
                return
            t_next += period
            delay = t_next - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.5:  # fell far behind: resync instead of bursting
                t_next = time.perf_counter()

    def stop(self, save=True):
        if not self.active:
            raise ToolFailure("not recording (call start_episode first)")
        self.stop_evt.set()
        self.thread.join(timeout=5)
        writer, self.writer = self.writer, None
        duration = time.monotonic() - self.t0
        result = dict(task=self.task, frames=self.frames, dropped=self.dropped, seconds=round(duration, 1),
                      error=self.error, saved=False, episode_index=None, dataset=str(self.root))
        try:
            if save and self.frames:
                result["episode_index"] = writer.save_episode()
                result["saved"] = result["episode_index"] is not None
            else:
                writer.discard_episode()
            result["episodes_in_dataset"] = writer.num_episodes
        finally:
            writer.close()
        return result

    def status(self):
        if not self.active:
            return "recording: off"
        return (f"recording: ON task={self.task!r} frames={self.frames} dropped={self.dropped} "
                f"t={time.monotonic() - self.t0:.1f}s" + (f" ERROR {self.error}" if self.error else ""))

    def shutdown(self):
        if self.active:
            try:
                r = self.stop(save=False)
                log.warning("server exiting: discarded the unsaved episode (%d frames)", r["frames"])
            except Exception:
                log.exception("closing the dataset failed")


# ----------------------------------------------------------------------------- session
class G1Session:
    """Everything the tools do, as plain methods (usable without MCP, e.g. from tests)."""

    def __init__(self, target="sim", domain=None, image_port=60000, image_host=None, iface=None,
                 dataset_root=None, repo_id="local/sim_claude", workspace=DEFAULT_WORKSPACE,
                 max_joint_step=None, max_joint_vel=None, gravity_scale=None, robot_factory=None,
                 connect_timeout=60.0):
        self.target, self.sim = target, target == "sim"
        self.domain, self.image_port, self.image_host, self.iface = domain, image_port, image_host, iface
        self.workspace = tuple(tuple(map(float, b)) for b in workspace)
        lim = LIMITS[target]
        self.max_joint_step = float(max_joint_step or lim["max_joint_step"])
        self.max_joint_vel = float(max_joint_vel or lim["max_joint_vel"])
        self.gravity_scale = float(lim["gravity_scale"] if gravity_scale is None else gravity_scale)
        self.connect_timeout = connect_timeout
        self._factory = robot_factory or self._make_robot
        self._robot = None
        self._connect_error = None
        self._connect_lock = threading.Lock()
        self._connected = threading.Event()
        self._kin_lock = threading.RLock()     # IK + FK share pinocchio/casadi state
        self._motion_lock = threading.Lock()   # one move at a time
        self._grav = None
        self.t_start = time.monotonic()
        root = dataset_root or os.path.join(os.path.dirname(ROOT), "datasets", "sim_claude")
        self.recorder = Recorder(self, os.path.abspath(root), repo_id)

    # ---------------------------------------------------------------- connection
    def _make_robot(self):
        from g1lab.robot_io import G1Arms

        return G1Arms(self.target, iface=self.iface, image_host=self.image_host, domain=self.domain,
                      image_port=self.image_port)

    def connect(self):
        """Connect (idempotent); safe to call from a background thread at startup."""
        with self._connect_lock:
            if self._robot is not None:
                return self._robot
            try:
                log.info("connecting to the %s robot (domain=%s, image port %s)", self.target, self.domain,
                         self.image_port)
                robot = self._factory()
                self._robot = robot
                # xr_teleoperate's controller streams q_target = 0 until the first command, so the arms
                # have been pulled towards zero since G1Arms() started: hold them where they are now,
                # which also makes last_command() true again.
                q = np.asarray(robot.state(), dtype=float)
                robot.command(q, self.gravity_tau(q))
                with self._kin_lock:
                    robot.fk()  # builds the IK model now rather than on the first tool call
                self._connect_error = None
                log.info("connected")
            except BaseException as e:
                self._connect_error = f"{type(e).__name__}: {e}"
                log.exception("robot connection failed")
            finally:
                self._connected.set()
            return self._robot

    def robot(self):
        if self._robot is not None:
            return self._robot
        if not self._connected.wait(self.connect_timeout):
            raise ToolFailure("still connecting to the robot; try again in a few seconds")
        if self._robot is None:
            # retry once synchronously (e.g. the sim was started after the server)
            self._connected.clear()
            self.connect()
        if self._robot is None:
            raise ToolFailure(
                f"robot not connected ({self._connect_error}). Is the {self.target} running on DDS domain "
                f"{self.domain} with its image server on port {self.image_port}?")
        return self._robot

    # ---------------------------------------------------------------- kinematics
    def fk(self, q=None):
        with self._kin_lock:
            return self.robot().fk(q)

    def ik(self, left_T, right_T):
        with self._kin_lock:
            return self.robot().ik(left_T, right_T)

    def gravity_tau(self, q):
        """Gravity feed-forward for the arms (what xr_teleoperate's IK returns as tau)."""
        if not self.gravity_scale:
            return np.zeros(14)
        try:
            with self._kin_lock:
                if self._grav is None:
                    import pinocchio as pin

                    model = self._robot._ik_solver().reduced_robot.model
                    self._grav = (pin, model, model.createData())
                pin, model, data = self._grav
                return self.gravity_scale * np.asarray(pin.rnea(model, data, np.asarray(q, float), np.zeros(14),
                                                                np.zeros(14)))
        except Exception:
            return np.zeros(14)

    def commanded_q(self):
        return np.asarray(self.robot().last_command(), dtype=float)

    def measured_q(self):
        return np.asarray(self.robot().state(), dtype=float)

    # ---------------------------------------------------------------- motion
    def move_q(self, q_target, duration, check_step=True):
        """Smooth (cosine) move of the commanded arm joints to q_target with gravity feed-forward.
        Returns (clamped q_target, actual duration)."""
        robot = self.robot()
        q1 = np.clip(np.asarray(q_target, dtype=float).reshape(14), JOINT_LOWER, JOINT_UPPER)
        with self._motion_lock:
            q0 = self.commanded_q()
            step = float(np.max(np.abs(q1 - q0)))
            if check_step and step > self.max_joint_step:
                j = JOINT_NAMES[int(np.argmax(np.abs(q1 - q0)))]
                raise ToolFailure(f"refused: {j} would move {step:.2f} rad in one call (limit "
                                  f"{self.max_joint_step:.2f} rad); split the motion into smaller steps")
            duration = max(float(duration), step / self.max_joint_vel, 0.1)
            n = max(1, int(round(duration * MOVE_RATE)))
            t_next = time.perf_counter()
            for i in range(1, n + 1):
                s = 0.5 - 0.5 * np.cos(np.pi * i / n)
                q = q0 + (q1 - q0) * s
                robot.command(q, self.gravity_tau(q))
                t_next += 1.0 / MOVE_RATE
                time.sleep(max(0.0, t_next - time.perf_counter()))
        return q1, duration

    def _settle(self, seconds=0.4):
        time.sleep(seconds)

    def _current_targets(self):
        """Wrist poses of the last command (not the measured, sagging pose), so relative moves
        don't drift."""
        return self.fk(self.commanded_q())

    def move_hands(self, left_xyz=None, right_xyz=None, left_rpy=None, right_rpy=None, relative=True,
                   duration=1.5):
        cur = self._current_targets()
        targets, notes = [], []
        for side, T, xyz, rpy in (("left", cur[0], left_xyz, left_rpy), ("right", cur[1], right_xyz, right_rpy)):
            p, R = T[:3, 3].copy(), T[:3, :3].copy()
            if xyz is not None:
                xyz = np.asarray(xyz, dtype=float).reshape(3)
                p = p + xyz if relative else xyz
            if rpy is not None:
                Rd = rpy_to_R(rpy)
                R = Rd @ R if relative else Rd
            p, clamped = clamp_workspace(p, self.workspace)
            if clamped:
                notes.append(f"{side} target clamped to the workspace: {', '.join(clamped)}")
            targets.append(make_T(p, R))
        return self._move_to_poses(targets[0], targets[1], duration, notes)

    def _move_to_poses(self, left_T, right_T, duration, notes=()):
        notes = list(notes)
        q_sol, _ = self.ik(left_T, right_T)
        q_sol = np.asarray(q_sol, dtype=float)
        L, R = self.fk(q_sol)
        errs = []  # (side, target, IK solution pose, position error, rotation error)
        for side, T, A in (("left", left_T, L), ("right", right_T, R)):
            errs.append((side, T, A, float(np.linalg.norm(A[:3, 3] - T[:3, 3])), rot_err_deg(A[:3, :3], T[:3, :3])))
        bad = [e for e in errs if e[3] > IK_MAX_POS_ERR]
        if bad:
            msg = "; ".join(f"{s}: target {fmt_vec(T[:3, 3])}, closest reachable {fmt_vec(A[:3, 3])} "
                            f"({d * 100:.1f} cm off)" for s, T, A, d, _ in bad)
            raise ToolFailure(f"refused: IK cannot reach the target within {IK_MAX_POS_ERR * 100:.0f} cm "
                              f"({msg}). The robot did not move. (At the wrapped box's centre height the palms reach about "
                              f"x = 0.40 m; higher targets reach further.)")
        for side, _, _, _, rerr in errs:
            if rerr > 10:
                notes.append(f"{side} orientation reachable only within {rerr:.0f} deg of the request")
        _, dur = self.move_q(q_sol, duration)
        self._settle()
        mL, mR = self.fk()
        lines = [f"moved in {dur:.2f} s."] + notes
        for side, T, M in (("left", left_T, mL), ("right", right_T, mR)):
            lines.append(f"{side:5s} target   {fmt_pose(T)}")
            lines.append(f"{side:5s} achieved {fmt_pose(M)}  (measured; error "
                         f"{np.linalg.norm(M[:3, 3] - T[:3, 3]) * 100:.1f} cm, "
                         f"{rot_err_deg(M[:3, :3], T[:3, :3]):.1f} deg)")
        lines.append(self._success_line())
        return "\n".join(l for l in lines if l)

    def _squeeze_default_centre(self):
        """The current palm midpoint while the hands are around the wrapped box (or no box is known);
        otherwise the box's grip point: its near half, at its centre height."""
        L, R = self._current_targets()
        palms = (L[:3, 3] + R[:3, 3]) / 2 + np.array([PALM_FORWARD, 0, 0])
        box = self._box_center()
        if box is None:
            return palms, "the current palm midpoint"
        grip = box.copy()
        grip[0] = box[0] - WRAP_HALF_WIDTH + GRIP_DEPTH
        if np.linalg.norm((palms - grip)[:2]) < 0.08 and abs(palms[2] - grip[2]) < 0.15:
            return palms, "the current palm midpoint (hands are around the box)"
        return grip, f"the wrapped box (its near half: palm centres {GRIP_DEPTH * 100:.0f} cm behind its near face)"

    def squeeze(self, width_m, height_z=None, center_xy=None, duration=1.5):
        width_m = float(width_m)
        if width_m < MIN_SQUEEZE_WIDTH:
            raise ToolFailure(f"refused: width_m must be >= {MIN_SQUEEZE_WIDTH} m (hands would collide)")
        notes = []
        if center_xy is None or height_z is None:
            ref, src = self._squeeze_default_centre()
        cx, cy = (float(center_xy[0]), float(center_xy[1])) if center_xy is not None else (ref[0], ref[1])
        cz = float(height_z) if height_z is not None else ref[2]
        if center_xy is None:
            notes.append(f"centre x/y from {src}")
        if height_z is None:
            notes.append(f"height from {src}")
        centre = np.array([cx, cy, cz])
        left_T, right_T = squeeze_targets(width_m, centre)
        for side, T in (("left", left_T), ("right", right_T)):
            p, clamped = clamp_workspace(T[:3, 3], self.workspace)
            if clamped:
                notes.append(f"{side} target clamped to the workspace: {', '.join(clamped)}")
            T[:3, 3] = p
        head = f"squeeze: palms {width_m * 100:.1f} cm apart, centred at {fmt_vec(centre)} m"
        return head + "\n" + self._move_to_poses(left_T, right_T, duration, notes)

    def move_joints(self, q, duration=2.0):
        q1 = joints_from_input(q, self.commanded_q())
        clipped = np.clip(q1, JOINT_LOWER, JOINT_UPPER)
        notes = []
        if np.any(np.abs(clipped - q1) > 1e-9):
            notes.append("some targets clamped to the joint limits")
        q1, dur = self.move_q(clipped, duration)
        self._settle()
        meas = self.measured_q()
        return "\n".join([f"moved in {dur:.2f} s; max |measured - commanded| = "
                          f"{np.max(np.abs(meas - q1)):.3f} rad"] + notes + [self._joint_table(), self._wrist_lines(),
                                                                          self._success_line()])

    def go_home(self, pose="zero", duration=3.0):
        if pose not in ("zero", "rest"):
            raise ToolFailure("pose must be 'zero' (xr_teleoperate home, forearms forward) or 'rest' "
                              "(arms down, the lab episodes' start pose)")
        q = np.zeros(14) if pose == "zero" else REST_POSE
        _, dur = self.move_q(q, duration, check_step=False)
        if pose == "zero":
            self.robot().go_home()  # xr_teleoperate's own go-home: waits until within 0.05 rad of zero
            self.robot().command(np.zeros(14), self.gravity_tau(np.zeros(14)))
        self._settle()
        return f"at the {pose} pose after {dur:.2f} s.\n" + self._wrist_lines()

    def reset_scene(self):
        if not self.sim:
            raise ToolFailure("reset_scene only exists in simulation")
        robot = self.robot()
        _, dur = self.move_q(REST_POSE, 2.0, check_step=False)  # get the hands out of the way first
        before = robot.sim_state()
        robot.reset_sim()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            time.sleep(0.1)
            if robot.sim_state() is not before and time.monotonic() - t0 > 0.5:
                break
        time.sleep(0.5)  # let the objects drop / settle
        return f"arms at the rest pose ({dur:.1f} s), objects reset.\n" + self._objects_text()

    def wait(self, seconds):
        seconds = float(seconds)
        if not 0 <= seconds <= 60:
            raise ToolFailure("seconds must be in [0, 60]")
        time.sleep(seconds)
        return f"waited {seconds:.2f} s. {self._success_line()}".strip()

    # ---------------------------------------------------------------- perception
    def _box_center(self):
        st = self.robot().sim_state() if self.sim else None
        if not st or "wrapped_box" not in st:
            return None
        b = st["wrapped_box"]
        return np.asarray(b["pos"]) - [0, 0, PELVIS_Z_SIM] + quat_to_R(b["quat"]) @ [0, 0, WRAP_HALF_HEIGHT]

    def _objects_text(self):
        if not self.sim:
            return ""
        st = self.robot().sim_state()
        if not st:
            return "objects: no rt/sim_state received yet"
        lines = ["objects (robot frame, m; yaw deg):"]
        for name, label in (("wrapped_box", "centre"), ("cardboard_box", "floor centre")):
            if name not in st:
                continue
            pos, quat = np.asarray(st[name]["pos"]), st[name]["quat"]
            R = quat_to_R(quat)
            p = pos - [0, 0, PELVIS_Z_SIM]
            if name == "wrapped_box":
                p = p + R @ [0, 0, WRAP_HALF_HEIGHT]
                size = "23x23x27 cm"
            else:
                size = "inner 32x48 cm"
            tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
            lines.append(f"  {name:13s} {label} {fmt_vec(p)}  yaw {np.degrees(np.arctan2(R[1, 0], R[0, 0])):+.0f}"
                         f"  tilt {tilt:.0f}  ({size})")
        lines.append(self._success_line())
        return "\n".join(lines)

    def _success_line(self):
        if not self.sim:
            return ""
        st = self.robot().sim_state()
        if not st:
            return ""
        return f"success: {bool(st.get('success'))}"

    def _joint_table(self):
        meas, cmd = self.measured_q(), self.commanded_q()
        lines = ["joints (rad)  measured | commanded:"]
        for i, n in enumerate(JOINT_NAMES):
            lines.append(f"  {n:20s} {meas[i]:+.3f} | {cmd[i]:+.3f}")
        return "\n".join(lines)

    def _wrist_lines(self):
        mL, mR = self.fk()
        cL, cR = self.fk(self.commanded_q())
        return "\n".join([
            "wrists (robot frame: x fwd, y left, z up):",
            f"  left  measured  {fmt_pose(mL)}",
            f"  left  commanded {fmt_pose(cL)}",
            f"  right measured  {fmt_pose(mR)}",
            f"  right commanded {fmt_pose(cR)}",
        ])

    def observe(self, cameras=None, max_width=480):
        """Returns (text, [(camera, jpeg bytes), ...])."""
        cameras = list(CAMERAS if cameras is None else cameras)
        bad = [c for c in cameras if c not in CAMERAS]
        if bad:
            raise ToolFailure(f"unknown camera(s) {bad}; choose from {CAMERAS}")
        robot = self.robot()
        images = robot.images() if cameras else {}
        jpegs, missing = [], []
        for c in cameras:
            if c in images:
                jpegs.append((c, encode_jpeg(images[c], max_width)))
            else:
                missing.append(c)
        parts = [f"{self.target} | t={time.monotonic() - self.t_start:.1f}s | {self.recorder.status()}",
                 self._joint_table(), self._wrist_lines(), self._objects_text()]
        if missing:
            parts.append(f"no frame yet from: {missing}")
        return "\n".join(p for p in parts if p), jpegs

    def status(self):
        lines = [f"target={self.target} domain={self.domain} image_port={self.image_port}",
                 f"connected={self._robot is not None}" + (f" (error: {self._connect_error})"
                                                            if self._connect_error else ""),
                 f"limits: workspace x{self.workspace[0]} y{self.workspace[1]} z{self.workspace[2]} m, "
                 f"max joint step {self.max_joint_step} rad/call, max joint speed {self.max_joint_vel} rad/s, "
                 f"IK reject > {IK_MAX_POS_ERR * 100:.0f} cm, gravity ff x{self.gravity_scale}",
                 f"dataset: {self.recorder.root} (repo_id {self.recorder.repo_id})",
                 self.recorder.status()]
        if self._robot is not None:
            lines += [self._wrist_lines(), self._success_line()]
        return "\n".join(l for l in lines if l)

    def shutdown(self):
        self.recorder.shutdown()


# ----------------------------------------------------------------------------- MCP wiring
INSTRUCTIONS = """\
You control the two arms of a Unitree G1 humanoid (29 DoF, rubber gloved hands without fingers to
actuate; legs/waist are held still). In the lab task, squeeze the 23x23x27 cm bubble-wrapped box between
both palms, lift it, and put it into the open cardboard box on the robot's left (+y), then release.
Frames: robot frame, pelvis-centred, x forward, y left, z up (metres). rpy in degrees; rpy (0,0,0) =
fingers forward, palms facing each other. Work in small steps and call observe() after each move to
check the cameras (cam_head looks down at the desk; the wrist cameras look past each hand).
Typical flow: reset_scene -> observe -> squeeze(width_m=0.30) to bracket the box -> squeeze(width_m=0.20)
to grip -> move_hands(left_xyz=[0,0,0.1], right_xyz=[0,0,0.1]) to lift -> move over the cardboard box ->
lower -> squeeze(width_m=0.32) to release -> observe (success flag). Record a demonstration with
start_episode(task) ... stop_episode(save=True)."""


def _import_mcp():
    try:
        from mcp.server.mcpserver import Image, MCPServer
    except ImportError:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP as MCPServer, Image
    return MCPServer, Image


def build_server(session):
    MCPServer, Image = _import_mcp()
    server = MCPServer(name="g1", instructions=INSTRUCTIONS)

    def run(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except ToolFailure:
            raise
        except Exception as e:
            log.exception("tool failed")
            raise ToolFailure(f"{type(e).__name__}: {e}") from e

    @server.tool(structured_output=False)
    def observe(cameras: list[str] | None = None, max_width: int = 480) -> list:
        """See the robot: one JPEG per camera (cam_head, cam_left_wrist, cam_right_wrist; downscaled to
        max_width px; pass cameras=[] for text only) plus text with the 14 arm joints (measured and
        commanded), both wrist poses (robot frame xyz m + rpy deg), and in sim the object poses and the
        task success flag."""
        text, jpegs = run(session.observe, cameras, max_width)
        out = [text]
        for cam, data in jpegs:
            out += [f"{cam}:", Image(data=data, format="jpeg")]
        return out

    @server.tool(structured_output=False)
    def move_hands(left_xyz: list[float] | None = None, right_xyz: list[float] | None = None,
                   left_rpy: list[float] | None = None, right_rpy: list[float] | None = None,
                   relative: bool = True, duration: float = 1.5) -> str:
        """Cartesian move of the wrists (robot frame: x forward, y left, z up; metres, rpy degrees).
        relative=True (default): xyz is added to the current commanded wrist position and rpy is a rotation
        applied on top of the current orientation; relative=False: absolute targets. Arms/parts left as
        None keep their current pose. Targets are clamped to the workspace (x 0.1..0.65, |y|<0.5,
        z -0.3..0.45); targets the IK cannot reach within 2 cm are refused (the robot does not move).
        Returns the target and achieved (measured) poses."""
        return run(session.move_hands, left_xyz, right_xyz, left_rpy, right_rpy, relative, duration)

    @server.tool(structured_output=False)
    def squeeze(width_m: float, height_z: float | None = None, center_xy: list[float] | None = None,
                duration: float = 1.5) -> str:
        """Place both palms facing each other (fingers forward), palm surfaces width_m apart, symmetric
        about a centre point given as the palm centres' midpoint (robot frame, m). center_xy / height_z
        default to the current palm midpoint while the hands are around the wrapped box, else (sim) to the
        box's near half at its centre height (the arms reach palm centres only to x ~0.40 m at that
        height). The box is 0.23 m wide: ~0.30 brackets it, ~0.19-0.21 grips it, ~0.32 releases it.
        Note: the move is interpolated in joint space; approach from behind/above if a hand could sweep
        through the box."""
        return run(session.squeeze, width_m, height_z, center_xy, duration)

    @server.tool(structured_output=False)
    def move_joints(q: list[float] | dict[str, float], duration: float = 2.0) -> str:
        """Move the arm joints (rad): a list of 14 values (left shoulder pitch/roll/yaw, elbow, wrist
        roll/pitch/yaw, then the same for the right arm) or a dict of named joints to change, e.g.
        {"kLeftElbow": 0.8, "right_wrist_yaw": 0.2}; others keep their commanded value."""
        return run(session.move_joints, q, duration)

    @server.tool(name="go_home", structured_output=False)
    def go_home_tool(pose: str = "zero") -> str:
        """Move the arms home smoothly. pose='zero': xr_teleoperate's home (all joints 0: elbows bent,
        forearms pointing forward at about pelvis height -- in the lab this can touch the wrapped box);
        pose='rest': arms hanging down, the start pose of the recorded lab episodes."""
        return run(session.go_home, pose)

    @server.tool(structured_output=False)
    def reset_scene() -> str:
        """Simulation only: move the arms to the rest pose, then reset the wrapped box and cardboard box
        (re-randomised if the sim runs with --randomize). Returns the new object poses."""
        return run(session.reset_scene)

    @server.tool(name="wait", structured_output=False)
    def wait_tool(seconds: float) -> str:
        """Hold the current command for `seconds` (0..60), e.g. to let objects settle."""
        return run(session.wait, seconds)

    @server.tool(structured_output=False)
    def start_episode(task: str = "put the bubble-wrapped box into the cardboard box.") -> str:
        """Start recording a LeRobot episode (30 Hz, state + 3 camera images from observe(), action = last
        joint command) in a background thread while you keep using the motion tools."""
        run(session.recorder.start, task)
        return f"recording episode {task!r} into {session.recorder.root}"

    @server.tool(structured_output=False)
    def stop_episode(save: bool = True) -> str:
        """Stop recording; save=True encodes and saves the episode (a few seconds), save=False discards it."""
        r = run(session.recorder.stop, save)
        return "\n".join(f"{k}: {v}" for k, v in r.items())

    @server.tool(structured_output=False)
    def get_status() -> str:
        """Connection, limits, recording state and current wrist poses."""
        return run(session.status)

    return server


def _protect_stdout():
    """Point fd 1 (and sys.stdout) at stderr for the rest of the process; return a private fd that
    still reaches the MCP client."""
    sys.stdout.flush()
    wire_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return wire_fd


async def _serve_stdio(server, wire_fd):
    import anyio
    from mcp.server.stdio import stdio_server

    low = getattr(server, "_lowlevel_server", None) or getattr(server, "_mcp_server", None)
    if low is None:  # unknown SDK layout: fall back to its own stdio handling
        os.dup2(wire_fd, 1)
        await server.run_stdio_async()
        return
    wire =anyio.wrap_file(io.TextIOWrapper(os.fdopen(wire_fd, "wb", closefd=False), encoding="utf-8",
                                            write_through=True))
    async with stdio_server(stdout=wire) as (read_stream, write_stream):
        await low.run(read_stream, write_stream, low.create_initialization_options())


def parse_args(argv=None):
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=["sim", "real"], default=env("G1_MCP_TARGET", "sim"))
    p.add_argument("--domain", type=int, default=None, help="DDS domain (default: sim 1, real 0)")
    p.add_argument("--image-port", type=int, default=int(env("G1_MCP_IMAGE_PORT", "60000")),
                   help="teleimager config port (sim started with --port-offset N -> 60000+N)")
    p.add_argument("--image-host", default=None, help="image server host (default: sim 127.0.0.1, real PC2)")
    p.add_argument("--iface", default=None, help="DDS network interface (default: sim lo, real auto)")
    p.add_argument("--dataset-root", default=env("G1_MCP_DATASET_ROOT", os.path.join(os.path.dirname(ROOT), "datasets", "sim_claude")))
    p.add_argument("--repo-id", default=env("G1_MCP_REPO_ID", "local/sim_claude"))
    p.add_argument("--allow-real", action="store_true", help="required with --target real")
    p.add_argument("--workspace", type=float, nargs=6, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
                   default=[v for b in DEFAULT_WORKSPACE for v in b], help="hand target clamp box (robot frame, m)")
    p.add_argument("--max-joint-step", type=float, default=None,
                   help="max joint change per tool call, rad (default sim 2.0, real 0.8)")
    p.add_argument("--max-joint-vel", type=float, default=None,
                   help="max joint speed of interpolated moves, rad/s (default sim 2.0, real 0.8)")
    p.add_argument("--gravity-scale", type=float, default=None,
                   help="scale of the gravity feed-forward torque (default sim 0.5, real 1.0 = xr_teleoperate; "
                        "0 disables it)")
    p.add_argument("--log-level", default=env("G1_MCP_LOG_LEVEL", "INFO"))
    a = p.parse_args(argv)
    if a.domain is None and env("G1_MCP_DOMAIN"):
        a.domain = int(env("G1_MCP_DOMAIN"))
    return a


def main(argv=None):
    wire_fd = _protect_stdout()  # before anything can print
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.target == "real" and not args.allow_real:
        log.error("refusing to drive the real robot without --allow-real")
        sys.exit(2)
    w = args.workspace
    session = G1Session(target=args.target, domain=args.domain, image_port=args.image_port,
                        image_host=args.image_host, iface=args.iface, dataset_root=args.dataset_root,
                        repo_id=args.repo_id, workspace=((w[0], w[1]), (w[2], w[3]), (w[4], w[5])),
                        max_joint_step=args.max_joint_step, max_joint_vel=args.max_joint_vel,
                        gravity_scale=args.gravity_scale)
    server = build_server(session)

    done = threading.Event()

    def cleanup():
        if not done.is_set():
            done.set()
            session.shutdown()

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *_: (cleanup(), os._exit(0)))
    threading.Thread(target=session.connect, name="mcp-connect", daemon=True).start()
    try:
        asyncio.run(_serve_stdio(server, wire_fd))
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
        sys.stderr.flush()
        # DDS / ZMQ threads can keep the interpreter alive; nothing else is left to flush.
        os._exit(0)


if __name__ == "__main__":
    main()
