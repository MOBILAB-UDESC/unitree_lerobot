"""MuJoCo stand-in for a real Unitree G1 (29 DoF, no dexterous hands) driven by xr_teleoperate --sim.

What it provides, mirroring what xr_teleoperate expects from unitree_sim_isaaclab:
  * DDS (domain 1 by default):  sub rt/lowcmd  (unitree_hg LowCmd_)   -> per-step PD torque (debug mode)
                                sub rt/arm_sdk  (unitree_hg LowCmd_)   -> arms, blended by weight (--motion)
                                sub rt/api/sport/request (loco RPC)    -> joystick walking (--motion)
                                pub rt/lowstate (unitree_hg LowState_) @ 500 Hz
                                sub rt/reset_pose/cmd (String_)        -> reset objects
                                pub rt/sim_state (String_, JSON)       -> object poses for --record
  * teleimager-compatible image server: ZMQ REP :60000 answers the camera config, ZMQ PUB per camera topic
    (head :55555 - side-by-side stereo JPEG by default; left/right wrist :55556/:55557 with add-ons).

Without --motion, xr_teleoperate drives every joint through rt/lowcmd (like debug mode on the robot).
With --motion, this file plays Unitree's motion controller: it owns legs and waist, blends the arms
from rt/arm_sdk, and turns LocoClient.Move(vx, vy, vyaw) into motion. vx/vy walk in the direction the
torso faces; vyaw twists the waist (look around), turning the whole body past the waist limit.
Walking is kinematic (the "gantry" carries the pelvis while the legs animate), so it needs FIXED mode.

Viewer keys:  9 = toggle fixed pelvis / free-standing,  0 = reset robot and objects.

Add-ons extend the robot and scene without touching this file (--addon NAME, a module on PYTHONPATH):
  wrist_cameras  (02_wrist_cameras)  5010 wrists, knit gloves, MOBILAB brackets + Intel D405 wrist cameras
  lab_scene      (03_real2sim)       the real2sim lab desk task (--scene lab is a shortcut for it)
An add-on module may define (all optional):
  REQUIRES = [...]                    other add-ons it needs (loaded first)
  PROVIDES_SCENE = True               it attaches its own environment instead of --scene
  add_args(parser)                    extra command-line flags
  build(spec, args, cameras, prefix)  edit the MjSpec; returns the {teleimager topic: [camera names]} map
  make_task(model, prefix, args)      object resets / success check (reset_objects, object_poses, success, arm_home)
  camera_effects(rgb, camera_name)    post-process a rendered image (exposure, blur, noise)
  VIEWER_GEOM_GROUPS, DEBUG_VIEW      viewer-only geom groups to show; overview camera for --debug-video
"""

import argparse
import contextlib
import importlib
import json
import os
import queue
import signal
import threading
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_api.msg.dds_ import Request_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# Upstream clones (install/install.sh puts them here; override with G1_THIRD_PARTY).
THIRD_PARTY = os.environ.get("G1_THIRD_PARTY", os.path.join(REPO, "third_party"))
ROBOT_XML = os.path.join(THIRD_PARTY, "unitree_mujoco", "unitree_robots", "g1", "g1_29dof.xml")
SCENE_XML = os.path.join(HERE, "scenes", "scene_demo.xml")
ADDON_ALIASES = {"lab": "lab_scene", "wrist": "wrist_cameras"}
ENV_PREFIX = "env/"  # names from the scene file get this prefix when attached around the robot

NUM_BODY_MOTORS = 29
MODE_MACHINE_G1_29 = 5  # echoed back by xr_teleoperate in its LowCmd_; any constant works in sim

# Gains used to hold the start pose until xr_teleoperate sends its first LowCmd_
# (same values G1_29_ArmController uses for legs/waist, arms and wrists).
HOLD_KP = np.array([300.0] * 15 + [80.0] * 4 + [40.0] * 3 + [80.0] * 4 + [40.0] * 3)
HOLD_KD = np.array([3.0] * 15 + [3.0] * 4 + [1.5] * 3 + [3.0] * 4 + [1.5] * 3)

ARMS = slice(15, 29)
ARM_SDK_WEIGHT = 29          # xr_teleoperate puts the arm_sdk blend weight in motor_cmd[29].q
WAIST_YAW = 12
CMD_TIMEOUT = 0.5            # s without rt/lowcmd or rt/arm_sdk before falling back to the sim's own control

LOCO_REQUEST_TOPIC = "rt/api/sport/request"  # unitree_sdk2py G1 LocoClient (service "sport")
LOCO_API_SET_FSM_ID = 7101   # Damp() and friends -> stop
LOCO_API_SET_VELOCITY = 7105
WAIST_YAW_LIMIT = 1.5        # rad; real joint allows 2.6, the body turns instead beyond this
BODY_FOLLOW_RATE = 1.5       # 1/s; while walking, the pelvis turns to face where the torso looks
GAIT_FREQ = 1.8              # steps per second (animation only)

KEY_TOGGLE_BASE = ord("9")
KEY_RESET = ord("0")


def load_addons(names):
    """Import add-on modules (and what they REQUIRE), dependencies first, each once."""
    out = []

    def load(name):
        mod = importlib.import_module(ADDON_ALIASES.get(name, name))
        if mod in out:
            return
        for dep in getattr(mod, "REQUIRES", []):
            load(dep)
        out.append(mod)

    for n in names:
        load(n)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", type=int, default=1, help="DDS domain id (xr_teleoperate --sim uses 1)")
    p.add_argument("--iface", default="lo", help="DDS network interface (match xr_teleoperate --network-interface)")
    p.add_argument("--scene", default="demo",
                   help="'demo' (table + cubes), a path to an environment MJCF, or 'lab' (= --addon lab_scene)")
    p.add_argument("--addon", action="append", default=[],
                   help="load an add-on module, e.g. wrist_cameras or lab_scene (repeatable)")
    p.add_argument("--debug-video", default=None, help="write a 15 fps mp4 of a fixed overview camera to this path")
    p.add_argument("--log-qpos", default=None,
                   help="save the simulation state at 30 Hz to this .npz when the sim exits (qpos, gantry and body "
                        "poses, success flag), to re-render a session offline (docs/make_media.py)")
    p.add_argument("--free", action="store_true", help="start free-standing instead of with a fixed pelvis")
    p.add_argument("--no-viewer", action="store_true", help="run headless")
    p.add_argument("--no-camera", action="store_true", help="disable the head camera image server")
    p.add_argument("--walk-scale", type=float, default=1.0,
                   help="multiplier on joystick walk speed (xr_teleoperate caps it at 0.3 m/s)")
    p.add_argument("--turn-scale", type=float, default=2.0,
                   help="multiplier on joystick turn rate (xr_teleoperate caps it at 0.3 rad/s)")
    p.add_argument("--cam-pos", type=float, nargs=3, default=[0.0576, 0.0, 0.43],
                   help="stereo rig centre in torso_link frame [m] (default: D435 head mount)")
    p.add_argument("--cam-pitch", type=float, default=40.0, help="downward camera pitch [deg]")
    p.add_argument("--baseline", type=float, default=0.064, help="stereo baseline [m]")
    p.add_argument("--fovy", type=float, default=90.0, help="vertical field of view [deg]")
    p.add_argument("--img-w", type=int, default=640, help="width of ONE eye (stream is 2x wide)")
    p.add_argument("--img-h", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--camera-shadows", choices=["all", "head", "none"], default="head",
                   help="which camera streams render shadows (a shadow pass costs ~5 ms per camera on an "
                        "integrated GPU; 'none' helps a slow PC reach 30 fps)")
    p.add_argument("--config-port", type=int, default=60000, help="teleimager camera-config REP port")
    p.add_argument("--zmq-port", type=int, default=55555, help="teleimager head camera PUB port")
    p.add_argument("--port-offset", type=int, default=0,
                   help="add to every image port (config + streams), to run several sims side by side "
                        "(use with a different --domain)")
    # Add-ons can add their own flags (shown by --help too), so find them first.
    first = argparse.ArgumentParser(add_help=False)
    first.add_argument("--scene", default="demo")
    first.add_argument("--addon", action="append", default=[])
    pre, _ = first.parse_known_args(argv)
    names = pre.addon + (["lab_scene"] if pre.scene == "lab" else [])
    addons = load_addons(names)
    for mod in addons:
        if hasattr(mod, "add_args"):
            mod.add_args(p)
    args = p.parse_args(argv)
    args.addons = addons
    return args


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
def camera_quat(pitch_deg):
    """Quaternion for a MuJoCo camera looking along body +x, tilted down by pitch, image-up ~ +z."""
    th = np.deg2rad(pitch_deg)
    x_axis = [0.0, -1.0, 0.0]
    y_axis = [np.sin(th), 0.0, np.cos(th)]
    z_axis = [-np.cos(th), 0.0, np.sin(th)]  # camera looks along -z
    mat = np.array([x_axis, y_axis, z_axis]).T.flatten()
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def copy_visual(spec, env):
    """Attaching an environment keeps the robot file's <visual>; take the environment's lighting instead."""
    for attr in ("ambient", "diffuse", "specular"):
        setattr(spec.visual.headlight, attr, getattr(env.visual.headlight, attr))
    spec.visual.rgba.haze = env.visual.rgba.haze
    spec.visual.quality.shadowsize = max(spec.visual.quality.shadowsize, env.visual.quality.shadowsize)


def build_model(args):
    """Returns (model, cameras): cameras maps teleimager topics to MuJoCo camera names (2 = stereo)."""
    if not os.path.exists(ROBOT_XML):
        raise SystemExit(f"G1 model not found at {ROBOT_XML} - run install/install.sh (or set G1_THIRD_PARTY)")
    spec = mujoco.MjSpec.from_file(ROBOT_XML)
    addons = getattr(args, "addons", [])
    if not any(getattr(a, "PROVIDES_SCENE", False) for a in addons):
        env = mujoco.MjSpec.from_file(SCENE_XML if args.scene == "demo" else args.scene)
        spec.attach(env, prefix=ENV_PREFIX, frame=spec.worldbody.add_frame())
        copy_visual(spec, env)

    # Head camera: a stereo pair at the head's D435 mount (the headset shows it in 3D). Add-ons may replace it.
    torso = spec.body("torso_link")
    quat = camera_quat(args.cam_pitch)
    for name, side in (("head_left", 1.0), ("head_right", -1.0)):
        cam = torso.add_camera()
        cam.name = name
        cam.pos = [args.cam_pos[0], args.cam_pos[1] + side * args.baseline / 2, args.cam_pos[2]]
        cam.quat = quat
        cam.fovy = args.fovy
    cameras = {"head_camera": ["head_left", "head_right"]}

    # Unitree's model has visual-only rubber hands; make them collide so they can't pass through things.
    for side in ("left", "right"):
        for geom in spec.body(f"{side}_wrist_yaw_link").geoms:
            if geom.meshname == f"{side}_rubber_hand":
                geom.contype, geom.conaffinity = 1, 1

    for mod in addons:
        if hasattr(mod, "build"):
            cameras = mod.build(spec, args, cameras, ENV_PREFIX) or cameras

    # "Gantry": a mocap body the pelvis is welded to in FIXED mode; moving it makes the robot walk.
    spec.worldbody.add_body(name="gantry", mocap=True, pos=spec.body("pelvis").pos)
    weld = spec.add_equality()
    weld.name = "pelvis_fix"
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.name1, weld.name2 = "pelvis", "gantry"
    # anchor, relpose pos, relpose quat, torquescale: pelvis frame == gantry frame
    # (MjSpec's default data is not "compute from qpos0" like the XML default).
    weld.data[:11] = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1]

    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, args.img_w)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, args.img_h)
    return spec.compile(), cameras


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def yaw_of(quat):
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ----------------------------------------------------------------------------------------------
# Cameras -> teleimager protocol
# ----------------------------------------------------------------------------------------------
CAMERA_PORTS = {"head_camera": (55555, 60001), "left_wrist_camera": (55556, 60002), "right_wrist_camera": (55557, 60003)}


class CameraServer:
    """Answers teleimager's config request and publishes one JPEG stream per camera topic.

    Uses teleimager's own ZMQ classes so the wire format matches xr_teleoperate's ImageClient.
    A topic with two MuJoCo cameras is a side-by-side stereo stream (binocular). Rendering stays on
    the caller's (GL) thread; camera effects, JPEG encoding and publishing run on a worker.
    """

    def __init__(self, model, args, cameras):
        from teleimager.image_client import ZMQ_PublisherManager, ZMQ_Responser

        self.args = args
        self.effects = [a.camera_effects for a in getattr(args, "addons", []) if hasattr(a, "camera_effects")]
        self.renderer = mujoco.Renderer(model, height=args.img_h, width=args.img_w)
        self.frames = 0    # rendered frames (run() reports the rate)
        self.streams = []  # (topic, zmq port, [camera ids], shadows)
        self.cam_config = {}
        for topic, (zmq_port, webrtc_port) in CAMERA_PORTS.items():
            if topic == "head_camera":
                zmq_port = args.zmq_port
            zmq_port += args.port_offset
            names = cameras.get(topic, [])
            enabled = bool(names)
            self.cam_config[topic] = {
                "enable_zmq": enabled, "zmq_port": zmq_port,
                "enable_webrtc": False, "webrtc_port": webrtc_port, "type": "mujoco",
                "image_shape": [args.img_h, args.img_w * max(1, len(names))],
                "binocular": len(names) == 2, "fps": int(args.fps),
            }
            if enabled:
                # Shadows double the render time; only the head view gets them (wrist views are close-ups).
                shadows = args.camera_shadows == "all" or (args.camera_shadows == "head" and topic == "head_camera")
                self.streams.append((topic, zmq_port, [(model.camera(n).id, n) for n in names], shadows))
        self.responser = ZMQ_Responser(self.cam_config, port=args.config_port + args.port_offset)
        self.publisher = ZMQ_PublisherManager.get_instance()
        self.queue = queue.Queue(maxsize=1)
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
        threading.Thread(target=self._encode_loop, daemon=True).start()

    def describe(self):
        return ", ".join(f"{t} {self.cam_config[t]['image_shape'][1]}x{self.cam_config[t]['image_shape'][0]}"
                         f" :{p}" for t, p, _, _ in self.streams)

    def render(self, data, lock):
        """lock guards `data` (physics runs on another thread); it is only held while the scene is read."""
        frames = []
        for _, port, cam_ids, shadows in self.streams:
            views = []
            for cam_id, name in cam_ids:
                with lock:
                    self.renderer.update_scene(data, camera=cam_id)
                self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
                views.append((name, self.renderer.render().copy()))
            frames.append((port, views))
        self.frames += 1
        if self.queue.full():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
        self.queue.put_nowait(frames)

    def _encode_loop(self):
        while True:
            for port, views in self.queue.get():
                # Camera effects (add-ons: exposure, blur, noise) run here, off the physics/GL thread.
                for effect in self.effects:
                    views = [(name, effect(view, name)) for name, view in views]
                rgb = views[0][1] if len(views) == 1 else np.hstack([v for _, v in views])
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.encode_params)
                if ok:
                    self.publisher.publish(jpg.tobytes(), port)


# ----------------------------------------------------------------------------------------------
# Simulator
# ----------------------------------------------------------------------------------------------
class G1XrSim:
    def __init__(self, args):
        self.args = args
        self.model, self.cameras = build_model(args)
        self.data = mujoco.MjData(self.model)
        m = self.model
        assert m.nu == NUM_BODY_MOTORS, f"expected {NUM_BODY_MOTORS} motors, model has {m.nu}"

        self.dt = m.opt.timestep
        self.ctrl_range = m.actuator_ctrlrange.copy()
        self.weld_id = m.equality("pelvis_fix").id
        self.gantry = m.body("gantry").mocapid[0]
        self.stand_pos = m.body("pelvis").pos.copy()
        # Leg geoms that touch the floor; disabled while hanging on the gantry so the feet don't drag.
        leg_bodies = {i for i in range(m.nbody) if any(k in m.body(i).name for k in ("hip", "knee", "ankle"))}
        self.leg_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in leg_bodies and m.geom_contype[g]]
        self.leg_geom_contact = (m.geom_contype[self.leg_geoms].copy(), m.geom_conaffinity[self.leg_geoms].copy())
        self.objects = [m.body(i).name for i in range(m.nbody)
                        if m.body_jntnum[i] == 1 and m.jnt_type[m.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE
                        and m.body(i).name != "pelvis"]
        self.imu_quat = m.sensor("imu_quat").adr[0]
        self.imu_gyro = m.sensor("imu_gyro").adr[0]
        self.imu_acc = m.sensor("imu_acc").adr[0]

        # Latest motor command: q, dq, kp, kd, tau. Starts as "hold the start pose".
        self.cmd_lock = threading.Lock()
        self.cmd = np.zeros((5, NUM_BODY_MOTORS))
        self.cmd[2], self.cmd[3] = HOLD_KP, HOLD_KD
        self.last_cmd_time = None
        # Latest rt/arm_sdk arm command (q, dq, kp, kd, tau for joints 15-28) and its blend weight.
        self.arm_sdk = np.zeros((5, 14))
        self.arm_sdk_weight = 0.0
        self.last_arm_sdk_time = None
        self.mode = None  # "lowcmd" | "motion" | None (sim holds the pose) - only used for logging

        # Motion-controller state (used when rt/lowcmd is not driving the robot).
        self.loco_lock = threading.Lock()
        self.vel_cmd = np.zeros(3)   # vx, vy, vyaw from LocoClient.Move
        self.vel_cmd_until = 0.0
        self.base_yaw = 0.0
        self.waist_yaw = 0.0
        self.gait_phase = 0.0
        self.gait_amp = 0.0
        self.legs_waist_q = np.zeros(15)
        self.free_walk_warned = False

        # Snapshot of sensor data for the lowstate publisher thread.
        self.state_lock = threading.Lock()
        self.state = np.zeros(3 * NUM_BODY_MOTORS + 10)

        self.task = None
        for mod in getattr(args, "addons", []):
            if hasattr(mod, "make_task"):
                self.task = mod.make_task(m, ENV_PREFIX, args)
        self.last_success = False
        # Arm rest pose: held by the sim's own controller until a controller connects, and set on reset.
        self.arm_home = np.array(getattr(self.task, "arm_home", np.zeros(14)), dtype=float)

        self.sim_lock = threading.RLock()  # guards model/data: physics thread vs cameras, state publishing
        self.running = False
        self.dropped_steps = 0
        self.fixed = not args.free
        self.pending = queue.Queue()  # actions requested from other threads ("toggle", "reset", "reset_objects")
        self.reset(robot=True)

    # -------------------------------- DDS --------------------------------
    def start_dds(self):
        ChannelFactoryInitialize(self.args.domain, self.args.iface)
        self.crc = CRC()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.low_state.mode_machine = MODE_MACHINE_G1_29
        self.low_state_pub = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_pub.Init()
        self.sim_state_pub = ChannelPublisher("rt/sim_state", String_)
        self.sim_state_pub.Init()
        # Same topic/format as Unitree's Isaac sim (unitree_lerobot counts rewards[0] == 1 as success).
        self.rewards_pub = ChannelPublisher("rt/rewards_state", String_)
        self.rewards_pub.Init()
        self.low_cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_sub.Init(self._on_low_cmd, 10)
        self.arm_sdk_sub = ChannelSubscriber("rt/arm_sdk", LowCmd_)
        self.arm_sdk_sub.Init(self._on_arm_sdk, 10)
        self.loco_sub = ChannelSubscriber(LOCO_REQUEST_TOPIC, Request_)
        self.loco_sub.Init(self._on_loco_request, 10)
        self.reset_sub = ChannelSubscriber("rt/reset_pose/cmd", String_)
        self.reset_sub.Init(self._on_reset_cmd, 10)
        threading.Thread(target=self._low_state_loop, daemon=True).start()

    def _on_low_cmd(self, msg: LowCmd_):
        c = np.array([[mc.q, mc.dq, mc.kp, mc.kd, mc.tau] for mc in msg.motor_cmd[:NUM_BODY_MOTORS]]).T
        with self.cmd_lock:
            self.cmd[:] = c
            self.last_cmd_time = time.monotonic()

    def _on_arm_sdk(self, msg: LowCmd_):
        c = np.array([[mc.q, mc.dq, mc.kp, mc.kd, mc.tau] for mc in msg.motor_cmd[ARMS]]).T
        with self.cmd_lock:
            self.arm_sdk[:] = c
            self.arm_sdk_weight = float(np.clip(msg.motor_cmd[ARM_SDK_WEIGHT].q, 0.0, 1.0))
            self.last_arm_sdk_time = time.monotonic()

    def _on_loco_request(self, msg: Request_):
        api = msg.header.identity.api_id
        with self.loco_lock:
            if api == LOCO_API_SET_VELOCITY:
                p = json.loads(msg.parameter)
                self.vel_cmd[:] = p["velocity"]
                self.vel_cmd_until = time.monotonic() + float(p.get("duration", 1.0))
            elif api == LOCO_API_SET_FSM_ID:
                self.vel_cmd[:] = 0.0

    def _on_reset_cmd(self, msg: String_):
        print(f"[sim] rt/reset_pose/cmd '{msg.data}' -> resetting objects")
        self.pending.put("reset_objects")

    def _low_state_loop(self):
        n = NUM_BODY_MOTORS
        period = 0.002
        next_t = time.perf_counter()
        while True:
            with self.state_lock:
                s = self.state.copy()
            ls = self.low_state
            for i in range(n):
                ms = ls.motor_state[i]
                ms.mode = 1
                ms.q, ms.dq, ms.tau_est = float(s[i]), float(s[n + i]), float(s[2 * n + i])
            o = 3 * n
            ls.imu_state.quaternion = s[o:o + 4].tolist()
            ls.imu_state.gyroscope = s[o + 4:o + 7].tolist()
            ls.imu_state.accelerometer = s[o + 7:o + 10].tolist()
            ls.tick = (ls.tick + 1) & 0xFFFFFFFF
            ls.crc = self.crc.Crc(ls)
            self.low_state_pub.Write(ls)
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))
            if time.perf_counter() - next_t > 0.1:  # fell far behind (e.g. suspended): resync
                next_t = time.perf_counter()

    def publish_sim_state(self):
        d = self.data
        state = {"fixed_base": self.fixed, "rtf": round(getattr(self, "rtf", 1.0), 2)}
        for name in self.objects:
            b = d.body(name)
            state[name.removeprefix(ENV_PREFIX)] = {"pos": np.round(b.xpos, 4).tolist(), "quat": np.round(b.xquat, 4).tolist()}
        if self.task:
            state.update(self.task.object_poses(d))  # includes the glued (static) cardboard box
            success = self.task.success(d)
            state["success"] = success
            if success and not self.last_success:
                print("[sim] task success")
            self.last_success = success
            self.rewards_pub.Write(String_(data=json.dumps({"rewards": [int(success)]})))
        self.sim_state_pub.Write(String_(data=json.dumps(state)))

    # ------------------------------ physics ------------------------------
    def reset(self, robot=True):
        m, d = self.model, self.data
        if robot:
            mujoco.mj_resetData(m, d)
            d.qpos[7 + 15:7 + 29] = self.arm_home
            with self.loco_lock:
                self.base_yaw = self.waist_yaw = self.gait_amp = 0.0
            d.mocap_pos[self.gantry] = self.stand_pos
            d.mocap_quat[self.gantry] = [1, 0, 0, 0]
        if self.task:
            self.task.reset_objects(d, randomize_objects=getattr(self.task, "randomize", False))
        elif not robot:
            for name in self.objects:
                jnt = m.body(name).jntadr[0]
                qa, va = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
                d.qpos[qa:qa + 7] = m.qpos0[qa:qa + 7]
                d.qvel[va:va + 6] = 0.0
        self._apply_base_mode()
        mujoco.mj_forward(m, d)

    def _apply_base_mode(self):
        m = self.model
        self.data.eq_active[self.weld_id] = 1 if self.fixed else 0
        contype, conaffinity = self.leg_geom_contact
        m.geom_contype[self.leg_geoms] = 0 if self.fixed else contype
        m.geom_conaffinity[self.leg_geoms] = 0 if self.fixed else conaffinity

    def set_fixed(self, fixed):
        self.fixed = fixed
        d = self.data
        if fixed:
            # Hang the robot back on the gantry where it stands now, upright and at standing height.
            yaw = yaw_of(d.qpos[3:7])
            with self.loco_lock:
                self.base_yaw = yaw
            d.mocap_pos[self.gantry] = [d.qpos[0], d.qpos[1], self.stand_pos[2]]
            d.mocap_quat[self.gantry] = yaw_quat(yaw)
            d.qpos[0:3] = d.mocap_pos[self.gantry]
            d.qpos[3:7] = d.mocap_quat[self.gantry]
            d.qvel[0:6] = 0.0
        self._apply_base_mode()
        mujoco.mj_forward(self.model, d)
        print(f"[sim] base mode: {'FIXED pelvis' if fixed else 'FREE-standing'}")

    def loco_step(self):
        """Advance the emulated motion controller by one physics step (joystick walking + waist twist)."""
        dt, d = self.dt, self.data
        with self.loco_lock:
            v = self.vel_cmd.copy() if time.monotonic() < self.vel_cmd_until else np.zeros(3)
            vx, vy = v[0] * self.args.walk_scale, v[1] * self.args.walk_scale
            vyaw = v[2] * self.args.turn_scale
            if not self.fixed and (vx or vy) and not self.free_walk_warned:
                print("[sim] walking needs FIXED mode (press 9) - free-standing has no balance controller")
                self.free_walk_warned = True
            if not self.fixed:
                vx = vy = 0.0

            # Right stick twists the waist; past the limit the whole body turns (in FIXED mode).
            waist = self.waist_yaw + vyaw * dt
            overflow = waist - np.clip(waist, -WAIST_YAW_LIMIT, WAIST_YAW_LIMIT)
            waist -= overflow
            if self.fixed:
                self.base_yaw += overflow
            speed = np.hypot(vx, vy)
            if speed > 1e-3:  # while walking the pelvis turns toward where the torso looks
                follow = waist * min(1.0, BODY_FOLLOW_RATE * dt)
                self.base_yaw += follow
                waist -= follow
            self.waist_yaw = waist

            # Left stick walks in the torso's facing direction.
            heading = self.base_yaw + waist
            c, s = np.cos(heading), np.sin(heading)
            d.mocap_pos[self.gantry][:2] += np.array([c * vx - s * vy, s * vx + c * vy]) * dt
            d.mocap_quat[self.gantry] = yaw_quat(self.base_yaw)

            # Stepping animation (the gantry carries the weight, so this is visual only).
            activity = max(speed / 0.3, abs(overflow) / dt / 0.6)  # walking, or the whole body turning
            target_amp = min(1.0, activity) if activity > 0.05 else 0.0
            self.gait_amp += (target_amp - self.gait_amp) * min(1.0, 5.0 * dt)
            self.gait_phase = (self.gait_phase + 2 * np.pi * GAIT_FREQ / 2 * dt) % (2 * np.pi)
            q = self.legs_waist_q
            q[:] = 0.0
            for hip, phase in ((0, self.gait_phase), (6, self.gait_phase + np.pi)):
                lift = self.gait_amp * max(0.0, np.sin(phase))
                q[hip + 0] = -0.45 * lift   # hip pitch
                q[hip + 3] = 0.9 * lift     # knee
                q[hip + 4] = -0.45 * lift   # ankle pitch
            q[WAIST_YAW] = waist

    def apply_pd(self):
        d = self.data
        n = NUM_BODY_MOTORS
        q, dq = d.sensordata[:n], d.sensordata[n:2 * n]
        now = time.monotonic()
        with self.cmd_lock:
            if self.last_cmd_time is not None and now - self.last_cmd_time < CMD_TIMEOUT:
                # Debug mode: xr_teleoperate commands every joint.
                mode = "lowcmd"
                q_des, dq_des, kp, kd, tau_ff = self.cmd
                tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
            else:
                # Motion-controller mode: the sim owns legs and waist, arms are blended from rt/arm_sdk.
                q_des = np.zeros(n)
                q_des[:15] = self.legs_waist_q
                q_des[ARMS] = self.arm_home
                tau = HOLD_KP * (q_des - q) - HOLD_KD * dq
                arm_fresh = self.last_arm_sdk_time is not None and now - self.last_arm_sdk_time < CMD_TIMEOUT
                mode = "motion" if arm_fresh else None
                w = self.arm_sdk_weight if arm_fresh else 0.0
                if w > 0.0:
                    aq, adq, akp, akd, atau = self.arm_sdk
                    tau_sdk = atau + akp * (aq - q[ARMS]) + akd * (adq - dq[ARMS])
                    tau[ARMS] = w * tau_sdk + (1.0 - w) * tau[ARMS]
        if mode != self.mode:
            self.mode = mode
            print({"lowcmd": "[sim] driven by rt/lowcmd (xr_teleoperate debug mode)",
                   "motion": "[sim] driven by rt/arm_sdk + joysticks (xr_teleoperate --motion)",
                   None: "[sim] no commands - sim holds the robot"}[mode])
        d.ctrl[:] = np.clip(tau, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def snapshot_state(self):
        d, n = self.data, NUM_BODY_MOTORS
        with self.state_lock:
            self.state[:3 * n] = d.sensordata[:3 * n]
            o = 3 * n
            self.state[o:o + 4] = d.sensordata[self.imu_quat:self.imu_quat + 4]
            self.state[o + 4:o + 7] = d.sensordata[self.imu_gyro:self.imu_gyro + 3]
            self.state[o + 7:o + 10] = d.sensordata[self.imu_acc:self.imu_acc + 3]

    def handle_pending(self):
        while not self.pending.empty():
            action = self.pending.get_nowait()
            if action == "toggle":
                self.set_fixed(not self.fixed)
            elif action == "reset":
                self.reset(robot=True)
                print("[sim] reset robot and objects")
            elif action == "reset_objects":
                self.reset(robot=False)

    def key_callback(self, key):
        if key == KEY_TOGGLE_BASE:
            self.pending.put("toggle")
        elif key == KEY_RESET:
            self.pending.put("reset")

    # ------------------------------ main loop ------------------------------
    def run(self):
        args = self.args
        self.start_dds()
        cam = None if args.no_camera else CameraServer(self.model, args, self.cameras)
        print(f"[sim] DDS domain {args.domain} on '{args.iface}' | lowstate @500Hz | objects: "
              f"{[o.removeprefix(ENV_PREFIX) for o in self.objects]}")
        if args.addons:
            print(f"[sim] add-ons: {', '.join(a.__name__ for a in args.addons)}")
        if cam:
            print(f"[sim] image server: config tcp://*:{args.config_port + args.port_offset} | {cam.describe()} "
                  f"@ {args.fps:g} fps")
        print(f"[sim] base mode: {'FIXED pelvis' if self.fixed else 'FREE-standing'} | "
              "viewer keys: 9 = toggle fixed/free, 0 = reset")

        viewer = None
        if not args.no_viewer:
            viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=self.key_callback)
            with viewer.lock():
                for mod in getattr(args, "addons", []):
                    for group in getattr(mod, "VIEWER_GEOM_GROUPS", ()):  # e.g. a head hidden from the cameras
                        viewer.opt.geomgroup[group] = 1
            if cam:
                # The desktop viewer shares the iGPU with the headset stream: keep its rendering cheap.
                with viewer.lock():
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

        cam_period, viewer_period, state_period = 1.0 / args.fps, 1.0 / 60.0, 1.0 / 30.0
        t0 = time.perf_counter()
        last_cam = last_view = last_state = last_video = -1.0
        last_report, cam_frames = 0.0, 0
        video = None
        if args.debug_video:
            video_renderer = mujoco.Renderer(self.model, 480, 640)
            video_cam = mujoco.MjvCamera()
            view = {"lookat": [0.35, 0.0, 0.85], "distance": 1.8, "azimuth": 135, "elevation": -25}
            for mod in getattr(args, "addons", []):
                view.update(getattr(mod, "DEBUG_VIEW", {}))
            video_cam.lookat[:] = view["lookat"]
            video_cam.distance, video_cam.azimuth, video_cam.elevation = (view["distance"], view["azimuth"],
                                                                          view["elevation"])
            video = cv2.VideoWriter(args.debug_video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
        log = [] if args.log_qpos else None
        shown_fixed = None
        self.running = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: setattr(self, "running", False))
        physics = threading.Thread(target=self._physics_loop, args=(viewer,), daemon=True)
        physics.start()
        try:
            # This (main / GL) thread only renders and publishes; physics keeps real time on its own thread.
            while self.running and (viewer is None or viewer.is_running()):
                now = time.perf_counter() - t0
                if now - last_report >= 5.0:  # tell the user when the PC can't keep up (data would be off)
                    dropped, self.dropped_steps = self.dropped_steps, 0
                    rtf = 1.0 - dropped * self.dt / (now - last_report)
                    fps = (cam.frames - cam_frames) / (now - last_report) if cam else args.fps
                    if rtf < 0.95 or fps < 0.85 * args.fps:
                        print(f"[sim] WARNING: PC overloaded - physics at {rtf:.2f}x real time, cameras at {fps:.0f} fps "
                              f"(wanted {args.fps:g}). Close other programs or use --no-viewer.", flush=True)
                    self.rtf = rtf
                    last_report, cam_frames = now, cam.frames if cam else 0

                if cam and now - last_cam >= cam_period:
                    # Fixed timeline (not "now"), so being a few ms late doesn't lower the frame rate.
                    last_cam = last_cam + cam_period if now - last_cam < 2 * cam_period else now
                    cam.render(self.data, self.sim_lock)
                if video and now - last_video >= 1 / 15:
                    last_video = now
                    with self.sim_lock:
                        video_renderer.update_scene(self.data, camera=video_cam)
                    video.write(cv2.cvtColor(video_renderer.render(), cv2.COLOR_RGB2BGR))
                if now - last_state >= state_period:
                    last_state = last_state + state_period if now - last_state < 2 * state_period else now
                    with self.sim_lock:
                        self.publish_sim_state()
                        if log is not None:
                            d, m = self.data, self.model
                            log.append((now, d.qpos.copy(), d.mocap_pos.copy(), d.mocap_quat.copy(),
                                        m.body_pos.astype(np.float32), m.body_quat.astype(np.float32),
                                        self.last_success))
                if viewer and now - last_view >= viewer_period:
                    last_view = now
                    if shown_fixed != self.fixed:  # set_texts takes the viewer lock itself
                        shown_fixed = self.fixed
                        viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                          "FIXED pelvis" if self.fixed else "FREE-standing",
                                          "9: toggle base   0: reset"))
                    viewer.sync()  # takes the viewer lock, which the physics thread holds while it steps

                due = [last_state + state_period]
                if cam:
                    due.append(last_cam + cam_period)
                if viewer:
                    due.append(last_view + viewer_period)
                sleep = min(due) - (time.perf_counter() - t0)
                if sleep > 0:
                    time.sleep(min(sleep, 0.01))
        finally:
            self.running = False
            physics.join(timeout=1.0)
            if viewer:
                viewer.close()
            if video:
                video.release()
            if log:
                keys = ("t", "qpos", "mocap_pos", "mocap_quat", "body_pos", "body_quat", "success")
                np.savez_compressed(args.log_qpos, **{k: np.array(v) for k, v in zip(keys, zip(*log))})
                print(f"[sim] wrote {args.log_qpos} ({len(log)} samples)", flush=True)
            print("[sim] bye", flush=True)

    def _physics_loop(self, viewer):
        """Real-time physics and control on its own thread. Rendering three cameras takes tens of ms per
        frame on an integrated GPU; done on this thread too, that would hold the PD loop back in bursts and
        make the robot's behaviour depend on the camera load (the GL calls release the Python GIL)."""
        viewer_lock = viewer.lock if viewer else contextlib.nullcontext
        t0 = time.perf_counter()
        steps = 0
        while self.running:
            now = time.perf_counter() - t0
            budget = 50  # at most 0.1 s of sim time per pass, so a stall can't freeze everything else
            try:
                with self.sim_lock, viewer_lock():
                    self.handle_pending()
                    while steps * self.dt < now and budget > 0:
                        self.loco_step()
                        self.apply_pd()
                        mujoco.mj_step(self.model, self.data)
                        self.snapshot_state()
                        steps += 1
                        budget -= 1
            except Exception:
                if not self.running:  # the viewer was closed under us while shutting down
                    break
                raise
            if budget == 0:  # too far behind: drop the backlog instead of fast-forwarding
                self.dropped_steps += int(now / self.dt) - steps
                steps = int(now / self.dt)
            sleep = steps * self.dt - (time.perf_counter() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    G1XrSim(parse_args()).run()
    # teleimager's ZMQ contexts and the DDS threads would otherwise block interpreter shutdown.
    os._exit(0)
