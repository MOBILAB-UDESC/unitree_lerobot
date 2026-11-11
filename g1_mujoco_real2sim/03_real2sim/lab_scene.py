"""Stage 3 add-on: the real2sim lab scene (g1_xr_sim.py --scene lab, i.e. --addon lab_scene).

Our lab's desk task, rebuilt from measurements, photos and the real recordings: put the bubble-wrapped box
into the cardboard box glued to the desk.
  * scene_lab.xml: desk, cardboard box (static), wrapped box (free body), floor; scene_room.xml: the room
    around it (workbench + printers, cabinet, walls; assets/lab/make_room.py).
  * cam_head: mono head camera at the G1's D435 mount, pose and FOV calibrated against the real videos.
  * Uses the wrist_cameras add-on (gloved 5010 hands + D405 wrist cameras).
  * Photometric calibration (calib/photometric.json): material colours, light level, per-camera gain,
    plus lens blur and sensor noise like the real 640x480 streams.
  * LabTask: object start poses (randomised with --randomize), success check (rt/rewards_state).

Camera names match the real dataset keys (observation.images.cam_head / cam_left_wrist / cam_right_wrist).
"""

import json
import os
import tempfile

import cv2
import mujoco
import numpy as np
from wrist_cameras import mat_to_quat

REQUIRES = ["wrist_cameras"]
PROVIDES_SCENE = True

HERE = os.path.dirname(os.path.abspath(__file__))
LAB_XML = os.path.join(HERE, "scene_lab.xml")
MESH_DIR = os.path.join(HERE, "assets", "lab", "mesh")

# Photometric calibration against the real recordings (tools/fit_materials.py): material colours, light
# level and per-camera RGB gain. Loaded when present.
PHOTOMETRIC_JSON = os.path.join(HERE, "calib", "photometric.json")
PHOTOMETRIC = json.load(open(PHOTOMETRIC_JSON)) if os.path.exists(PHOTOMETRIC_JSON) else {}
# Image softness and sensor noise of the real streams (640x480): Gaussian blur sigma [px] and noise std
# [0-255] per camera, chosen so the sim frames have the real frames' edge sharpness and noise level
# (measured on the recordings: head 136 / 0.8, left wrist 154 / 0.9, right wrist 127 / 1.0).
CAMERA_BLUR = {"cam_head": 0.75, "cam_left_wrist": 0.25, "cam_right_wrist": 0.6}
CAMERA_NOISE = {"cam_head": 1.0, "cam_left_wrist": 0.5, "cam_right_wrist": 1.2}

HEAD_GEOM_GROUP = 5  # robot head shell, pelvis and legs: hidden from the cameras, shown in the viewer
VIEWER_GEOM_GROUPS = (HEAD_GEOM_GROUP,)
DEBUG_VIEW = {"lookat": [0.35, 0.12, 0.85], "distance": 1.5, "azimuth": 135, "elevation": -25}

# Image streams served to xr_teleoperate / recorders: teleimager topic -> MuJoCo camera names.
LAB_CAMERAS = {
    "head_camera": ["cam_head"],
    "left_wrist_camera": ["cam_left_wrist"],
    "right_wrist_camera": ["cam_right_wrist"],
}

# --- head camera: the G1's head D435i (g1 URDF: d435_joint xyz 0.0576235 0.01753 0.42987, rpy 0 0.8307767 0).
# FOV from the datasheet: RGB 69 x 42 deg at 16:9; the 640x480 stream has fx ~ fy ~ 610 px -> 42.9 deg
# vertical (55.3 horizontal). Pose corrected against the 52 real recordings with each episode's wrapped
# box at its kinematic position and the FOV held fixed (tools/fit_head_full.py, FIXED_FOVY=42.9:
# wrap + desk + glove IoU): 3 cm back, 3 cm up, pitched 52.5 deg (URDF 47.6) - plausibly the torso
# posture while the robot hangs on its gantry.
HEAD_CAM_POS = [0.0576235 - 0.03, 0.01753, 0.42987 + 0.03]
HEAD_CAM_PITCH_DEG = 52.5
HEAD_CAM_FOVY = 42.9

def look_rotation(forward, up_hint, roll_deg=0.0):
    """Frame whose -z looks along `forward` (MuJoCo camera convention), y ~ up_hint, then rolled."""
    z = -forward / np.linalg.norm(forward)
    x = np.cross(up_hint, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    c, s = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    return R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def head_camera_quat(pitch_deg):
    th = np.radians(pitch_deg)
    forward = np.array([np.cos(th), 0.0, -np.sin(th)])
    return mat_to_quat(look_rotation(forward, np.array([0.0, 0.0, 1.0])))


def _abs_assets(env, scene_dir):
    """Make mesh/texture paths absolute so they survive attaching into the robot spec."""
    mesh_dir = os.path.join(scene_dir, env.meshdir)
    tex_dir = os.path.join(scene_dir, env.texturedir)
    for m in env.meshes:
        m.file = os.path.join(mesh_dir, m.file)
    for t in env.textures:
        if t.file:
            t.file = os.path.join(tex_dir, t.file)
    env.meshdir = env.texturedir = ""


def add_args(p):
    p.add_argument("--head-fovy", type=float, default=None, help="lab: override the head camera vertical FOV")
    p.add_argument("--randomize", action="store_true",
                   help="lab: random object start poses + lighting/tint/camera jitter on every reset (sim2real)")
    p.add_argument("--seed", type=int, default=None, help="lab: random seed for --randomize")


def build(spec, args, cameras, env_prefix):
    """Attach the lab environment, the calibrated mono head camera and the photometric calibration."""
    env = mujoco.MjSpec.from_file(LAB_XML)
    _abs_assets(env, HERE)
    spec.attach(env, prefix=env_prefix, frame=spec.worldbody.add_frame())
    for attr in ("ambient", "diffuse", "specular"):  # the lab's lighting (attach keeps the robot file's <visual>)
        setattr(spec.visual.headlight, attr, getattr(env.visual.headlight, attr))
    spec.visual.rgba.haze = env.visual.rgba.haze
    spec.visual.quality.shadowsize = env.visual.quality.shadowsize
    apply_photometric(spec)

    # The real head camera looks out through the visor; Unitree's head mesh is a solid shell that the
    # camera (at its calibrated pose) would see from inside. Put it in render group 5: cameras use the
    # default groups 0-2, the desktop viewer turns group 5 on (VIEWER_GEOM_GROUPS).
    for geom in spec.body("torso_link").geoms:
        if geom.meshname == "head_link":
            geom.group = HEAD_GEOM_GROUP
    # Same group for the pelvis and legs: behind the desk no camera sees them, and they are 43 % of the
    # triangles the three cameras would draw 30 times a second (the difference between 20 and 30 fps on an
    # integrated GPU). Their collision geoms are untouched.
    for body in spec.bodies:
        if body.name == "pelvis" or any(k in body.name for k in ("hip", "knee", "ankle")):
            for geom in body.geoms:
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH and geom.contype == 0 and geom.conaffinity == 0:
                    geom.group = HEAD_GEOM_GROUP

    # Head camera (mono, like the real recordings) replaces the default stereo pair.
    cam = spec.body("torso_link").add_camera()
    cam.name = "cam_head"
    cam.pos = HEAD_CAM_POS
    cam.quat = head_camera_quat(HEAD_CAM_PITCH_DEG)
    cam.fovy = getattr(args, "head_fovy", None) or HEAD_CAM_FOVY
    return {**cameras, **LAB_CAMERAS}


def make_task(model, prefix, args):
    return LabTask(model, prefix, getattr(args, "randomize", False), getattr(args, "seed", None))


# ----------------------------------------------------------------------------------------------
# Task: put the bubble-wrapped box into the cardboard box
# ----------------------------------------------------------------------------------------------
WRAP = "wrapped_box"
CARDBOARD = "cardboard_box"
BOX_INNER_HALF = np.array([0.161, 0.241])  # inner half-extents of the cardboard box (x, y in its frame)
BOX_OUTER_HALF = np.array([0.165, 0.245])
WRAP_HALF = 0.117  # wrapped box half width incl. the bubble-wrap lumps
# Start-pose ranges (x, y, yaw deg) around the nominal poses in scene_lab.xml. Wrapped box: ~1.5 std of
# the grasp positions in the real episodes (from forward kinematics at the squeeze).
OBJECT_RANGES = {
    WRAP: dict(dx=(-0.05, 0.03), dy=(-0.035, 0.02), dyaw=(-12, 12)),
    CARDBOARD: dict(dx=(-0.01, 0.01), dy=(-0.01, 0.01), dyaw=(-2, 2)),  # glued; small sim2real jitter only
}
MIN_GAP = 0.005  # keep the wrapped box's left face this far right of the cardboard box's right wall


# Arm rest pose: the mean first-frame observation.state of the 52 real episodes (ARM_START_REAL_MEAN,
# std ~0.1 rad) with both hands pulled ~8 cm back (+2 cm up, 3 cm outward) by IK, same orientation:
# at the exact mean pose the gloved hands sit inside the wrapped box / on the cardboard wall (on the
# real robot the operator places the objects around the hands); pulled back, no layout is disturbed.
ARM_START_REAL_MEAN = [0.188, -0.055, -0.177, -0.164, 0.069, -0.289, 0.079,
                       0.155, 0.026, 0.031, -0.247, -0.160, -0.074, -0.143]
ARM_START = [0.611, 0.042, -0.177, -0.544, -0.025, -0.33, -0.028,
             0.56, -0.026, -0.031, -0.607, -0.029, -0.123, -0.011]


class LabTask:
    """Success check and (domain-)randomised resets for the lab scene."""

    arm_home = ARM_START  # held by the sim until a controller connects, and set on reset

    def __init__(self, model, prefix, randomize=False, seed=None):
        self.m, self.prefix, self.randomize = model, prefix, randomize
        self.rng = np.random.default_rng(seed)
        self.wrap = model.body(prefix + WRAP).id
        self.box = model.body(prefix + CARDBOARD).id
        self.box_pos0 = model.body_pos[self.box].copy()
        # Nominal values to randomise around.
        self.light_diffuse0 = model.light_diffuse.copy()
        self.light_dir0 = model.light_dir.copy()
        self.mat_rgba0 = model.mat_rgba.copy()
        self.cam_pos0 = model.cam_pos.copy()
        self.cam_quat0 = model.cam_quat.copy()

    def _free_qpos(self, body):
        jnt = self.m.body_jntadr[body]
        return self.m.jnt_qposadr[jnt], self.m.jnt_dofadr[jnt]

    def reset_objects(self, data, randomize_objects=True):
        """Wrapped box: free body. Cardboard box: glued to the desk (static body), placed via body_pos."""
        rng, m = self.rng, self.m

        def sample(name):
            r = OBJECT_RANGES[name]
            return (rng.uniform(*r["dx"]), rng.uniform(*r["dy"]), np.radians(rng.uniform(*r["dyaw"])))

        # Cardboard box (static): move the body itself.
        bx, by, byaw = sample(CARDBOARD) if randomize_objects else (0.0, 0.0, 0.0)
        m.body_pos[self.box] = self.box_pos0 + [bx, by, 0.0]
        m.body_quat[self.box] = [np.cos(byaw / 2), 0, 0, np.sin(byaw / 2)]

        qa, va = self._free_qpos(self.wrap)
        data.qpos[qa:qa + 7] = m.qpos0[qa:qa + 7]
        data.qvel[va:va + 6] = 0.0
        if randomize_objects:
            wx, wy, wyaw = sample(WRAP)
            data.qpos[qa] += wx
            data.qpos[qa + 1] += wy
            data.qpos[qa + 3:qa + 7] = [np.cos(wyaw / 2), 0, 0, np.sin(wyaw / 2)]
            # The two boxes start touching: never let the samples overlap (incl. both yaws).
            wrap_half_y = WRAP_HALF * (abs(np.cos(wyaw)) + abs(np.sin(wyaw)))
            box_half_y = BOX_OUTER_HALF[1] * abs(np.cos(byaw)) + BOX_OUTER_HALF[0] * abs(np.sin(byaw))
            limit = m.body_pos[self.box][1] - box_half_y - wrap_half_y - MIN_GAP
            data.qpos[qa + 1] = min(data.qpos[qa + 1], limit)
        if self.randomize:
            self.randomize_appearance()

    def object_poses(self, data):
        """World poses of the task objects, for rt/sim_state."""
        return {name: {"pos": np.round(data.xpos[b], 4).tolist(), "quat": np.round(data.xquat[b], 4).tolist()}
                for name, b in ((WRAP, self.wrap), (CARDBOARD, self.box))}

    def randomize_appearance(self):
        """Sim2real domain randomisation: lights, material tints, small camera pose jitter."""
        m, rng = self.m, self.rng
        m.light_diffuse[:] = self.light_diffuse0 * rng.uniform(0.75, 1.25, (m.nlight, 1))
        d = self.light_dir0 + rng.normal(0, 0.15, self.light_dir0.shape)
        m.light_dir[:] = d / np.linalg.norm(d, axis=1, keepdims=True)
        tint = rng.uniform(0.9, 1.1, (m.nmat, 3))
        m.mat_rgba[:, :3] = np.clip(self.mat_rgba0[:, :3] * tint, 0, 1)
        m.cam_pos[:] = self.cam_pos0 + rng.normal(0, 0.005, self.cam_pos0.shape)
        for i in range(m.ncam):
            axis = rng.normal(size=3)
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), np.radians(rng.normal(0, 1.0)))
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, self.cam_quat0[i], dq)
            m.cam_quat[i] = q

    def success(self, data):
        """Wrapped box centre inside the cardboard box walls, resting low (it may lean on a wall), still."""
        R = data.xmat[self.box].reshape(3, 3)
        centre = data.xpos[self.wrap] + data.xmat[self.wrap].reshape(3, 3)[:, 2] * 0.135  # body origin = bottom face
        rel = R.T @ (centre - data.xpos[self.box])  # box origin = its floor
        inside = np.all(np.abs(rel[:2]) < BOX_INNER_HALF)
        low = rel[2] < 0.22  # upright centre is 0.135-0.14 above the floor; leaning on the rim is still in
        _, va = self._free_qpos(self.wrap)
        still = np.linalg.norm(data.qvel[va:va + 3]) < 0.05
        return bool(inside and low and still)


def apply_photometric(spec):
    """Material colours, texture gains and light level fitted to the real recordings (calib/photometric.json)."""
    if not PHOTOMETRIC:
        return
    for name, rgba in PHOTOMETRIC.get("materials", {}).items():
        try:
            spec.material(name).rgba = rgba
        except Exception:
            pass
    # Textures that are too dark even at rgba = 1 get brightened themselves: builtin ones through their
    # colours, image files through a scaled copy in a cache folder.
    for name, gain in PHOTOMETRIC.get("texture_gain_rgb", {}).items():
        try:
            tex = spec.texture(spec.material(name).textures[mujoco.mjtTextureRole.mjTEXROLE_RGB])
        except Exception:
            continue
        if tex is None:
            continue
        gain = np.asarray(gain, float)
        if tex.file:
            img = cv2.imread(tex.file, cv2.IMREAD_COLOR)
            out = os.path.join(tempfile.gettempdir(), "g1_lab_tex_cache", name.replace("/", "_") + ".png")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            cv2.imwrite(out, np.clip(img * gain[::-1], 0, 255).astype(np.uint8))
            tex.file = out
        else:
            for attr in ("rgb1", "rgb2", "markrgb"):
                setattr(tex, attr, np.clip(np.asarray(getattr(tex, attr)) * gain, 0, 1))
    k = PHOTOMETRIC.get("light_scale", 1.0)
    for light in spec.lights:
        light.diffuse = np.asarray(light.diffuse) * k
    hl = spec.visual.headlight
    hl.ambient, hl.diffuse = np.asarray(hl.ambient) * k, np.asarray(hl.diffuse) * k


_NOISE_BANK = {}


def _noise(shape, rng):
    """Unit-variance sensor noise from a small pre-generated bank (drawing 0.9 M normals per frame and
    camera is too slow for 3 cameras at 30 Hz); a random bank entry, randomly flipped."""
    bank = _NOISE_BANK.get(shape)
    if bank is None:
        bank = _NOISE_BANK[shape] = np.random.default_rng(0).standard_normal((8, *shape), dtype=np.float32)
    n = bank[rng.integers(len(bank))]
    return n[::-1] if rng.integers(2) else n


def camera_effects(rgb, cam_name, rng=np.random.default_rng()):
    """Per-camera gain (exposure / white balance), lens softness and sensor noise, like the real streams."""
    if cam_name not in CAMERA_BLUR:
        return rgb
    out = rgb.astype(np.float32)
    gain = PHOTOMETRIC.get("camera_gain_rgb", {}).get(cam_name)
    if gain:
        out *= np.asarray(gain, np.float32)
    sigma = CAMERA_BLUR.get(cam_name, 0.0) * rgb.shape[1] / 640.0
    if sigma > 0.2:
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    noise = CAMERA_NOISE.get(cam_name, 0.0)
    if noise:
        out += noise * _noise(out.shape, rng)
    return np.clip(out, 0, 255).astype(np.uint8)
