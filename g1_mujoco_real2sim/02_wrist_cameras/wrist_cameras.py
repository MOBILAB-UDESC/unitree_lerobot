"""Stage 2 add-on: wrist cameras on the G1 (g1_xr_sim.py --addon wrist_cameras).

Adds the hardware our real G1 carries on its wrists:
  * Unitree's 5010 wrist yaw links (unitree_ros g1_description) in place of the 4010 ones in the MJCF.
  * The official rubber hands wearing black knit gloves (assets/src/blender_build_wrist.py); they collide,
    with a grippy knit/rubber friction and torsional friction (the palm is a contact patch).
  * MOBILAB D405 wrist mounts (github.com/MOBILAB-UDESC/cad-models, unitree-g1/d405-wrist-mount-5010) with
    the ring clamp, each holding an Intel RealSense D405 (Intel's realsense2_description mesh).
  * A camera at each D405's colour imager: cam_left_wrist / cam_right_wrist, served as the teleimager
    topics left_wrist_camera / right_wrist_camera (ports 55556 / 55557).

The mounts and cameras are child bodies without joints, i.e. rigidly locked to the wrist yaw link, with the
real masses (bracket + clamp ~50 g, D405 ~60 g) and no contacts.
"""

import os

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, "assets", "mesh")
TEX_DIR = os.path.join(HERE, "assets", "tex")

WRIST_CAMERAS = {"left_wrist_camera": ["cam_left_wrist"], "right_wrist_camera": ["cam_right_wrist"]}

# --- wrist mount (frame definition from MOBILAB's d405-wrist-mount-5010/adapt_5010.py) ---
# Bracket frame B: ring axis = Z, ring u = X, ring v = Y (v > 0 is the camera side, on the yaw link's
# top cap = +z_yaw). Clamping band B.z 15..25 mm sits on the yaw link at x ~ 28..37 mm.
MOUNT_AXIAL_OFFSET = 0.013  # yaw_x = B.z + offset (right); left bracket is the right one mirrored in Z
# D405 pocket in the right bracket's frame, measured from its STL (wall planes): a1 = thickness axis
# (front window plate at 17.7 mm, back plate at 42.7), a2 = width (walls -25.3 / 17.7), a3 = height (top
# wall 113.2). The camera looks along -a1, its imager baseline runs along a2.
POCKET_A1 = np.array([-0.31, 0.423, -0.852])
POCKET_A2 = np.array([-0.94, 0.0, 0.342])
POCKET_CENTRE_MM = (30.2, -3.8, 92.2)  # along a1, a2, a3
D405_MESH = os.path.join(MESH_DIR, "d405_intel.stl")  # Intel realsense2_description d405.stl (mm, origin
# on the front glass between the imagers, looking along +z; colour = left imager, 9 mm along +x)
D405_HALF = np.array([0.021, 0.021, 0.0115])   # 42 x 42 x 23 mm

# Camera pose in the wrist yaw frame. Orientation self-calibrated from our real wrist videos with the
# datasheet focal length held fixed (03_real2sim/tools/fit_wrist_rotation.py: the wrist's rotation between
# frames is known from the recorded joints, so the background's image motion fixes the camera's
# orientation on the wrist; 1.6 px median error at 320x240). Optical axis in the wrist frame
# (0.88, -+0.18, -0.44): along the hand, ~26 deg down, ~10 deg inward. Position: the D405's colour imager
# seated in the MOBILAB pocket (POCKET_*).
WRIST_CAM_CALIB = {
    "left": dict(quat=[0.84175, -0.04287, -0.52785, -0.10491], pos=[0.0355, -0.0026, 0.0915]),
    "right": dict(quat=[0.84175, 0.04287, -0.52785, 0.10491], pos=[0.0355, 0.0026, 0.0915]),
}
# D405 datasheet: 87 x 58 deg (H x V) at 1280x720 from the 1280x800 sensor; the 640x480 stream is that
# sensor area x0.6 -> fx = fy ~ 385 px -> 63.9 deg vertical (79.4 horizontal), no distortion (RealSense
# 400 series). Replace with your unit's value from `rs-enumerate-devices -c` if you have it.
WRIST_CAM_FOVY = 63.9

GLOVE_FRICTION = [1.5, 0.06, 0.001]  # knit glove with rubber pads: sliding, torsional, rolling


def bracket_to_yaw(side):
    """Rotation (3x3) and translation mapping bracket coordinates into <side>_wrist_yaw_link."""
    if side == "right":   # x = Bz + c, y = Bx, z = By
        R = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
    else:                 # left bracket = right mirrored in Z: x = -Bz + c, y = -Bx, z = By
        R = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)
    return R, np.array([MOUNT_AXIAL_OFFSET, 0.0, 0.0])


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q


def build(spec, args, cameras, prefix):
    """Gloved 5010 hands + D405 mounts + wrist cameras on the robot spec."""
    spec.add_texture(name="glove_knit", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=os.path.join(TEX_DIR, "glove_knit.png"))
    glove = spec.add_material(name="glove", specular=0.05, shininess=0.1)
    glove.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "glove_knit"
    spec.add_material(name="pla_slate", rgba=[0.07, 0.07, 0.08, 1], specular=0.25)  # black PLA
    spec.add_material(name="d405_body", rgba=[0.10, 0.10, 0.11, 1], specular=0.5, shininess=0.6)

    for side in ("left", "right"):
        # 5010 wrist and gloved hand (the glove mesh replaces the bare rubber hand, for looks and contact).
        spec.mesh(f"{side}_wrist_yaw_link").file = os.path.join(MESH_DIR, f"{side}_wrist_yaw_link_5010.STL")
        spec.add_mesh(name=f"glove_{side}", file=os.path.join(MESH_DIR, f"glove_{side}.obj"))
        yaw = spec.body(f"{side}_wrist_yaw_link")
        for geom in yaw.geoms:
            if geom.meshname == f"{side}_rubber_hand":
                geom.meshname = f"glove_{side}"
                geom.material = "glove"
                geom.rgba = [0.5, 0.5, 0.5, 1]  # MuJoCo's default rgba -> the material colour is used
                geom.contype, geom.conaffinity = 1, 1
                geom.friction = GLOVE_FRICTION
                geom.condim = 4  # torsional friction: the palm is a contact patch, not a point

        # MOBILAB bracket (visual only, so it can't snag on the hand/forearm) with its real mass.
        spec.add_mesh(name=f"d405_mount_{side}", file=os.path.join(MESH_DIR, f"d405_mount_{side}.stl"))
        R, t = bracket_to_yaw(side)
        mount = yaw.add_body(name=f"{side}_d405_mount", pos=t, quat=mat_to_quat(R))
        mount.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"d405_mount_{side}", material="pla_slate",
                       contype=0, conaffinity=0, group=2, mass=0.05)

        # Intel's D405 mesh seated in the bracket pocket (left bracket = right mirrored in Z).
        mirror = np.diag([1.0, 1.0, -1.0]) if side == "left" else np.eye(3)
        a1 = mirror @ (POCKET_A1 / np.linalg.norm(POCKET_A1))
        a2 = mirror @ POCKET_A2
        a2 = a2 - (a2 @ a1) * a1
        a2 /= np.linalg.norm(a2)
        a3 = np.cross(a1, a2) if side == "right" else -np.cross(a1, a2)
        centre = (POCKET_CENTRE_MM[0] * a1 + POCKET_CENTRE_MM[1] * a2 + POCKET_CENTRE_MM[2] * a3) * 1e-3
        z = -a1                      # mesh +z: out of the front glass (the viewing direction)
        x = a2                       # mesh +x: along the imager baseline
        y = np.cross(z, x)
        glass = centre + z * D405_HALF[2]  # mesh origin is on the front glass
        cam_body = mount.add_body(name=f"{side}_d405", pos=glass, quat=mat_to_quat(np.stack([x, y, z], 1)))
        spec.add_mesh(name=f"d405_{side}", file=D405_MESH, scale=[0.001] * 3)
        cam_body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"d405_{side}", material="d405_body",
                          contype=0, conaffinity=0, group=2, mass=0.06)

        # The camera itself, in the wrist yaw frame.
        cal = WRIST_CAM_CALIB[side]
        c = yaw.add_camera()
        c.name = f"cam_{side}_wrist"
        c.pos, c.quat, c.fovy = cal["pos"], cal["quat"], WRIST_CAM_FOVY

    # The glove is ~3 cm in front of the lens, but MuJoCo's near clip plane is znear * model extent: pin both
    # so the gloves stay visible however big the scene is (5 mm near plane).
    spec.stat.extent = 2.5
    spec.stat.center = [0.4, 0.1, 0.9]
    spec.visual.map.znear = 0.002
    return {**cameras, **WRIST_CAMERAS}
