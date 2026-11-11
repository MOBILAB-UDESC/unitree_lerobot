"""Our G1 robot types, in unitree_lerobot's RobotConfig terms (unitree_lerobot/utils/constants.py).

    from unitree_bridge.configs import register
    register()          # adds Unitree_G1_ArmsOnly_3Cam / Unitree_G1_ArmsWaist_3Cam to ROBOT_CONFIGS

The robot: G1-29 with fixed rubber hands (no grippers -> no left_ee/right_ee), a mono head camera and two
wrist cameras, recorded by xr_teleoperate --record with --ee none. Joint order / names / camera keys match
g1lab.lerobot_writer, so a converted dataset merges with the real pick_cube set and the g1lab sim sets.

Camera map (xr_teleoperate's colors keys -> dataset image keys):
  mono head  (3 colors):  color_0 head, color_1 left wrist, color_2 right wrist
  stereo head (4 colors): color_0 = left eye -> head, color_1 (right eye) dropped, color_2/3 wrists
The stereo layout is registered under the same robot_type (select it with binocular=True); the dataset
format is the same, only the source keys differ.

ArmsWaist reads states/actions "waist.qpos" - the key Unitree's own waist configs (Unitree_G1_*_UseWaist)
use. xr_teleoperate as used here does NOT write it (see g1lab/record_live.py), so ArmsWaist only applies to
JSON that carries that key; for plain xr_teleoperate JSON convert with ArmsOnly and pad with g1lab.pad_waist.
"""

JOINT_NAMES = [
    f"k{side}{joint}"
    for side in ("Left", "Right")
    for joint in ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow", "WristRoll", "WristPitch", "WristYaw")
]
CAMERAS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
CAM_MAP_MONO = {"color_0": "cam_head", "color_1": "cam_left_wrist", "color_2": "cam_right_wrist"}
CAM_MAP_STEREO = {"color_0": "cam_head", "color_2": "cam_left_wrist", "color_3": "cam_right_wrist"}
ARMS_JSON = ["left_arm.qpos", "right_arm.qpos"]

ARMS_ONLY = "Unitree_G1_ArmsOnly_3Cam"
ARMS_WAIST = "Unitree_G1_ArmsWaist_3Cam"


def config_fields(binocular=False):
    """{robot_type: RobotConfig kwargs} - plain dicts, so they can be checked without unitree_lerobot."""
    cam_map = CAM_MAP_STEREO if binocular else CAM_MAP_MONO
    return {
        ARMS_ONLY: dict(motors=list(JOINT_NAMES), cameras=list(CAMERAS), camera_to_image_key=dict(cam_map),
                        json_state_data_name=list(ARMS_JSON), json_action_data_name=list(ARMS_JSON)),
        ARMS_WAIST: dict(motors=JOINT_NAMES + ["kWaistYaw"], cameras=list(CAMERAS),
                         camera_to_image_key=dict(cam_map),
                         json_state_data_name=ARMS_JSON + ["waist.qpos"],
                         json_action_data_name=ARMS_JSON + ["waist.qpos"]),
    }


def register(binocular=False, constants=None):
    """Insert our configs into unitree_lerobot's ROBOT_CONFIGS dict (in place, at runtime; no file edits).

    The converter does `from unitree_lerobot.utils.constants import ROBOT_CONFIGS`, i.e. it holds the same
    dict object, so mutating it is enough. `constants` lets tests pass a stand-in module.
    """
    if constants is None:
        from unitree_lerobot.utils import constants
    RobotConfig = constants.RobotConfig
    added = {}
    for name, fields in config_fields(binocular).items():
        constants.ROBOT_CONFIGS[name] = RobotConfig(**fields)
        added[name] = constants.ROBOT_CONFIGS[name]
    return added
