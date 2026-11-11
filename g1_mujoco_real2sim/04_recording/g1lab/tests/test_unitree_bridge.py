"""unitree_bridge without unitree_lerobot: registration into a stand-in constants module + input checks.

    cd 04_recording && python -m pytest g1lab/tests/test_unitree_bridge.py -v -p no:cacheprovider
"""

import dataclasses
import json
import types

import pytest

from g1lab.lerobot_writer import CAMERAS, JOINT_NAMES, JOINT_NAMES_WAIST, ROBOT_TYPE, ROBOT_TYPE_WAIST
from unitree_bridge import configs, convert


@dataclasses.dataclass(frozen=True)
class RobotConfig:  # same fields as unitree_lerobot.utils.constants.RobotConfig
    motors: list
    cameras: list
    camera_to_image_key: dict
    json_state_data_name: list
    json_action_data_name: list


def fake_constants():
    return types.SimpleNamespace(RobotConfig=RobotConfig, ROBOT_CONFIGS={"Unitree_G1_Dex3": object()})


def test_register_adds_both_types_in_place():
    const = fake_constants()
    table = const.ROBOT_CONFIGS
    configs.register(constants=const)
    assert const.ROBOT_CONFIGS is table and "Unitree_G1_Dex3" in table
    arms, waist = table[ROBOT_TYPE], table[ROBOT_TYPE_WAIST]
    assert arms.motors == JOINT_NAMES and waist.motors == JOINT_NAMES_WAIST      # same as our writer
    assert arms.cameras == CAMERAS and set(arms.camera_to_image_key.values()) == set(CAMERAS)
    assert arms.camera_to_image_key["color_0"] == "cam_head"
    assert waist.json_state_data_name[-1] == "waist.qpos"


def test_register_stereo_map_drops_right_eye():
    const = fake_constants()
    configs.register(binocular=True, constants=const)
    m = const.ROBOT_CONFIGS[ROBOT_TYPE].camera_to_image_key
    assert m == {"color_0": "cam_head", "color_2": "cam_left_wrist", "color_3": "cam_right_wrist"}


def write_episode(path, n_colors=3, waist=False, arm_len=7):
    states = {"left_arm": {"qpos": [0.0] * arm_len}, "right_arm": {"qpos": [0.0] * arm_len},
              "left_ee": {"qpos": []}, "right_ee": {"qpos": []}, "body": {"qpos": []}}
    if waist:
        states["waist"] = {"qpos": [0.1]}
    frame = {"colors": {f"color_{i}": f"colors/{i}.jpg" for i in range(n_colors)}, "states": states,
             "actions": json.loads(json.dumps(states))}
    path.mkdir(parents=True)
    (path / "data.json").write_text(json.dumps({"text": {"goal": "pick up cube."}, "data": [frame]}))


def test_layout_and_checks(tmp_path):
    raw = tmp_path / "data"
    write_episode(raw / "pick_cube" / "episode_0000")
    write_episode(raw / "pick_cube" / "episode_0001", n_colors=4)
    eps = convert.find_episodes(raw)
    assert [e.name for e in eps] == ["episode_0000", "episode_0001"]
    with pytest.raises(SystemExit, match="PARENT"):
        convert.find_episodes(raw / "pick_cube")
    assert convert.check_episode(eps[0] / "data.json", ROBOT_TYPE) is False
    assert convert.check_episode(eps[1] / "data.json", ROBOT_TYPE) is True       # color_3 -> stereo head
    with pytest.raises(SystemExit, match="waist"):                              # xr_teleoperate JSON has no waist
        convert.check_episode(eps[0] / "data.json", ROBOT_TYPE_WAIST)
    with pytest.raises(SystemExit, match="color_2"):                            # a wrist camera missing
        convert.check_episode(_one(tmp_path / "two", n_colors=2), ROBOT_TYPE)


def _one(path, **kw):
    write_episode(path / "t" / "e", **kw)
    return path / "t" / "e" / "data.json"


def test_waist_json_accepted_and_bad_arm_count(tmp_path):
    assert convert.check_episode(_one(tmp_path / "w", waist=True), ROBOT_TYPE_WAIST) is False
    with pytest.raises(SystemExit, match="14"):
        convert.check_episode(_one(tmp_path / "a", arm_len=6), ROBOT_TYPE)
    bad = tmp_path / "broken.json"
    bad.write_text('{"data": [')
    with pytest.raises(SystemExit, match="crashed"):
        convert.check_episode(bad, ROBOT_TYPE)
