---
# LeRobot dataset card (Hub metadata spec: https://github.com/huggingface/hub-docs/blob/main/datasetcard.md)
license: apache-2.0   # TODO: confirm the license with the lab before publishing
task_categories:
- robotics
tags:
- LeRobot
- unitree
- g1
- teleoperation
- real-robot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# G1 pick cube (real robot, 14 arm joints)

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot) (format v3.0, lerobot 0.4.1).

Real Unitree G1 (29 DoF) teleoperated with [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) (XR controllers). The hands are fixed rubber hands with knit gloves, so there are no gripper channels. The episodes were recorded with xr_teleoperate `--record` and converted to LeRobot.

- **Task:** `pick up cube.`
- **Episodes / frames:** 52 / 17345 at 30 fps
- **robot_type:** `Unitree_G1_ArmsOnly_3Cam`
- **Joints (14):** `kLeftShoulderPitch, kLeftShoulderRoll, kLeftShoulderYaw, kLeftElbow, kLeftWristRoll, kLeftWristPitch, kLeftWristYaw`, then the same seven for the right arm. These are G1 motors 15–28.
- **Lab:** MOBILAB / UDESC (mobilab-udesc)

## Features

| key | type | shape | notes |
|---|---|---|---|
| `observation.state` | float32 | 14 | measured arm joint positions (rad) |
| `action` | float32 | 14 | commanded arm joint targets (rad, xr_teleoperate IK output) |
| `observation.images.cam_head` | video (AV1, yuv420p) | 480x640x3 | head camera (one 640x480 view) |
| `observation.images.cam_left_wrist` | video | 480x640x3 | left wrist Intel D405 |
| `observation.images.cam_right_wrist` | video | 480x640x3 | right wrist Intel D405 |

The waist was not moved in these episodes. `mobilab-udesc/g1_pick_cube_real_waist` is the same data padded to 15 joints (waist yaw = 0), so it can be mixed with waist-turning sim data.

## Usage

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("mobilab-udesc/g1_pick_cube_real")
```

Train ACT: `lerobot-train --dataset.repo_id=mobilab-udesc/g1_pick_cube_real --policy.type=act --policy.device=cuda`.

## Citation

[More Information Needed]
