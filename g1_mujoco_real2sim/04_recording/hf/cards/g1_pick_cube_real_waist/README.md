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

# G1 pick cube (real robot, padded to 15 joints)

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot) (format v3.0, lerobot 0.4.1).

This is the same 52 real episodes as `mobilab-udesc/g1_pick_cube_real`, in the 15-joint `Unitree_G1_ArmsWaist_3Cam` layout. `kWaistYaw` is appended to `observation.state` and `action` with the value **0**. That is exact, because the waist did not move in these recordings. The videos are unchanged. The conversion was done with `g1lab.pad_waist`, which also pads the per-episode and global stats.

The point of this layout is to mix the real data with sim episodes that do turn the waist (for example `mobilab-udesc/g1_box_into_box_sim_scripted`), using lerobot's `aggregate_datasets`. That function needs identical features.

- **Task:** `pick up cube.`
- **Episodes / frames:** 52 / 17345 at 30 fps
- **robot_type:** `Unitree_G1_ArmsWaist_3Cam`
- **Joints (15):** the 14 arm joints (left, then right: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw) + `kWaistYaw` (G1 motor 12)

## Features

| key | type | shape | notes |
|---|---|---|---|
| `observation.state` | float32 | 15 | measured joint positions (rad); waist = 0 |
| `action` | float32 | 15 | commanded joint targets (rad); waist = 0 |
| `observation.images.cam_head` | video (AV1, yuv420p) | 480x640x3 | head camera |
| `observation.images.cam_left_wrist` | video | 480x640x3 | left wrist Intel D405 |
| `observation.images.cam_right_wrist` | video | 480x640x3 | right wrist Intel D405 |

**Don't train a 15-joint policy on this set alone.** The waist std in `meta/stats.json` is 0. lerobot normalizes the state as `(x - mean) / (std + 1e-8)`, so at run time any small measured waist offset becomes a huge input. Use the 14-joint set for real-only training, and use this one only mixed with data that moves the waist. `05_training/run_act.py` warns when a checkpoint has such constant joints.

## Citation

[More Information Needed]
