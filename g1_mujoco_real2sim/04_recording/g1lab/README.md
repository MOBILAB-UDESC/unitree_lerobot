# g1lab: recording library (reference)

`g1lab` is the Python package behind [stage 4](../README.md). It records Unitree G1 episodes as LeRobot v3.0 datasets in the **same format as our real-robot dataset**, so real and sim data can be mixed for training. This page is the reference; the step-by-step guide is [../README.md](../README.md).

## 1. Modules

| Module | What it is |
|---|---|
| [robot_io.py](robot_io.py) | `G1Arms`: one arms (+ waist) interface, identical for the sim and the real robot |
| [record_live.py](record_live.py) | Read-only 30 Hz recorder that runs next to any controller |
| [lerobot_writer.py](lerobot_writer.py) | `LeRobotWriter`: writes episodes in our dataset format |
| [convert_xr_json.py](convert_xr_json.py) | Converts `xr_teleoperate --record` folders to LeRobot |
| [pad_waist.py](pad_waist.py) | Pads a 14-joint dataset to 15 joints (waist = 0) |
| [check_dataset.py](check_dataset.py) | Prints episodes, episode lengths, joint ranges of motion, joints that never move and the share of repeated camera frames |
| [scripted/box_into_box.py](scripted/box_into_box.py) | Code-as-policy episode generator for the lab task |
| [remote_vla.py](remote_vla.py), [vla_mocks.py](vla_mocks.py) | Remote VLA client (dream / openpi / gr00t) and local mock servers |
| [mcp_server.py](mcp_server.py) | MCP server for AI assistants, see [MCP.md](MCP.md) |

## 2. Environment

```bash
install/setup_g1lab_env.sh
```

It builds the conda env `g1lab` (Python 3.10): lerobot 0.4.1 with CPU torch, pinocchio 3.1.0 and casadi, mujoco 3.17.0, opencv, mcp, and the editable `unitree_sdk2py` and `teleimager` from `third_party/`. It is safe to re-run. Run `install/install.sh` first (it makes the clones).

Activate it in each terminal, from the repository folder:

```bash
source scripts/env.sh g1lab
```

- `scripts/env.sh` also removes ROS 2 variables from the terminal (they break the env) and puts the stage folders on `PYTHONPATH`, so `python -m g1lab....` works.
- **numpy stays at 1.26.** conda-forge's pinocchio 3.1.0 crashes with numpy 2. `pip check` reports one `rerun-sdk` conflict; that is expected. Do not "fix" it by upgrading numpy.

## 3. Dataset format

| Item | Value |
|---|---|
| Format | LeRobot v3.0, 30 fps |
| `observation.state`, `action` | float32, 14 or 15 values, radians |
| `observation.images.cam_head`, `cam_left_wrist`, `cam_right_wrist` | 480x640 video, AV1 (`libsvtav1`), yuv420p |
| robot_type, 14 joints | `Unitree_G1_ArmsOnly_3Cam` (G1 motors 15-28: left arm, then right arm) |
| robot_type, 15 joints | `Unitree_G1_ArmsWaist_3Cam` (the 14 + `kWaistYaw`, motor 12) |

Joint order: `kLeftShoulderPitch, kLeftShoulderRoll, kLeftShoulderYaw, kLeftElbow, kLeftWristRoll, kLeftWristPitch, kLeftWristYaw`, then the same seven for the right arm, then `kWaistYaw` in the 15-joint layout.

## 4. G1Arms

```python
from g1lab.robot_io import G1Arms
robot = G1Arms("sim")                 # or "real" (DDS domain 0, image server on the robot's PC2)
obs = robot.observe()                 # {"state": (14,), "images": {cam_head, cam_left_wrist, cam_right_wrist}}
robot.command(q14)                    # arm joint targets
robot.move_to(q14, duration=2.0)      # smooth interpolated move
q14, tau = robot.ik(left_T, right_T)  # 4x4 wrist poses (robot frame) -> joints, xr_teleoperate's IK
left_T, right_T = robot.fk()          # measured wrist poses
robot.sim_state(); robot.success(); robot.reset_sim()   # sim only
```

- It uses xr_teleoperate's own arm controller and IK, so gains and limits match teleop.
- `G1Arms("sim", waist=True)` adds the waist: state and commands have 15 values, `command_waist(yaw)` turns the torso.
- A second sim started with `--domain 3 --port-offset 1000` is reached with `G1Arms("sim", domain=3, image_port=61000)`.
- Robot frame: origin at the pelvis, x forward, y left, z up, metres.

## 5. Live recorder

```bash
python -m g1lab.record_live --root datasets/teleop_waist --task "put the wrapped box inside the cardboard box"
```

Keys: **s** = start / stop and save, **d** = discard, **q** = quit.

Unattended, in sim: start on every sim reset request, save once the success flag has held, drop failed attempts:

```bash
python -m g1lab.record_live --root datasets/auto --auto-start --start-delay 2 --auto-success --only-success
```

What is stored at each 1/30 s tick:

| Field | Source |
|---|---|
| `observation.state` | Measured joints from `rt/lowstate` |
| `action`, arms | The command on the wire: `rt/lowcmd` if it is live (hand-tracking teleop, scripts, `run_act.py`), else `rt/arm_sdk` (controller-mode teleop). With no live controller, the measured pose. |
| `action`, waist | `rt/lowcmd` live: its waist target. `rt/arm_sdk` live: the waist **measured one tick later**, because in that mode the motion controller turns the waist from the joystick and no waist command exists on the wire. Override with `--waist-action`. |
| Images | The three 640x480 streams. A stereo head stream keeps its left eye. |

- It never publishes anything, so it cannot move the robot.
- Images are buffered as JPEG and written by a worker thread. Recording continues while the previous episode is encoded (15-20 s per episode).
- 15 joints by default; `--no-waist` writes 14.
- All options: `python -m g1lab.record_live --help`.

## 6. Writer

```python
from g1lab.lerobot_writer import LeRobotWriter

w = LeRobotWriter(root="datasets/my_set", repo_id="local/my_set", task="put the wrapped box inside the cardboard box", fps=30)
w.start_episode()                      # optional: start_episode(task="...") for this episode only
for each control tick:
    w.add_frame(state, action, {"cam_head": rgb, "cam_left_wrist": rgb, "cam_right_wrist": rgb})
w.save_episode()                       # or w.discard_episode()
w.close()                              # REQUIRED
```

- `LeRobotWriter(..., waist=True)` writes the 15-joint layout.
- **Images** must be RGB `uint8` with shape `(480, 640, 3)`, exactly the three camera keys. Anything else raises `ValueError`. Convert OpenCV's BGR first.
- **`save_episode()` blocks** while it encodes three videos (13-22 s for an 11 s episode on a laptop CPU). Do not call it inside a control loop.
- **`close()` is required.** A dataset that was never closed has unreadable parquet files. `with LeRobotWriter(...) as w:` calls it for you.
- **Append:** if `root` already holds a dataset, new episodes are appended after a compatibility check (fps, robot type, features).
- The writer is not thread-safe: one call at a time.

## 7. Converter: xr_teleoperate `--record` -> LeRobot

```bash
python -m g1lab.convert_xr_json \
    --raw-dir third_party/xr_teleoperate/teleop/utils/data/box_into_box \
    --root datasets/teleop_arms --repo-id local/teleop_arms \
    --task "put the wrapped box inside the cardboard box"
```

- `--raw-dir` is a task folder holding `episode_XXXX/data.json` and `colors/*.jpg`. A folder of task folders also works.
- If `--root` exists, episodes are appended.
- Output: 14 joints (`state = left_arm.qpos + right_arm.qpos`). Hand and body data are dropped, like in the real dataset.
- Cameras: `color_0` -> `cam_head`, `color_1` -> `cam_left_wrist`, `color_2` -> `cam_right_wrist`. If `color_3` exists the head was stereo: `color_0` (left eye) -> head, `color_2/3` -> wrists. Override with `--cam-map`.
- `--task` sets the task sentence for every episode. Without it each episode keeps its own `text.goal`.
- Robust to common damage: a `data.json` left open by a crash is repaired, a dropped camera frame is replaced by the previous one, episodes without a complete frame are skipped with a warning.

Unitree's own converter can be used instead, see [../unitree_bridge/README.md](../unitree_bridge/README.md).

## 8. Pad 14 -> 15 joints

```bash
python -m g1lab.pad_waist datasets/pick_cube_real datasets/real_waist
```

Appends `kWaistYaw = 0` to state and action, updates the statistics and metadata, copies the videos unchanged. Use it to merge the real data with sim data that moves the waist. Do not train a 15-joint policy on the padded set alone (the waist has zero variance).

## 9. Mixing with the real data

Use [../../05_training/merge_datasets.py](../../05_training/merge_datasets.py). It checks that fps, robot type and features are identical and then calls LeRobot's aggregator. See [stage 5](../../05_training/README.md).

## 10. Tests

```bash
source scripts/env.sh g1lab
python -m pytest 04_recording -q
```

- Need no sim: writer, converter, recorder logic, VLA client against the mocks, MCP server with a fake robot, unitree_bridge with a stand-in module.
- Skipped unless available: comparisons against the real dataset (expected at `datasets/pick_cube_real`, from the repository folder), and the tests that drive a running sim (they read `G1_MCP_TEST_DOMAIN` / `G1_MCP_TEST_IMAGE_PORT` and `G1LAB_SIM_DOMAIN` / `G1LAB_SIM_IMAGE_PORT`).

## Troubleshooting

- **`ModuleNotFoundError: No module named 'g1lab'`**: run `source scripts/env.sh g1lab` from the repository folder.
- **`_ARRAY_API not found` or a crash when importing pinocchio**: numpy 2 got installed. Run `pip install "numpy<2"`.
- **`PyAV build lacks libsvtav1`** at the end of the setup script: the env has a PyAV build without the AV1 encoder. Remove the env (`conda env remove -n g1lab`) and run `install/setup_g1lab_env.sh` again.
- **`cannot append to ...`**: the existing dataset has another joint layout or fps. Use a new `--root`, or the matching `--no-waist` setting.
- **`DDS already initialised as ...`**: one Python process can only join one DDS domain. Start a second process for a second sim.

More cases: [docs/TROUBLESHOOTING.md](../../docs/TROUBLESHOOTING.md).
