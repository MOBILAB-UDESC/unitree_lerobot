# unitree_bridge: Unitree's cleanup + converter tools on our G1

Part of [stage 4](../README.md) (cleaning, way a). Reference for the wrapper in this folder.

Unitree's [unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot) (Apache-2.0, pins lerobot 0.4.1) ships the tools to clean up xr_teleoperate `--record` sessions and turn them into LeRobot datasets. Its converter only knows Unitree's own robot types (Dex1/Dex3/Inspire/Brainco hands, stereo heads). This folder registers ours at runtime and then runs their converter unchanged:

| robot_type | state / action | cameras |
|---|---|---|
| `Unitree_G1_ArmsOnly_3Cam` | 14 arm joints (`left_arm.qpos` + `right_arm.qpos`) | `cam_head`, `cam_left_wrist`, `cam_right_wrist` |
| `Unitree_G1_ArmsWaist_3Cam` | the 14 + `kWaistYaw` (`waist.qpos`) | same |

Mono head: `color_0/1/2` → head / left wrist / right wrist. Stereo head (`color_3` present): `color_0` (left eye) → head, `color_1` is dropped, `color_2/3` → wrists. Joint names and camera keys are the same as `g1lab.lerobot_writer`, so the result merges with the real `g1_pick_cube_real` set and the sim sets.

> **Status: untested.** unitree_lerobot is not installed on the dev PC. The wrapper is written against its documented API (`unitree_lerobot/utils/constants.py`: `RobotConfig`, `ROBOT_CONFIGS`; `convert_unitree_json_to_lerobot.json_to_lerobot(raw_dir, repo_id, robot_type, push_to_hub=False, mode="video")`). The registration and input checks are unit-tested with a stand-in module (`g1lab/tests/test_unitree_bridge.py`).

## Setup (separate env)

Don't install unitree_lerobot into `g1lab` or `g1sim`. Use its own env, following its README (it brings lerobot 0.4.1; the editor also needs PyQt5). `install/install.sh` already cloned it to `third_party/unitree_lerobot`:

```bash
cd third_party/unitree_lerobot
conda create -y -n unitree_lerobot python=3.10 && conda activate unitree_lerobot
pip install -e .            # see their README for the exact steps (lerobot submodule, ffmpeg)
pip install PyQt5
export PYTHONPATH=/path/to/unitree_lerobot/g1_mujoco_real2sim/04_recording   # for `python -m unitree_bridge.convert`
```

## Flow: data_editor → sort_and_rename → convert

xr_teleoperate `--record` writes `third_party/xr_teleoperate/teleop/utils/data/<task>/episode_XXXX/{data.json, colors/*.jpg}`. The commands below are run from `third_party/unitree_lerobot`, so that folder is `../xr_teleoperate/teleop/utils/data`.

1. **Review and clean up** with Unitree's editor. It plays back each recorded episode. Delete the failed ones and trim dead frames at the start and end:
   ```bash
   (cd data_editor && python data_editor_EN.py)      # PyQt5 GUI; "Select Dataset Path" -> the <task> folder
   ```
2. **Renumber** the episodes after deletions, so they are `episode_0000..N-1` again:
   ```bash
   python unitree_lerobot/utils/sort_and_rename_folders.py --data_dir ../xr_teleoperate/teleop/utils/data/<task>
   ```
3. **Convert** with our robot types. `--raw-dir` is the **parent** of the task folders, as upstream expects:
   ```bash
   python -m unitree_bridge.convert --raw-dir ../xr_teleoperate/teleop/utils/data \
       --repo-id local/pick_cube_teleop --robot-type Unitree_G1_ArmsOnly_3Cam --output-root ../../datasets
   ```
   - Upstream always writes `$HF_LEROBOT_HOME/<repo_id>` and **deletes it first**. `--output-root` points `HF_LEROBOT_HOME` elsewhere, and an existing output is refused unless you pass `--overwrite`.
   - The task string is each episode's `text.goal`.
   - The wrapper adds two things upstream lacks. It calls `finalize()`, without which lerobot 0.4.1 parquet files have no footer. It also uses the real dataset's chunk/file-size settings.
4. **Check / edit** with lerobot's own tools (in the g1lab env, from the repository folder):
   ```bash
   lerobot-dataset-viz --repo-id local/pick_cube_teleop --root datasets/local/pick_cube_teleop --episode-index 0
   HF_LEROBOT_HOME=$PWD/datasets lerobot-edit-dataset --repo_id local/pick_cube_teleop \
       --operation.type delete_episodes --operation.episode_indices "[3, 7]"
   ```
   Do not pass `--root` to `lerobot-edit-dataset`: with lerobot 0.4.1 it fails. The original is kept as `datasets/local/pick_cube_teleop_old`.

### The waist

With `--input-mode controller --motion`, the operator twists the waist with the right joystick. xr_teleoperate's recorder doesn't store that. With `--ee none` its `body` fields are empty, so its JSON has no `waist.qpos` and `Unitree_G1_ArmsWaist_3Cam` is refused with a hint. You have two options:
- **Arms only (waist stays still):** convert with `Unitree_G1_ArmsOnly_3Cam`. Then, if you want to mix with 15-joint data, pad with `python -m g1lab.pad_waist SRC DST` (waist = 0).
- **Waist-turning teleop:** record with `python -m g1lab.record_live` next to xr_teleoperate. It writes the 15-joint LeRobot dataset directly, so no JSON or conversion is involved. Clean it up with `lerobot-edit-dataset --operation.type delete_episodes` instead of the data_editor.

`Unitree_G1_ArmsWaist_3Cam` is for JSON that carries `states/actions.waist.qpos`. That is the key Unitree's own `*_UseWaist` configs read.

### Known upstream issues (Nov 2026)

- `eval_g1.py` doesn't work with the current image server. It uses the old image client API (issue #103), ignores `--root`, and splits the head image into halves. Use `05_training/run_act.py` instead.
- `lerobot-edit-dataset --operation.type merge` loads every source from the same `--root`. To merge datasets in different directories, use `05_training/merge_datasets.py`.
