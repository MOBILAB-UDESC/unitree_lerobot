# Troubleshooting

All known problems in one place, by area. Each stage guide repeats its own most common ones. Back to the [main README](../README.md).

First checks, for any problem:

- Are you in the repository folder?
- Did you activate the env in **this** terminal? (`source scripts/env.sh g1sim` or `source scripts/env.sh g1lab`; the `scripts/run_*.sh` scripts do it themselves.)
- Is the simulator running, and only **one** program commanding the robot?

## 1. Install and environments

| Problem | Fix |
|---|---|
| `Could not find conda environment: g1sim` | `install/install.sh` did not finish. Run it again (safe). If your env has another name: `G1_ENV=<name> scripts/run_sim.sh`. |
| `G1 model not found at .../third_party/unitree_mujoco/...` | The clones are missing. Run `install/install.sh`, or set `G1_THIRD_PARTY` to where they are. |
| `cannot import name 'Flag' from 'params_proto'` | `source scripts/env.sh g1sim && pip install "params-proto==2.13.2"` |
| Python picks up ROS 2 packages, pinocchio fails to import, DDS behaves oddly | ROS 2 is sourced in `~/.bashrc`. Always enter the env with `source scripts/env.sh <env>`: it removes the ROS 2 variables from that terminal. Plain `conda activate` does not. |
| `_ARRAY_API not found`, or a crash when importing pinocchio in `g1lab` | numpy 2 got installed. `pip install "numpy<2"` |
| pip reports a `rerun-sdk` / numpy conflict in `g1lab` | Expected. Do not upgrade numpy. |
| `setup_g1lab_env.sh` fails at `pip install -e .../unitree_sdk2_python` | Run `install/install.sh` first; it makes the clones. |
| `ModuleNotFoundError: No module named 'g1lab'` (or `g1_xr_sim`, `lab_scene`) | `source scripts/env.sh g1lab` from the repository folder. It sets `PYTHONPATH`. |
| `pytest: No module named pytest` in `g1sim` | `pip install pytest` |

## 2. Simulator

| Problem | Fix |
|---|---|
| No MuJoCo window | You passed `--no-viewer`, or there is no display (SSH). Use a local desktop session. |
| The robot fell over | You are in FREE mode. Press **9** to hang it back on the gantry, **0** to reset. |
| The robot does not walk | Walking needs FIXED mode (key **9**) and controller-mode teleop. Free-standing has no balance controller. |
| `[sim] walking needs FIXED mode (press 9)` | Same. |
| `Address already in use` at start | Another sim is running. Stop it (`scripts/pykill.sh g1_xr_sim.py --list` shows them), or start this one with `--domain N --port-offset M`. |
| `--randomize` is "unrecognized" | It belongs to the lab scene. Use it with `--scene lab`. |
| `--help` does not list `--randomize`, `--seed`, `--head-fovy` | Add the scene or add-on to the command: `scripts/run_sim.sh --scene lab --help` lists the add-on options too. |
| Objects do not reset | Key **0** in the MuJoCo window (click it first). Only bodies with a `<freejoint/>` are objects. |
| Low frame rate | `--no-viewer`, `--camera-shadows none`, or smaller images: `--img-w 480 --img-h 360` (demo scene). |
| `[sim] WARNING: PC overloaded - physics at 0.9x real time, cameras at 20 fps` | The PC cannot keep up, so recorded data would have slow physics or repeated frames. Close other programs, use `--no-viewer` and `--camera-shadows none`, and never run two simulators that render cameras on one integrated GPU. A few warnings while an episode is being saved (video encoding) are harmless. |
| Everything is slow and `top` shows many Python threads at 100 % | Start programs through `scripts/env.sh` (all `scripts/*.sh` do). It limits each process to one math thread; otherwise every process starts one thread per core and they starve the simulator. |

## 3. Quest 3 and teleop

| Problem | Fix |
|---|---|
| The Quest cannot open the link | Quest and PC must be on the same Wi-Fi. Open the ports: `sudo ufw allow 8012 && sudo ufw allow 8080`. Some guest or campus Wi-Fi networks block device-to-device traffic: use another network or a phone hotspot. |
| Certificate warning | Expected (self-signed). **Advanced** -> **Proceed**. |
| The page opens but stays black or empty | Use the link `run_teleop.sh` printed just now; the PC's IP may have changed. Check that the sim is running. |
| No "Virtual Reality" button, or it does nothing | VR only works on the HTTPS page. Use the short `:8080` link and accept the certificate. |
| Teleop waits at `Waiting to subscribe dds` | Start `scripts/run_sim.sh` first. Same `--domain` on both sides (default 1). |
| The robot does not follow | Click the teleop terminal and press **r**. |
| Arms jump when I press **r** | Hold your arms like the robot's arms before pressing **r**. |
| Teleop crashed when I clicked both thumbsticks | Known. That is the soft emergency stop of the real robot; the pinned xr_teleoperate commit crashes on it. Restart `run_teleop.sh`. |
| Choppy headset video | `scripts/run_sim.sh --no-viewer`. Move closer to the Wi-Fi router. Switch the wrist layout to `off`. |
| How do I leave VR? | Meta button on the right controller. With hand tracking: palm toward your face, pinch and hold. |
| The PC got a new IP address | Use the new link that `run_teleop.sh` prints and accept the certificate warning again. |

## 4. Wrist views (stage 2)

| Problem | Fix |
|---|---|
| `[wrist views] no wrist camera stream to show` | Start the sim with `--addon wrist_cameras` or `--scene lab`, then restart teleop. |
| **X** does nothing | Only in controller mode. Otherwise use the WRIST VIEW button or the page button. |
| The in-VR button closes my gripper | Known when a gripper is used: the trigger does both. Use **X** or the page button. |
| The in-VR button or **X** does not react on the headset | Not yet verified on a physical Quest 3. Use `--wrist-layout` at start and please report it. |

## 5. Real2sim tools (stage 3)

| Problem | Fix |
|---|---|
| Missing files in `calib/real_refs/` | Run `extract_real_refs.py` first, with the option that tool needs (table in stage 3, Part B, step 5). |
| `ffmpeg: not found` | `sudo apt install ffmpeg` |
| `fit_head_full.py`: `kinematic_wrap.json` missing | Run `kinematic_object_poses.py` first. |
| `blender: command not found` | Install Blender, or skip: the generated meshes are already in the repository. |

## 6. Recording (stage 4)

| Problem | Fix |
|---|---|
| `[record] no data: lowstate MISSING, cameras []` | Sim not running, or other `--domain` / `--port-offset`. Give the recorder the same `--domain` and `--image-port` (60000 + offset). |
| `[record] no data: lowstate ok, cameras ['cam_head']` | No wrist cameras in the sim. Use `--scene lab`. |
| Status line shows `src -` | Nobody commands the robot. Press **r** in teleop, or start your policy. |
| Recorder keys do nothing | Click its terminal. `stdin is not a terminal` -> use `--auto-start` or `--start-now`. |
| The recorded waist is always 0 | You used Unitree's `--record` (arms only) or `--no-waist`. Record with `g1lab.record_live` (default 15 joints). |
| Dataset unreadable after a crash | The recorder was killed while saving. Always quit with **q** and wait for `[record] done`. Remove the broken folder and record again. |
| `cannot append to ...` | The folder holds a dataset with another joint layout or fps. Use a new `--root`. |
| Scripted policy fails every attempt | Wrong scene (needs `--scene lab`), or teleop is still running and fights it. |
| `DREAM_API_KEY is not set` | `export DREAM_API_KEY=...` in the terminal. Never write keys into files. |
| VLA arms barely move | You are talking to the mock server (test motion only), or the model is not fine-tuned for the G1. |
| `lerobot-edit-dataset` fails with a missing `..._old/...` file | Do not pass `--root`. Use `HF_LEROBOT_HOME=$PWD/datasets lerobot-edit-dataset --repo_id <folder name> ...` |
| `unitree_bridge.convert` refuses `Unitree_G1_ArmsWaist_3Cam` | Unitree's JSON has no waist. Convert as `Unitree_G1_ArmsOnly_3Cam` and pad, or record with `record_live`. See [unitree_bridge](../04_recording/unitree_bridge/README.md). |
| `hf: command not found` | `source scripts/env.sh g1lab` |

## 7. Training and deployment (stage 5)

| Problem | Fix |
|---|---|
| Training is killed / PC freezes | Out of memory. Close the sim, `--batch-size 2`, `--num-workers 0`. Train on a GPU machine. |
| `Torch not compiled with CUDA enabled` | `g1lab` has CPU PyTorch. `--device cpu`, or a CUDA build on the GPU machine. |
| `... exists; pick a new --output or pass --resume` | New `--output` folder, or `--resume`. |
| `Your dataset must be tagged with a codebase version` | Tag the Hub dataset `v3.0`, see [PUBLISHING.md](PUBLISHING.md). |
| `[act] WARNING: observation.state joints [...] never moved in the training data` | The waist problem: train on data with waist motion, or a 14-joint policy. |
| Policy moves in sim but never succeeds | Too few steps (the smoke test is not a real training), or too few episodes. Check the data first with `lerobot-dataset-viz`. |
| The sim robot ignores `run_act.py` | Another controller is running, or `--domain` / `--image-port` do not match. |
| `run_act.py --target real` does nothing after the checklist | It waits for you to type `yes`. |
| Real robot: arms do not move | The robot is not in debug mode, or its motion controller still owns the arms. See the checklist in stage 5. |

## 8. Stopping stray processes

List Python processes whose command line contains a text, then stop them:

```bash
scripts/pykill.sh g1_xr_sim.py --list
```

```bash
scripts/pykill.sh g1_xr_sim.py
```

Use a specific text (for example `"domain 3"`) if several sims are running and you only want one.
