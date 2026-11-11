# Third-party code and assets

What this repository uses that was made by others, where it comes from and under which license. Back to the [main README](README.md).

The code written for this repository is licensed under Apache-2.0. Everything listed here keeps its own license.

## 1. Upstream repositories cloned by the installer

`install/install.sh` clones these into `third_party/` (not in git). They are used **unmodified**.

| Repository | Used for | License | Pinned commit |
|---|---|---|---|
| [unitreerobotics/xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate) | Teleoperation front-end, arm controller, IK | Apache-2.0 | `817fb00c` |
| its submodule `teleop/teleimager` | Camera streaming protocol and client | Apache-2.0 | (with xr_teleoperate) |
| its submodule `teleop/televuer` | VR web server | MIT | (with xr_teleoperate) |
| [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | G1 MuJoCo model `g1_29dof.xml` and its meshes | BSD-3-Clause | `1eb6642e` |
| [unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) | DDS messages and channels | BSD-3-Clause | `814556d1` |
| [unitreerobotics/unitree_lerobot](https://github.com/unitreerobotics/unitree_lerobot) | Data editor, folder renumbering, JSON converter (stage 4) | Apache-2.0 | `main` |

The licenses above were read from the `LICENSE` file of each repository at the pinned commit (unitree_lerobot: from GitHub). Check them again if you change a pin.

## 2. Libraries installed from package indexes

| Library | Used for | License |
|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Physics and rendering | Apache-2.0 |
| [LeRobot](https://github.com/huggingface/lerobot) 0.4.1 | Dataset format, ACT policy, training and editing tools | Apache-2.0 |
| PyTorch, pinocchio, casadi, OpenCV, vuer, pyzmq, PyAV and their dependencies | Installed by pip / conda | Their own licenses |

## 3. Assets stored in this repository

| Files | Origin | License / note |
|---|---|---|
| `02_wrist_cameras/assets/mesh/*_wrist_yaw_link_5010.STL`, `02_wrist_cameras/assets/src/*_rubber_hand.STL` | Unitree G1 description in [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros) | BSD-3-Clause |
| `02_wrist_cameras/assets/mesh/glove_*.obj` | Made here from Unitree's rubber hand mesh (inflated, UV-mapped) | Derived from the above |
| `02_wrist_cameras/assets/mesh/d405_intel.stl`, `02_wrist_cameras/assets/src/_d405.urdf.xacro` | Intel RealSense D405 model from `realsense2_description` in [IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros) | Apache-2.0 |
| `02_wrist_cameras/assets/mesh/d405_mount_*.stl`, `02_wrist_cameras/assets/src/unitree-g1-d405-wrist-*` , `adapt_5010.py` | MOBILAB bracket for the 5010 wrist, [MOBILAB-UDESC/cad-models](https://github.com/MOBILAB-UDESC/cad-models) (`unitree-g1/d405-wrist-mount-5010`). It is an adaptation of the D405 wrist mount published with xr_teleoperate. | MOBILAB; original mount Apache-2.0. The cad-models repository does not state a license yet. |
| `02_wrist_cameras/assets/tex/glove_knit.png` | Made here | MOBILAB, same license as this repository |
| `03_real2sim/assets/lab/mesh/*`, `03_real2sim/assets/lab/tex/*` | Made here by the generator scripts from our own photos and recordings | MOBILAB, same license as this repository |
| `03_real2sim/assets/lab/ref/*.jpg`, `ref/room/*.jpg` | Photos of the MOBILAB lab, taken by the lab | MOBILAB, same license as this repository |
| The G1 robot model used at run time | Loaded from `third_party/unitree_mujoco`; not copied into this repository | BSD-3-Clause |

## 4. Product names and trademarks

- Unitree, G1, Meta Quest, Intel RealSense, NVIDIA, Bambu Lab, Creality, Clearpath and other names belong to their owners. They are used here only to describe the hardware.
- Bambu Lab 3D printers (H2D, A1, AMS) appear **only as photo textures** in the modelled lab room (`03_real2sim/assets/lab/tex/room_*.png`), because they stand behind the desk in the real lab. No Bambu Lab CAD data or software is included.
- Creality and Clearpath logos are visible in some of the lab photos (`03_real2sim/assets/lab/ref/`), because those products stand in the lab.

## 5. Remote services

The remote VLA client ([04_recording/g1lab/remote_vla.py](04_recording/g1lab/remote_vla.py)) can talk to Dream Labs' Dream-1 API, a Physical Intelligence openpi server or an NVIDIA Isaac GR00T server. It contains no code from them: the request formats were written from their public documentation. Using those services is subject to their own terms. API keys are read from environment variables and are never stored in this repository.

## 6. Datasets

The datasets on the Hugging Face Hub were recorded by MOBILAB. Their license is stated on each dataset card ([04_recording/hf/cards](04_recording/hf/cards)).
