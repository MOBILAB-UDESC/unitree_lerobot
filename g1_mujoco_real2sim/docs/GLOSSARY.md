# Glossary

Plain-word explanations of the terms used in this repository. Back to the [main README](../README.md).

## 1. Robot and simulation

| Term | Meaning |
|---|---|
| **G1** | Unitree's humanoid robot. Ours has 29 motors ("29 DoF"): 12 in the legs, 3 in the waist, 7 in each arm. The hands are fixed rubber hands with knit gloves; there are no fingers or grippers to control. |
| **PC2** | The G1's onboard development computer, at 192.168.123.164 on the robot's network. On the real robot the camera (image) server runs there. |
| **DoF** | Degrees of freedom: the number of independent joints. |
| **MuJoCo** | The physics simulator used here. It computes how bodies move and collide, and renders camera images. |
| **MJCF** | MuJoCo's XML file format for robots and scenes (`*.xml` files such as `scene_lab.xml`). |
| **URDF** | Another XML file format for robots (links, joints, meshes). Unitree publishes the G1 in it; the IK of xr_teleoperate reads it. |
| **Mesh / STL / OBJ** | A 3D shape made of triangles, and two file formats for it. |
| **Add-on** | In this repo: a small Python module that changes the robot or the scene when the simulator starts (`--addon NAME`), without editing the simulator. Stage 2 and stage 3 are add-ons. |
| **Gantry** | The frame a real G1 hangs from in the lab. In the sim, an invisible body that holds the pelvis in FIXED mode. |
| **FIXED / FREE** | The two base modes of the sim (key 9). FIXED: the pelvis is held, the robot can "walk". FREE: the robot stands on its feet and can fall. |
| **Kinematic walking** | Walking where the body is simply moved along while the legs play an animation. No balance is computed. |
| **Real-time factor** | Simulated time divided by clock time. 1.0 means the physics keeps up with the clock; below 1.0 the PC is overloaded and the sim runs in slow motion. The sim publishes it as `rtf`. |
| **PD control** | The basic joint controller: torque = stiffness x position error + damping x velocity error. |
| **IK (inverse kinematics)** | Computing the joint angles that put a hand at a wanted position and orientation. |
| **FK (forward kinematics)** | The opposite: from joint angles to the hand's position. |
| **Robot frame** | Coordinates with the origin at the pelvis: x forward, y to the robot's left, z up, in metres. |

## 2. Communication

| Term | Meaning |
|---|---|
| **DDS** | The network messaging system Unitree robots use. Programs publish and subscribe to named **topics**; nobody needs to know the other's address. |
| **Topic** | A named message channel, e.g. `rt/lowstate` (joint measurements) or `rt/lowcmd` (joint commands). |
| **Domain id** | A number that separates DDS networks. Programs only hear each other on the same domain. The real robot uses 0, the sim uses 1, so they cannot be mixed up. |
| **Network interface** | Which network card DDS uses. `lo` (loopback) means "this PC only". |
| **`rt/lowcmd` / `rt/lowstate`** | Low-level command to, and state from, every motor. |
| **`rt/arm_sdk`** | Arm-only command topic used when Unitree's motion controller owns the legs. |
| **Debug mode / motion mode** | Two ways to drive a G1. Debug: your program commands every joint (`rt/lowcmd`). Motion: Unitree's controller keeps balance and walks; you add arm commands (`rt/arm_sdk`) and joystick velocities. |
| **Damp** | A robot state where the motors go limp. Used as a soft emergency stop. |
| **teleimager** | Unitree's small camera streaming protocol (part of xr_teleoperate). A config port (60000) says which cameras exist; each camera then streams JPEG images on its own port. |
| **ZMQ** | The socket library teleimager uses. |
| **Port** | A number that identifies a network service on a PC, e.g. 8012 for the Quest web page. |
| **Self-signed certificate** | A do-it-yourself HTTPS certificate. Browsers warn about it; here that is expected and safe to accept. |
| **MCP** | Model Context Protocol: a standard way to give an AI assistant tools. Our MCP server gives it "look" and "move the arms" tools. |

## 3. Teleoperation and VR

| Term | Meaning |
|---|---|
| **Teleoperation (teleop)** | A person drives the robot remotely; here with a VR headset. |
| **xr_teleoperate** | Unitree's teleoperation program. It reads your hand or controller poses from the headset, solves IK, and commands the arms. Used here unmodified. |
| **Quest 3** | Meta's VR headset. It runs the teleop web page in its browser. |
| **WebXR** | The browser feature that lets a web page show VR content. It only works on HTTPS pages. |
| **Hand tracking / controller mode** | Two input methods: bare hands seen by the headset cameras, or the two handheld controllers. |
| **Thumbstick** | The small joystick on each controller. |
| **Stereo / mono** | Two images (one per eye, gives depth) or one image. |
| **RealSense D405 / D435i** | Intel cameras. The G1 has a D435i in its head; ours also has one D405, a small close-range model, on each wrist. Only their colour images are used here. |
| **PiP (picture-in-picture)** | A small image drawn inside a bigger one. The `pip` layout shows the wrist views small inside the head view. |
| **FOV (field of view)** | The angle a camera sees. A larger FOV shows more of the scene. |
| **Intrinsics** | A camera's internal numbers (focal length, image centre). They fix its FOV. |

## 4. Real2sim

| Term | Meaning |
|---|---|
| **Real2sim** | Rebuilding a real setup in simulation, so the sim looks and behaves like it. |
| **Sim2real** | Using something made in simulation (data, a policy) on the real robot. |
| **Calibration** | Adjusting sim parameters (camera pose, colours, friction) until sim output matches real measurements. |
| **Photometric match** | Calibration of colours and brightness. |
| **Domain randomization** | Changing object positions, lights and colours a little at every reset, so a policy does not depend on one exact picture. |
| **Open-loop replay** | Playing back recorded commands without looking at what happens. Used to test if the sim physics reacts like the real world. |
| **Privileged state** | Information only a simulator has, such as the exact pose of every object. |

## 5. Data and learning

| Term | Meaning |
|---|---|
| **Episode** | One recorded attempt at the task, from start to end. |
| **LeRobot** | Hugging Face's open-source robotics library: dataset format, policies, training scripts. |
| **LeRobot dataset (v3.0)** | A folder with tables of joint values (parquet files), camera videos (mp4) and metadata (`meta/info.json`). Version 3.0 is the layout LeRobot 0.4.1 writes. |
| **Rerun viewer** | The window `lerobot-dataset-viz` opens to play an episode: the camera videos and joint curves on a timeline. |
| **`observation.state` / `action`** | Per time step: the measured joint angles, and the commanded joint angles. |
| **robot_type** | A label in the dataset that names the joint layout, e.g. `Unitree_G1_ArmsWaist_3Cam`. |
| **Hugging Face Hub** | The website where datasets and models are stored and shared. |
| **Policy** | The program that decides the next action from what the robot sees. It can be a person (teleop), hand-written code, or a neural network. |
| **Imitation learning** | Training a policy to copy recorded demonstrations. |
| **ACT** | Action Chunking with Transformers: an imitation-learning policy that predicts a short sequence ("chunk") of future actions from camera images and joint angles. |
| **Action chunk** | The sequence of future actions a policy predicts in one go. |
| **Checkpoint** | A saved copy of a policy's weights during or after training. |
| **Closed loop** | The policy keeps looking at new observations while it acts. |
| **VLA** | Vision-language-action model: a large neural network that takes images and a text instruction and outputs robot actions. Usually runs on a remote server. |
| **Embodiment** | The specific robot body a model was trained for. A VLA must know the G1's embodiment to control it. |
| **Code-as-policy** | A policy written as an ordinary short program (read object poses, compute hand targets, move), often by an AI assistant. |
| **Variance / normalization** | Policies scale each input by how much it varied in the training data. An input that never varied (variance zero) breaks that scaling. This is the "waist problem" in stage 5. |
| **Conda env** | A separate Python installation with its own packages. This repo uses two: `g1sim` (stages 1-3) and `g1lab` (stages 4-5). `source scripts/env.sh NAME` activates one. |
| **Smoke test** | A very short run that only checks that nothing crashes. |
