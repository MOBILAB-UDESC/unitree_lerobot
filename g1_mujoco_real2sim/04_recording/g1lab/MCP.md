# G1 MCP server: an AI assistant as the policy

MCP (Model Context Protocol) is a standard way to give an AI assistant tools it can call. This server gives an assistant such as Claude Code ten tools to **see through the G1's cameras and move its arms step by step**, in the MuJoCo lab sim or on the real robot, and to record what it does as LeRobot episodes.

It is a variant of source B (code-as-policy) of [stage 4](../README.md): the assistant writes or runs the steps itself. You do not need it for the other sources.

- Code: [mcp_server.py](mcp_server.py). Launcher: `scripts/run_mcp.sh`.
- It drives the robot through `g1lab.robot_io.G1Arms`, the same DDS topics and image protocol xr_teleoperate uses.
- It talks MCP over stdio. The assistant's program starts it; you normally do not run it by hand.

## 1. Prerequisites

- `install/install.sh` and `install/setup_g1lab_env.sh` done (the server runs in the `g1lab` env).
- An MCP client. The examples use Claude Code. Get it at <https://claude.com/claude-code>.

## 2. Start the sim

```bash
scripts/run_sim.sh --scene lab
```

You should see `[sim] add-ons: wrist_cameras, lab_scene` (DDS domain 1, images on port 60000).

## 3. Register the server with your assistant

With Claude Code, from the repository folder:

```bash
claude mcp add g1 -- "$PWD/scripts/run_mcp.sh"
```

Or put this in the project's `.mcp.json`. Replace the path with the absolute path of your clone:

```json
{
  "mcpServers": {
    "g1": {
      "command": "/absolute/path/to/unitree_lerobot/g1_mujoco_real2sim/scripts/run_mcp.sh",
      "args": ["--domain", "1", "--image-port", "60000"],
      "env": {"G1_MCP_LOG": "/tmp/g1_mcp.log"}
    }
  }
}
```

You should see the server `g1` with ten tools in the assistant's MCP list. Then ask, for example: "Look at the scene with observe, then put the wrapped box into the cardboard box."

## 4. Tools

Positions are in the robot frame: origin at the pelvis, x forward, y left, z up, in metres. Rotations are roll, pitch, yaw in degrees (`R = Rz(yaw) Ry(pitch) Rx(roll)`). With rpy (0, 0, 0) the fingers point forward and the palms face each other.

| Tool | What it does |
|---|---|
| `observe(cameras, max_width=480)` | One JPEG per camera (`cam_head`, `cam_left_wrist`, `cam_right_wrist`) plus text: the 14 arm joints (measured and commanded), both wrist poses, and in the sim the object poses and the `success` flag. |
| `move_hands(left_xyz, right_xyz, left_rpy, right_rpy, relative=True, duration=1.5)` | Cartesian move solved by IK. Anything left out keeps its current pose. Targets are clamped to the workspace box. A target the IK misses by more than 2 cm is refused and the robot does not move. |
| `squeeze(width_m, height_z, center_xy, duration=1.5)` | Palms facing each other, `width_m` apart, symmetric about a centre point. |
| `move_joints(q, duration=2.0)` | 14 values, or a dict of named joints such as `{"kLeftElbow": 0.5}`. |
| `go_home(pose="zero")` | `zero` = xr_teleoperate's home (forearms forward; in the lab this can touch the box). `rest` = the start pose of the lab episodes. |
| `reset_scene()` | Sim only. Arms to `rest`, then reset the objects. |
| `wait(seconds)` | Wait 0 to 60 s. |
| `start_episode(task)` / `stop_episode(save=True)` | Record a LeRobot episode at 30 Hz in the background (14 arm joints, three cameras). |
| `get_status()` | Connection, limits, recording state, wrist poses. |

- All moves are cosine-interpolated in joint space and run one at a time.
- Reach: at the height of the box centre the palms reach to about x = 0.40 m, so the hands grip the near half of the wrapped box.

## 5. Options

Pass them after the launcher, e.g. in `"args"` of `.mcp.json`.

| Option | Default | Notes |
|---|---|---|
| `--target` | `sim` | `sim` or `real` |
| `--allow-real` | off | Required with `--target real`. Without it the server refuses to start. |
| `--domain` | sim 1, real 0 | DDS domain id |
| `--image-port` | 60000 | Sim started with `--port-offset N` -> 60000 + N |
| `--image-host` | sim 127.0.0.1, real the robot's PC2 | Image server address |
| `--iface` | sim `lo`, real automatic | DDS network interface |
| `--dataset-root` | `datasets/sim_claude` | Where `start_episode` records |
| `--repo-id` | `local/sim_claude` | Dataset label |
| `--workspace XMIN XMAX YMIN YMAX ZMIN ZMAX` | 0.10 0.65 -0.5 0.5 -0.30 0.45 | Box that clamps hand targets (robot frame, m) |
| `--max-joint-step` | sim 2.0 rad, real 0.8 rad | Largest joint change allowed in one tool call |
| `--max-joint-vel` | sim 2.0 rad/s, real 0.8 rad/s | Moves are slowed down to stay under this |
| `--gravity-scale` | sim 0.5, real 1.0 | Scale of the gravity feed-forward torque |
| `--log-level` | `INFO` | |

| Environment variable | What it sets |
|---|---|
| `G1_MCP_ENV` | Conda env of the server (default `g1lab`; recording needs lerobot, which only that env has) |
| `G1_MCP_LOG` | Append the server log to this file (default: stderr) |
| `G1_MCP_TARGET`, `G1_MCP_DOMAIN`, `G1_MCP_IMAGE_PORT`, `G1_MCP_DATASET_ROOT`, `G1_MCP_REPO_ID`, `G1_MCP_LOG_LEVEL` | Defaults for the options of the same name |

A second sim, started with `scripts/run_sim.sh --scene lab --domain 4 --port-offset 2000`, is reached with `scripts/run_mcp.sh --domain 4 --image-port 62000`.

## 6. Real robot

```bash
scripts/run_mcp.sh --target real --allow-real
```

- Lower limits apply (0.8 rad per call, 0.8 rad/s).
- The robot must be in debug mode and nothing else may command it. Keep the emergency stop in reach of a second person.
- Status: the real-robot path exists but has **not been run on the robot yet**.

## 7. Tests

Unit tests with a fake robot (no sim needed):

```bash
source scripts/env.sh g1lab
python -m pytest 04_recording/g1lab/tests/test_mcp_server.py -q
```

Tests through stdio against a running sim. Start one with `scripts/run_sim.sh --scene lab --no-viewer --domain 4 --port-offset 2000`, then:

```bash
G1_MCP_TEST_DOMAIN=4 G1_MCP_TEST_IMAGE_PORT=62000 python -m pytest 04_recording/g1lab/tests/test_mcp_server.py -q
```

## Troubleshooting

- **The assistant lists no `g1` tools**: check the path in `.mcp.json` (absolute, no `~`), and that `scripts/run_mcp.sh` is executable. Read the log file set in `G1_MCP_LOG`.
- **Tools time out with a connection error**: the sim is not running, or `--domain` / `--image-port` do not match the sim.
- **`refusing to drive the real robot without --allow-real`**: add `--allow-real`, on purpose.
- **`start_episode` fails with an import error**: the server is not running in the `g1lab` env. Leave `G1_MCP_ENV` unset.
- **The robot does not move and the tool says the target was refused**: the IK could not reach it within 2 cm. Ask for a closer target.
- **Garbage on the MCP connection**: something printed to stdout. Always start the server through `scripts/run_mcp.sh`.
