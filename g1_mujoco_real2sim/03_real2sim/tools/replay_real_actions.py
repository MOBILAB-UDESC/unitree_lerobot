"""Physics check: replay the real episodes' recorded actions open-loop in the lab sim.

Each episode starts from its recorded first state, with the wrapped box at its kinematic grasp
position (calib/kinematic_wrap.json); the 30 Hz actions are sent through the same PD gains
xr_teleoperate uses. Success is the sim's task check at the end (box resting in the cardboard box).
If mass, friction and squeeze compliance are realistic, most real episodes should also succeed here.

    python 03_real2sim/tools/replay_real_actions.py [--episodes 0 1 2] [--video out.mp4]
"""

import argparse
import json
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1_xr_sim  # noqa: E402

REFS = os.path.join(os.path.dirname(HERE), "calib", "real_refs")
# xr_teleoperate G1_29_ArmController gains: legs/waist 300/3, arms 80/3, wrists 40/1.5
KP = g1_xr_sim.HOLD_KP
KD = g1_xr_sim.HOLD_KD


def make_sim():
    args = g1_xr_sim.parse_args(["--scene", "lab", "--no-viewer"])
    return g1_xr_sim.G1XrSim(args)


def run_episode(sim, ep, kin, renderer=None, frames=None):
    m, d = sim.model, sim.data
    ref = np.load(os.path.join(REFS, f"ep{ep:03d}.npz"))
    sim.reset(robot=True)
    d.qpos[7 + 15:7 + 29] = ref["state"][0]
    wq = m.jnt_qposadr[m.body_jntadr[sim.task.wrap]]
    d.qpos[wq:wq + 2] = kin[str(ep)][:2]
    mujoco.mj_forward(m, d)
    q_cmd = np.zeros(29)
    q_cmd[15:29] = ref["state"][0]
    steps_per_frame = int(round(1 / 30 / sim.dt))
    max_wrap_z = 0.0
    for a in list(ref["action"]) + [ref["action"][-1]] * 45:  # + 1.5 s to settle
        q_cmd[15:29] = a
        sim.cmd[:] = 0
        sim.cmd[0], sim.cmd[2], sim.cmd[3] = q_cmd, KP, KD
        for _ in range(steps_per_frame):
            sim.last_cmd_time = time.monotonic()
            sim.apply_pd()
            mujoco.mj_step(m, d)
        max_wrap_z = max(max_wrap_z, d.xpos[sim.task.wrap][2])
        if renderer is not None:
            renderer.update_scene(d, camera="cam_head")
            head = renderer.render().copy()
            side = mujoco.MjvCamera()
            side.lookat[:] = [0.45, 0.2, 0.9]
            side.distance, side.azimuth, side.elevation = 1.4, 150, -15
            renderer.update_scene(d, camera=side)
            frames.append(np.hstack([head, renderer.render()]))
    wrap, box = d.xpos[sim.task.wrap], d.xpos[sim.task.box]
    return dict(success=sim.task.success(d), lifted=round(float(max_wrap_z - 0.7405), 3),
                final=[round(float(v), 3) for v in wrap], box=[round(float(v), 3) for v in box])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, nargs="*")
    p.add_argument("--video", help="write a head-camera mp4 of the first episode")
    a = p.parse_args()
    kin = json.load(open(os.path.join(os.path.dirname(HERE), "calib", "kinematic_wrap.json")))
    eps = a.episodes or sorted(int(k) for k in kin)
    sim = make_sim()
    renderer, frames = (mujoco.Renderer(sim.model, 480, 640), []) if a.video else (None, None)
    results = {}
    for i, ep in enumerate(eps):
        results[ep] = run_episode(sim, ep, kin, renderer if i == 0 else None, frames)
        print(ep, results[ep], flush=True)
    ok = sum(r["success"] for r in results.values())
    lifted = sum(r["lifted"] > 0.15 for r in results.values())
    print(f"SUCCESS {ok}/{len(eps)}  lifted>15cm {lifted}/{len(eps)}")
    if a.video and frames:
        import cv2
        out = cv2.VideoWriter(a.video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 480))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        print("video:", a.video)


if __name__ == "__main__":
    main()
    os._exit(0)
