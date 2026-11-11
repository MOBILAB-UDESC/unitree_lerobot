"""Real2sim calibration: render the sim at the real robot's recorded joint states, next to the real frames.

Kinematic replay (no physics): the 14 arm joints are set from observation.state, the rest of the robot
is at its standing pose, and the objects are placed with --wrap / --box. Writes, per frame, a
[real | sim | 50 % blend] strip for each camera into one contact sheet.

Reference data comes from tools/extract_real_refs.py (03_real2sim/calib/real_refs).

    python 03_real2sim/tools/calibrate_replay.py --episode 0 --frames 0 120 240 --out calib_sheet.jpg
"""

import argparse
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1_xr_sim  # noqa: E402

REFS = os.path.join(os.path.dirname(HERE), "calib", "real_refs")
CAMS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
ARM_QPOS = slice(7 + 15, 7 + 29)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--frames", type=int, nargs="+", default=[0, 120, 240])
    p.add_argument("--cams", nargs="+", default=CAMS)
    p.add_argument("--wrap", type=float, nargs=3, default=None, help="wrapped box x y yaw(deg)")
    p.add_argument("--box", type=float, nargs=3, default=None, help="cardboard box x y yaw(deg)")
    p.add_argument("--head-fovy", type=float, default=None)
    p.add_argument("--scale", type=float, default=0.5, help="output tile scale")
    p.add_argument("--out", required=True)
    return p.parse_args()


def place(model, data, body, xyyaw):
    jnt = model.body(body).jntadr[0]
    qa = model.jnt_qposadr[jnt]
    x, y, yaw = xyyaw
    data.qpos[qa:qa + 3] = [x, y, model.qpos0[qa + 2]]
    data.qpos[qa + 3:qa + 7] = [np.cos(np.radians(yaw) / 2), 0, 0, np.sin(np.radians(yaw) / 2)]


def make_sim(head_fovy=None):
    args = g1_xr_sim.parse_args(["--scene", "lab"] + (["--head-fovy", str(head_fovy)] if head_fovy else []))
    model, _ = g1_xr_sim.build_model(args)
    return model, mujoco.MjData(model)


def main():
    a = parse()
    model, data = make_sim(a.head_fovy)
    ref = np.load(os.path.join(REFS, f"ep{a.episode:03d}.npz"))
    renderer = mujoco.Renderer(model, 480, 640)
    rows = []
    for f in a.frames:
        mujoco.mj_resetData(model, data)
        data.qpos[ARM_QPOS] = ref["state"][min(f, len(ref["state"]) - 1)]
        if a.wrap:
            place(model, data, "env/wrapped_box", a.wrap)
        if a.box:
            place(model, data, "env/cardboard_box", a.box)
        mujoco.mj_forward(model, data)
        strip = []
        for cam in a.cams:
            path = os.path.join(REFS, f"ep{a.episode:03d}_f{f:04d}_{cam}.png")
            real = cv2.imread(path) if os.path.exists(path) else np.zeros((480, 640, 3), np.uint8)
            renderer.update_scene(data, camera=cam)
            sim = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            blend = cv2.addWeighted(real, 0.5, sim, 0.5, 0)
            for img, label in ((real, "real"), (sim, "sim"), (blend, "blend")):
                cv2.putText(img, f"{label} {cam} f{f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            strip.append(np.hstack([real, sim, blend]))
        rows.append(np.vstack(strip))
    sheet = np.vstack(rows)
    sheet = cv2.resize(sheet, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(a.out, sheet)
    print("wrote", a.out, sheet.shape)


if __name__ == "__main__":
    main()
