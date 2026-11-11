"""Refine the head camera with the desk as a far-field reference (its pose is measured on the real setup).

Objective per real first frame: IoU of the blue desk-top (real colour mask vs sim segmentation,
ignoring pixels the real objects/gloves cover and the background shelf rows) + IoU of the gloves
(dark mask). Searches pitch, forward/up offset and FOV around the current lab_scene values.

    python 03_real2sim/tools/fit_head_desk.py
"""

import glob
import itertools
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
import calibrate_replay as cr  # noqa: E402
import lab_scene  # noqa: E402
from fit_objects import real_masks  # noqa: E402

REFS = os.path.join(SIM, "calib", "real_refs")
W, H = 160, 120
ROI_TOP = int(0.30 * H)  # the blue shelf behind the desk only shows above this


def real_desk(bgr):
    hsv = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    desk = (h >= 95) & (h <= 125) & (s > 80) & (v > 50)
    gray = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    wrap, card = real_masks(bgr)
    occluded = wrap | card | (gray < 50)
    return desk, gray < 50, occluded


def main():
    files = sorted(glob.glob(os.path.join(REFS, "ep*_f0000_cam_head.png")))
    refs = []
    for f in files:
        ep = int(os.path.basename(f)[2:5])
        refs.append((np.load(os.path.join(REFS, f"ep{ep:03d}.npz"))["state"][0], *real_desk(cv2.imread(f))))
    print(f"{len(refs)} first frames", flush=True)

    base_pos = list(lab_scene.HEAD_CAM_POS)
    results = []
    grid = itertools.product([44, 46, 48, 50, 52], [-0.03, 0.0, 0.03], [-0.03, 0.0, 0.03], [42, 44, 46, 48, 50])
    model = data = renderer = None
    for pitch, dx, dz, fovy in grid:
        lab_scene.HEAD_CAM_PITCH_DEG = pitch
        lab_scene.HEAD_CAM_POS = [base_pos[0] + dx, base_pos[1], base_pos[2] + dz]
        if model is None:
            model, data = cr.make_sim(fovy)
            for g in range(model.ngeom):  # hide the task objects
                if model.body(model.geom_bodyid[g]).name in ("env/wrapped_box", "env/cardboard_box"):
                    model.geom_rgba[g, 3] = 0.0
                    model.geom_group[g] = 5
            renderer = mujoco.Renderer(model, H, W)
            desk_body = model.body("env/desk").id
            desk_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == desk_body]
            cam = model.camera("cam_head").id
        model.cam_pos[cam] = lab_scene.HEAD_CAM_POS
        model.cam_quat[cam] = lab_scene.head_camera_quat(pitch)
        model.cam_fovy[cam] = fovy
        sd, sg = [], []
        for q, desk, dark, occ in refs:
            mujoco.mj_resetData(model, data)
            data.qpos[cr.ARM_QPOS] = q
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            renderer.enable_segmentation_rendering()
            seg = renderer.render()[..., 0]
            renderer.disable_segmentation_rendering()
            sim_desk = np.isin(seg, desk_geoms) & ~occ
            sim_dark = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2GRAY) < 50
            roi = np.zeros_like(desk)
            roi[ROI_TOP:] = True
            a, b = desk & roi, sim_desk & roi
            sd.append((a & b).sum() / max(1, (a | b).sum()))
            sg.append((dark & sim_dark).sum() / max(1, (dark | sim_dark).sum()))
        score = np.mean(sd) + 0.5 * np.mean(sg)
        results.append((score, np.mean(sd), np.mean(sg), pitch, dx, dz, fovy))
    results.sort(key=lambda r: -r[0])
    for r in results[:6]:
        print("score %.3f desk %.3f glove %.3f | pitch %d dx %+.2f dz %+.2f fovy %d" % r)


if __name__ == "__main__":
    main()
