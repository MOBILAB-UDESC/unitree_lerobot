"""Estimate the wrapped box / cardboard box start poses of every real episode from its first head frame.

Real masks come from colour (bubble wrap = bright, unsaturated; cardboard = tan); sim masks from
MuJoCo's segmentation render through the calibrated head camera, with the arms at the recorded
state. A coarse-to-fine grid search maximises the IoU per object. Prints per-episode poses and the
mean / range to put into scene_lab.xml and lab_scene.OBJECT_RANGES.

    python 03_real2sim/tools/fit_objects.py [--episodes 0 1 2 ...]
"""

import argparse
import glob
import itertools
import json
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
import calibrate_replay as cr  # noqa: E402

REFS = os.path.join(SIM, "calib", "real_refs")
W, H = 160, 120


def real_masks(bgr):
    hsv = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    wrap = (v > 165) & (s < 55)
    card = (h >= 6) & (h <= 24) & (s > 70) & (v > 70)
    return wrap, card


class SimMasks:
    def __init__(self):
        self.m, self.d = cr.make_sim()
        self.r = mujoco.Renderer(self.m, H, W)
        self.r.enable_segmentation_rendering()
        m = self.m
        self.wrap_body = m.body("env/wrapped_box").id
        self.card_body = m.body("env/cardboard_box").id
        self.wrap_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.wrap_body]
        self.card_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.card_body]

    def set_pose(self, body, x, y, yaw):
        jnt = self.m.body_jntadr[body]
        qa = self.m.jnt_qposadr[jnt]
        self.d.qpos[qa:qa + 3] = [x, y, self.m.qpos0[qa + 2]]
        self.d.qpos[qa + 3:qa + 7] = [np.cos(np.radians(yaw) / 2), 0, 0, np.sin(np.radians(yaw) / 2)]

    def render(self, arm_q, wrap, card):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[cr.ARM_QPOS] = arm_q
        self.set_pose(self.wrap_body, *wrap)
        self.set_pose(self.card_body, *card)
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera="cam_head")
        seg = self.r.render()[..., 0]  # geom ids (objtype in channel 1)
        return np.isin(seg, self.wrap_geoms), np.isin(seg, self.card_geoms)


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def fit(sim, arm_q, real_wrap, real_card):
    card = (0.445, 0.39, 0.0)
    wrap = (0.43, -0.05, 0.0)
    # Cardboard first (wrap held fixed), then wrap; each coarse -> fine.
    for spans, steps in (((0.10, 0.12, 12), (0.02, 0.02, 4)), ((0.025, 0.025, 4), (0.005, 0.005, 1))):
        best = (-1, card)
        for dx, dy, dyaw in itertools.product(*[np.arange(-s, s + 1e-9, st) for s, st in zip(spans, steps)]):
            c = (card[0] + dx, card[1] + dy, card[2] + dyaw)
            _, sc = sim.render(arm_q, wrap, c)
            best = max(best, (iou(sc, real_card), c))
        card = best[1]
    for spans, steps in (((0.14, 0.16, 0), (0.02, 0.02, 1)), ((0.025, 0.025, 30), (0.005, 0.005, 10))):
        best = (-1, wrap)
        for dx, dy, dyaw in itertools.product(*[np.arange(-s, s + 1e-9, st) for s, st in zip(spans, steps)]):
            w = (wrap[0] + dx, wrap[1] + dy, wrap[2] + dyaw)
            sw, _ = sim.render(arm_q, w, card)
            best = max(best, (iou(sw, real_wrap), w))
        wrap, wrap_iou = best[1], best[0]
    _, sc = sim.render(arm_q, wrap, card)
    return wrap, card, wrap_iou, iou(sc, real_card)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, nargs="*")
    p.add_argument("--out", default=os.path.join(SIM, "calib", "object_fits.json"))
    a = p.parse_args()
    eps = a.episodes or sorted(int(os.path.basename(f)[2:5]) for f in glob.glob(os.path.join(REFS, "ep*_f0000_cam_head.png")))
    sim = SimMasks()
    results = {}
    for ep in eps:
        state = np.load(os.path.join(REFS, f"ep{ep:03d}.npz"))["state"][0]
        rw, rc = real_masks(cv2.imread(os.path.join(REFS, f"ep{ep:03d}_f0000_cam_head.png")))
        wrap, card, iw, ic = fit(sim, state, rw, rc)
        results[ep] = dict(wrap=[round(float(v), 4) for v in wrap], card=[round(float(v), 4) for v in card],
                           iou_wrap=round(float(iw), 3), iou_card=round(float(ic), 3))
        print(ep, results[ep], flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
    W_ = np.array([r["wrap"] for r in results.values()])
    C_ = np.array([r["card"] for r in results.values()])
    np.set_printoptions(precision=3, suppress=True)
    print("wrap mean", W_.mean(0), "min", W_.min(0), "max", W_.max(0))
    print("card mean", C_.mean(0), "min", C_.min(0), "max", C_.max(0))


if __name__ == "__main__":
    main()
