"""Head camera fit with known object positions (the decisive real2sim head calibration).

The wrapped box of each episode is placed at its kinematic grasp position (kinematic_wrap.json), so
its silhouette in the head image pins the camera's pitch/height/FOV in the far field; the desk and
the gloves add more constraints. Score = IoU(wrap) + IoU(desk, unoccluded) + 0.5 IoU(gloves).

    python 03_real2sim/tools/fit_head_full.py
"""

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
import lab_scene  # noqa: E402
from fit_head_desk import ROI_TOP, real_desk  # noqa: E402
from fit_objects import real_masks  # noqa: E402

REFS = os.path.join(SIM, "calib", "real_refs")
W, H = 160, 120


def main():
    kin = json.load(open(os.path.join(SIM, "calib", "kinematic_wrap.json")))
    eps = sorted(int(k) for k in kin)[::max(1, len(kin) // 20)]
    model, data = cr.make_sim()
    r = mujoco.Renderer(model, H, W)
    cam = model.camera("cam_head").id
    wrap_b, card_b, desk_b = (model.body(n).id for n in ("env/wrapped_box", "env/cardboard_box", "env/desk"))
    geoms = {b: [g for g in range(model.ngeom) if model.geom_bodyid[g] == b] for b in (wrap_b, desk_b)}
    wq = model.jnt_qposadr[model.body_jntadr[wrap_b]]

    refs = []
    for ep in eps:
        bgr = cv2.imread(os.path.join(REFS, f"ep{ep:03d}_f0000_cam_head.png"))
        wrap, _ = real_masks(bgr)
        desk, dark, occ = real_desk(bgr)
        q = np.load(os.path.join(REFS, f"ep{ep:03d}.npz"))["state"][0]
        refs.append((q, kin[str(ep)][:2], wrap, desk, dark, occ))
    roi = np.zeros((H, W), bool)
    roi[ROI_TOP:] = True

    def score(pitch, dx, dz, fovy):
        model.cam_pos[cam] = [lab_scene.HEAD_CAM_POS[0] + dx, lab_scene.HEAD_CAM_POS[1], lab_scene.HEAD_CAM_POS[2] + dz]
        model.cam_quat[cam] = lab_scene.head_camera_quat(pitch)
        model.cam_fovy[cam] = fovy
        iw, idk, ig = [], [], []
        for q, (wx, wy), wrap, desk, dark, occ in refs:
            mujoco.mj_resetData(model, data)
            data.qpos[cr.ARM_QPOS] = q
            data.qpos[wq:wq + 2] = [wx, wy]
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=cam)
            r.enable_segmentation_rendering()
            seg = r.render()[..., 0]
            r.disable_segmentation_rendering()
            sdark = cv2.cvtColor(r.render(), cv2.COLOR_RGB2GRAY) < 50
            sw = np.isin(seg, geoms[wrap_b])
            sd = np.isin(seg, geoms[desk_b]) & ~occ & roi
            iw.append((wrap & sw).sum() / max(1, (wrap | sw).sum()))
            dd = desk & roi
            idk.append((dd & sd).sum() / max(1, (dd | sd).sum()))
            ig.append((dark & sdark).sum() / max(1, (dark | sdark).sum()))
        return np.mean(iw) + np.mean(idk) + 0.5 * np.mean(ig), np.mean(iw), np.mean(idk), np.mean(ig)

    res = []
    fovys = [float(os.environ["FIXED_FOVY"])] if os.environ.get("FIXED_FOVY") else [40, 43, 46, 49, 52]
    for pitch, dx, dz, fovy in itertools.product(range(36, 58, 2), [-0.06, -0.03, 0.0, 0.03], [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02], fovys):
        s = score(pitch, dx, dz, fovy)
        res.append((*s, pitch, dx, dz, fovy))
    res.sort(key=lambda x: -x[0])
    best = res[0]
    # local refinement
    p0, x0, z0, f0 = best[4:]
    for pitch, dx, dz, fovy in itertools.product(np.arange(p0 - 1.5, p0 + 1.6, 0.5), [x0 - 0.01, x0, x0 + 0.01],
                                                  [z0 - 0.01, z0, z0 + 0.01], [f0] if os.environ.get("FIXED_FOVY") else np.arange(f0 - 2, f0 + 2.1, 1.0)):
        s = score(pitch, dx, dz, fovy)
        res.append((*s, pitch, dx, dz, fovy))
    res.sort(key=lambda x: -x[0])
    for r_ in res[:5]:
        print("score %.3f wrap %.3f desk %.3f glove %.3f | pitch %.1f dx %+.3f dz %+.3f fovy %.1f" % r_, flush=True)


if __name__ == "__main__":
    main()
