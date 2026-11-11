"""Place the modelled lab room (assets/lab/make_room.py) where the real cameras saw it.

The calibrated head and wrist cameras look past the desk at the workbench, cabinet, foam cube and walls.
For sampled real frames (robot at the recorded joints), every pixel that the sim says shows the room is
labelled by colour in both images (blue bench frame, wood top, beige cabinet, dark printers, neutral
wall / foam / grey foam); the room's pose (shift x/y, yaw about the bench centre) and the cabinet stack's height are
searched to maximise the label agreement. Prints the values to put in make_room.py (ROOM_* constants).

    python 03_real2sim/tools/fit_room.py
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

REFS = os.path.join(SIM, "calib", "real_refs")
W, H = 320, 240
CAMS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
BLUE, WOOD, BEIGE, DARK, WHITE = range(1, 6)
GEOM_CLASS = {"bench_frame": BLUE, "bench_top": WOOD, "cabinet": BEIGE, "foam_cube": WHITE, "h2d": DARK,
              "ams": DARK, "filament_box": DARK, "dryer": DARK, "monitor": DARK, "desk2": DARK,
              "wall": WHITE, "whiteboard": WHITE, "side_cabinet": BEIGE}
STACK = ("cabinet", "foam_cube")   # moved up/down together by dz
DZ_MAX = 0.12
DX_MIN = -0.15   # the workbench stands behind the desk (floor visible between them in the head view)
SCORED = (BLUE, WOOD, BEIGE, DARK)  # the distinctive classes; white wall/floor is everywhere


def classify(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    out = np.zeros(h.shape, np.int8)
    out[(s < 45) & (v > 70)] = WHITE  # wall in shade and the grey foam are too alike to tell apart
    out[(h >= 10) & (h <= 32) & (s >= 30) & (s < 100) & (v > 140)] = BEIGE
    out[(h >= 5) & (h <= 25) & (s >= 110) & (v > 90)] = WOOD
    out[(h >= 95) & (h <= 125) & (s > 100) & (v > 50)] = BLUE
    out[v < 55] = DARK
    return out


def samples(n_eps=13, per_ep=4):
    out = []
    for f in sorted(glob.glob(os.path.join(REFS, "ep*_color.npz")))[:: max(1, 52 // n_eps)]:
        ep = int(os.path.basename(f)[2:5])
        npz = np.load(f)
        z = {c: npz[c] for c in CAMS}
        st = np.load(os.path.join(REFS, f"ep{ep:03d}.npz"))["state"]
        n = min(len(st), len(z["cam_head"]))
        for t in np.linspace(0, n - 1, per_ep).astype(int):
            out.append((st[t], {c: classify(cv2.resize(z[c][t], (W, H))) for c in CAMS}))
    return out


def main():
    model, data = cr.make_sim()
    r = mujoco.Renderer(model, H, W)
    r.enable_segmentation_rendering()
    room = [i for i in range(model.ngeom) if model.geom(i).name.startswith("env/room_")]
    cls = np.zeros(model.ngeom + 1, np.int8)
    for i in room:
        name = model.geom(i).name[len("env/room_"):]
        for key, c in GEOM_CLASS.items():
            if name.startswith(key):
                cls[i] = c
    stack = np.array([i for i in room if model.geom(i).name[len("env/room_"):] in STACK])
    pos0 = model.geom_pos.copy()
    quat0 = model.geom_quat.copy()
    bench = model.geom("env/room_bench_top").id
    pivot = pos0[bench].copy()
    refs = samples()
    print(f"{len(refs)} frames", flush=True)

    def apply(dx, dy, yaw_deg, dz, sx=0.0, sy=0.0):
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        p = (pos0[room] - pivot) @ R.T + pivot + [dx, dy, 0]
        model.geom_pos[room] = p
        q = np.array([np.cos(np.radians(yaw_deg) / 2), 0, 0, np.sin(np.radians(yaw_deg) / 2)])
        for i in room:
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, q, quat0[i])
            model.geom_quat[i] = out
        model.geom_pos[stack] += [sx, sy, dz]  # the cabinet stack also moves on its own

    def score(params):
        """Mean intersection-over-union of the distinctive classes over room pixels."""
        apply(*params)
        inter = np.zeros(len(SCORED))
        union = np.zeros(len(SCORED))
        for q, real in refs:
            mujoco.mj_resetData(model, data)
            data.qpos[cr.ARM_QPOS] = q
            mujoco.mj_forward(model, data)
            for cam in CAMS:
                r.update_scene(data, camera=cam)
                seg = r.render()
                gid = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], model.ngeom)
                sim = cls[gid]
                m = (sim > 0) & (real[cam] > 0)
                rs, ss = real[cam][m], sim[m]
                for k, c in enumerate(SCORED):
                    inter[k] += np.count_nonzero((rs == c) & (ss == c))
                    union[k] += np.count_nonzero((rs == c) | (ss == c))
        return float(np.mean(inter / np.maximum(union, 1)))

    if os.environ.get("ROOM_EVAL"):  # score given candidates: "dx dy yaw dz sx sy; ..."
        for cand in os.environ["ROOM_EVAL"].split(";"):
            p = tuple(float(v) for v in cand.split())
            print(f"  {score(p):.3f} " + " ".join(f"{v:+.3f}" for v in p), flush=True)
        return
    start = tuple(float(v) for v in os.environ.get("ROOM_START", "0 0 0 0 0 0").split())
    best = (score(start), start)
    print(f"start: {best[0]:.3f}", flush=True)
    # coarse-to-fine coordinate search
    steps = [0.06, 0.08, 3.0, 0.06, 0.1, 0.1]
    for rnd in range(4):
        improved = True
        while improved:
            improved = False
            for k, sgn in itertools.product(range(len(steps)), (1, -1)):
                p = list(best[1])
                p[k] += sgn * steps[k]
                if abs(p[3]) > DZ_MAX or p[0] < DX_MIN or not -0.3 <= p[4] <= 0.3 or not -0.3 <= p[5] <= 0.8:
                    continue
                s = score(p)
                if s > best[0] + 1e-4:
                    best = (s, tuple(p))
                    improved = True
                    print(f"  {s:.3f} " + " ".join(f"{v:+.3f}" for v in p), flush=True)
        steps = [x / 2 for x in steps]
    s, (dx, dy, yaw, dz, sx, sy) = best
    print(f"best {s:.3f}: ROOM_SHIFT = ({dx:+.3f}, {dy:+.3f}), ROOM_YAW_DEG = {yaw:+.1f}, CABINET_DZ = {dz:+.3f}, "
          f"CABINET_SHIFT = ({sx:+.3f}, {sy:+.3f})  (relative to the scene the fit started from)")


if __name__ == "__main__":
    main()
