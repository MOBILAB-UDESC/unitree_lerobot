"""Photometric real2sim: match the sim's material colours, light level and each camera's gain to the real
recordings.

For sampled pre-grasp real frames (robot at the recorded joints, wrapped box at its kinematic start
position), the sim segmentation gives, per material, which pixels show it; the mean real colour there (over the pixels that look like that
material, see MATERIAL_CLASS) vs the mean rendered colour gives a per-material correction. Iterated a few times (lighting couples them),
then a per-camera RGB gain absorbs each camera's exposure / white balance. Writes calib/photometric.json,
which lab_scene loads at build time. Run it twice: the second run starts from the first one's texture gains
(TEXTURE_GAIN_START=1) and settles the colours of the materials that were stuck at the rgba limit.

    python 03_real2sim/tools/fit_materials.py
"""

import glob
import json
import os
import sys

import cv2
import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, SIM)
import lab_scene  # noqa: E402

lab_scene.PHOTOMETRIC.clear()  # fit from scratch
if os.environ.get("TEXTURE_GAIN_START") and os.path.exists(lab_scene.PHOTOMETRIC_JSON):
    lab_scene.PHOTOMETRIC["texture_gain_rgb"] = json.load(open(lab_scene.PHOTOMETRIC_JSON)).get("texture_gain_rgb", {})
import calibrate_replay as cr  # noqa: E402
from fit_room import BEIGE, BLUE, DARK, WHITE, WOOD, classify  # noqa: E402

REFS = os.path.join(SIM, "calib", "real_refs")
OUT = os.path.join(SIM, "calib", "photometric.json")
W, H = 320, 240
CAMS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
MATERIALS = ["env/desk_top", "env/desk_panel", "env/desk_leg", "env/cardboard_out", "env/cardboard_in",
             "env/bubble_wrap", "glove", "env/floor_tiles", "env/foam_mat",
             # the room past the desk (assets/lab/make_room.py)
             "env/room_wall", "env/room_bench_frame", "env/room_bench_top", "env/room_cabinet",
             "env/room_foam_cube", "env/room_h2d_left", "env/room_h2d_right", "env/room_a1"]


# What each material looks like in the real images, as coarse colour classes (fit_room.classify). Only mask
# pixels whose real colour is in the material's class are compared: the sim and real silhouettes never
# overlap exactly (object poses, the room's placement), and without this gate the neighbours' colours leak
# in (the blue desk into the cardboard rim, the printers into the wall ...).
MATERIAL_CLASS = {
    "env/desk_top": (BLUE,), "env/desk_panel": (WHITE, DARK), "env/desk_leg": (DARK, WHITE),
    "env/cardboard_out": (WOOD, BEIGE), "env/cardboard_in": (WOOD, BEIGE), "env/bubble_wrap": (WHITE,),
    "glove": (DARK,), "env/floor_tiles": (WHITE,), "env/foam_mat": (WHITE, DARK), "env/room_wall": (WHITE,),
    "env/room_bench_frame": (BLUE,), "env/room_bench_top": (WOOD,), "env/room_cabinet": (BEIGE, WHITE),
    "env/room_foam_cube": (WHITE, DARK), "env/room_h2d_left": (DARK,), "env/room_h2d_right": (DARK,),
    "env/room_a1": (WHITE, DARK),
}


def samples(n_eps=26, per_ep=3):
    kin = json.load(open(os.path.join(SIM, "calib", "kinematic_wrap.json")))
    out = []
    for f in sorted(glob.glob(os.path.join(REFS, "ep*_color.npz")))[:: max(1, 52 // n_eps)]:
        ep = int(os.path.basename(f)[2:5])
        npz = np.load(f)
        z = {c: npz[c] for c in CAMS}  # decompress once
        st = np.load(os.path.join(REFS, f"ep{ep:03d}.npz"))["state"]
        n = min(len(st), len(z["cam_head"]))
        for t in np.linspace(0, int(n * 0.3), per_ep).astype(int):
            # copy: a slice would keep the whole decompressed episode (~1 GB) alive
            out.append((st[t].copy(), kin[str(ep)][:2], {c: z[c][t].copy() for c in CAMS}))
        del z, npz
    return out


def main():
    model, data = cr.make_sim()
    r = mujoco.Renderer(model, H, W)
    wq = model.jnt_qposadr[model.body_jntadr[model.body("env/wrapped_box").id]]
    mat_ids = {name: model.material(name).id for name in MATERIALS}
    geom_mat = model.geom_matid.copy()
    refs = samples()
    print(f"{len(refs)} frames x {len(CAMS)} cameras", flush=True)
    light0 = model.light_diffuse.copy()
    head0 = (model.vis.headlight.ambient.copy(), model.vis.headlight.diffuse.copy())
    light_scale = 1.0
    kernel = np.ones((5, 5), np.uint8)

    def measure():
        """Per material: (mean real BGR, mean sim BGR, pixel count) over the mask pixels whose real colour
        is in the material's class; per camera: (sum real, sum sim, count) over all modelled pixels."""
        px = {m: [np.zeros(3), np.zeros(3), 0, 0] for m in MATERIALS}
        cams = {c: [np.zeros(3), np.zeros(3), 0] for c in CAMS}
        for q, wxy, imgs in refs:
            mujoco.mj_resetData(model, data)
            data.qpos[cr.ARM_QPOS] = q
            data.qpos[wq:wq + 2] = wxy
            mujoco.mj_forward(model, data)
            for cam in CAMS:
                r.update_scene(data, camera=cam)
                sim = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR).astype(np.float64)
                r.enable_segmentation_rendering()
                seg = r.render()
                r.disable_segmentation_rendering()
                gid = np.where(seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM, seg[..., 0], -1)
                real = imgs[cam].astype(np.float64)
                real_class = classify(imgs[cam])
                matid = np.where(gid >= 0, geom_mat[np.clip(gid, 0, None)], -1)
                for m, mid in mat_ids.items():
                    mask = cv2.erode((matid == mid).astype(np.uint8), kernel).astype(bool)
                    if mask.sum() < 30:
                        continue
                    px[m][3] += mask.sum()
                    mask &= np.isin(real_class, MATERIAL_CLASS[m])
                    px[m][0] += real[mask].sum(0)
                    px[m][1] += sim[mask].sum(0)
                    px[m][2] += mask.sum()
                fg = gid >= 0
                if model.ngeom and "lab_dome" in [model.geom(i).name for i in range(model.ngeom)]:
                    fg &= gid != model.geom("lab_dome").id
                cams[cam][0] += real[fg].sum(0)
                cams[cam][1] += sim[fg].sum(0)
                cams[cam][2] += fg.sum()
        mats = {}
        for m, (rs, ss, n, total) in px.items():
            if n > 500 and n > 0.15 * total:   # enough of the mask really shows this material
                mats[m] = (rs / n, ss / n, n)
        return mats, cams

    for it in range(5):
        mats, _ = measure()
        ratios = {}
        for m, (rs, ss, n) in mats.items():
            if n > 500:
                ratios[m] = np.clip(rs / np.maximum(ss, 1.0), 0.5, 2.0)  # BGR
        # Global light level from the median brightness ratio, materials take the rest.
        g = float(np.median([v.mean() for v in ratios.values()]))
        light_scale *= g
        model.light_diffuse[:] = light0 * light_scale
        model.vis.headlight.ambient[:] = head0[0] * light_scale
        model.vis.headlight.diffuse[:] = head0[1] * light_scale
        for m, ratio in ratios.items():
            mid = mat_ids[m]
            rgb = model.mat_rgba[mid, :3] * (ratio[::-1] / g)  # BGR ratio -> RGB
            model.mat_rgba[mid, :3] = np.clip(rgb, 0.02, 1.0)
        print(f"iter {it}: light x{light_scale:.2f}; " + ", ".join(
            f"{m.split('/')[-1]} {np.round(ratio[::-1], 2)}" for m, ratio in ratios.items()), flush=True)

    # Materials whose colour hit the rgba <= 1 limit are still too dark (dark textures: glove, desk legs,
    # foam mats ...): record the remaining ratio as a gain for the texture itself (lab_scene applies it).
    mats, cams = measure()
    tex_gain = {}
    for m, (rs, ss, n) in mats.items():
        if n > 500:
            ratio = (rs / np.maximum(ss, 1.0))[::-1]  # RGB
            if np.any(ratio > 1.04) and np.any(model.mat_rgba[mat_ids[m], :3] > 0.995):
                tex_gain[m] = np.round(np.clip(ratio, 0.8, 4.0), 3).tolist()
    gains = {c: np.round((rs / n) / np.maximum(ss / n, 1.0), 3)[::-1].tolist() for c, (rs, ss, n) in cams.items() if n}
    for m, g0 in lab_scene.PHOTOMETRIC.get("texture_gain_rgb", {}).items():  # gains this run started from
        tex_gain[m] = np.round(np.clip(np.array(g0) * np.array(tex_gain.get(m, [1, 1, 1])), 0.8, 4.0), 3).tolist()
    result = {
        "light_scale": round(light_scale, 3),
        "materials": {m: np.round(model.mat_rgba[mat_ids[m]], 3).tolist() for m in MATERIALS},
        "texture_gain_rgb": tex_gain,
        "camera_gain_rgb": gains,
    }
    json.dump(result, open(OUT, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
