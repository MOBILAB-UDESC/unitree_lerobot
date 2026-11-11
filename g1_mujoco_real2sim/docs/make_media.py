"""Render the pictures and GIFs used in the docs (docs/media/) from the simulator itself.

Everything is rendered offscreen, in this process (no DDS, no headset), so the docs can be refreshed
whenever a scene changes:

    source scripts/env.sh
    python docs/make_media.py                      # everything that needs no recorded session
    python docs/make_media.py --log session.npz    # + hero / episode GIFs from a recorded session

A session log comes from the simulator's --log-qpos option while any controller drives it, e.g.

    scripts/run_sim.sh --scene lab --no-viewer --log-qpos session.npz        # terminal 1 (Ctrl-C when done)
    python -m g1lab.scripted.box_into_box --episodes 1                        # terminal 2

The real-vs-sim pictures need the reference frames of the real dataset (03_real2sim/calib/real_refs,
see 03_real2sim/tools/extract_real_refs.py); without them those pictures are skipped.
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import mujoco
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for sub in ("01_teleop/g1sim", "02_wrist_cameras", "03_real2sim", "03_real2sim/tools"):
    sys.path.insert(0, os.path.join(REPO, sub))
import g1_xr_sim  # noqa: E402

MEDIA = os.path.join(HERE, "media")
REFS = os.path.join(REPO, "03_real2sim", "calib", "real_refs")
ROOM_PHOTOS = os.path.join(REPO, "03_real2sim", "assets", "lab", "ref", "room")
CAMS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ARMS_FORWARD = [0.611, 0.042, -0.177, -0.544, -0.025, -0.33, -0.028,
                0.56, -0.026, -0.031, -0.607, -0.029, -0.123, -0.011]   # hands in front of the chest
ARM_QPOS = slice(7 + 15, 7 + 29)


# ------------------------------------------------------------------------------------------ helpers
class Scene:
    """A simulator instance (no DDS) plus offscreen renderers."""

    def __init__(self, argv):
        self.args = g1_xr_sim.parse_args(["--no-viewer", "--no-camera"] + argv)
        self.sim = g1_xr_sim.G1XrSim(self.args)
        self.m, self.d = self.sim.model, self.sim.data
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = 1920, 1080   # room for the big stills
        self.renderers = {}
        self.opt = mujoco.MjvOption()
        for mod in self.args.addons:
            for group in getattr(mod, "VIEWER_GEOM_GROUPS", ()):
                self.opt.geomgroup[group] = 1

    def renderer(self, w, h):
        if (w, h) not in self.renderers:
            self.renderers[(w, h)] = mujoco.Renderer(self.m, h, w)
        return self.renderers[(w, h)]

    def view(self, lookat, distance, azimuth, elevation, size=(960, 540)):
        """Free camera, like the desktop viewer (shows the robot's head too)."""
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r = self.renderer(*size)
        r.update_scene(self.d, camera=cam, scene_option=self.opt)
        return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)

    def camera(self, name, size=(640, 480)):
        """A robot camera, with the add-ons' camera effects (exposure, blur, noise)."""
        r = self.renderer(*size)
        r.update_scene(self.d, camera=name)
        rgb = r.render()
        for mod in self.args.addons:
            if hasattr(mod, "camera_effects"):
                rgb = mod.camera_effects(rgb, name)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def pose_arms(self, q=ARMS_FORWARD):
        self.d.qpos[ARM_QPOS] = q
        mujoco.mj_forward(self.m, self.d)

    def step(self, seconds):
        for _ in range(int(round(seconds / self.sim.dt))):
            self.sim.loco_step()
            self.sim.apply_pd()
            mujoco.mj_step(self.m, self.d)


def label(img, text, scale=0.6, org=(8, 22)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fit(img, w=None, h=None):
    s = w / img.shape[1] if w else h / img.shape[0]
    return cv2.resize(img, (w or int(round(img.shape[1] * s)), h or int(round(img.shape[0] * s))),
                      interpolation=cv2.INTER_AREA)


def save(name, img, quality=88):
    path = os.path.join(MEDIA, name)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    print(f"  {name}  {img.shape[1]}x{img.shape[0]}  {os.path.getsize(path) / 1e3:.0f} kB")


def save_gif(name, frames, fps, colors=128):
    """One shared palette (from a mosaic of frames) keeps the file small and free of flicker."""
    ims = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, 6).astype(int)]
    mosaic = Image.fromarray(cv2.cvtColor(np.vstack(picks), cv2.COLOR_BGR2RGB))
    palette = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    ims = [im.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for im in ims]
    path = os.path.join(MEDIA, name)
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  {name}  {frames[0].shape[1]}x{frames[0].shape[0]}  {len(frames)} frames  "
          f"{os.path.getsize(path) / 1e6:.1f} MB")


# ------------------------------------------------------------------------------------------ stage 1
def stage1():
    s = Scene([])
    s.pose_arms()
    save("stage1_demo_scene.png", s.view([0.35, 0.0, 0.85], 2.2, 140, -18, (1280, 720)))
    stereo = np.hstack([s.camera("head_left"), s.camera("head_right")])
    save("stage1_head_stereo.png", label(stereo, "head camera stream: left eye | right eye (1280x480)"))

    # Controller mode: the left stick walks (kinematic, on the gantry), the right stick twists the waist.
    s = Scene([])
    s.sim.arm_home = np.array(ARMS_FORWARD) * 0.6
    s.pose_arms(s.sim.arm_home)
    plan = [(0.8, (0, 0, 0), "controller mode"), (1.8, (0, 0, 0.3), "right stick: twist the waist"),
            (3.0, (0, 0, -0.3), "right stick: twist the waist"), (1.4, (0, 0, 0.3), "right stick: twist the waist"),
            (2.5, (-0.3, 0, 0), "left stick: walk"), (2.0, (0, 0.25, 0), "left stick: walk"),
            (2.2, (0.3, -0.1, 0), "left stick: walk"), (0.6, (0, 0, 0), "")]
    frames = []
    for seconds, vel, text in plan:
        with s.sim.loco_lock:
            s.sim.vel_cmd[:] = vel
            s.sim.vel_cmd_until = time.monotonic() + 3600
        for _ in range(int(seconds * 10)):
            s.step(0.1)
            pelvis = s.d.body("pelvis").xpos
            frame = s.view([pelvis[0] + 0.2, pelvis[1], 0.8], 2.6, 150, -15, (480, 300))
            frames.append(label(frame, text))
    save_gif("stage1_walk.gif", frames, 10, colors=64)


# ------------------------------------------------------------------------------------------ stage 2
def stage2():
    s = Scene(["--addon", "wrist_cameras"])
    s.pose_arms()
    wrist = s.d.body("left_wrist_yaw_link").xpos
    close = s.view(wrist + [0.03, 0.0, 0.03], 0.45, 215, -25, (1100, 720))
    save("stage2_wrist_hardware.png", label(close, "5010 wrist + knit glove + MOBILAB bracket + RealSense D405"))

    head, left, right = s.camera("head_left"), s.camera("cam_left_wrist"), s.camera("cam_right_wrist")
    row = np.hstack([label(head.copy(), "head"), label(left.copy(), "left wrist"), label(right.copy(), "right wrist")])
    save("stage2_wrist_views.png", row)

    # Illustrations of the two headset layouts (what the operator sees; composed from the same frames).
    big = fit(head, w=960)
    pip = big.copy()
    ph = int(big.shape[0] * 0.33)
    for img, x in ((left, 12), (right, big.shape[1] - 12 - int(ph * 4 / 3))):
        small = fit(img, h=ph)
        pip[big.shape[0] - ph - 12:big.shape[0] - 12, x:x + small.shape[1]] = small
        cv2.rectangle(pip, (x, big.shape[0] - ph - 12), (x + small.shape[1], big.shape[0] - 12), (255, 255, 255), 2)
    save("stage2_layout_pip.png", label(pip, "layout 'pip': wrist views small in the bottom corners (illustration)"))
    h = 300
    side = np.full((h + 60, 3 * 400 + 40, 3), 18, np.uint8)
    for k, (img, text) in enumerate(((left, "left wrist"), (head, "head"), (right, "right wrist"))):
        tile = fit(img, h=h)
        side[40:40 + h, 10 + k * 410:10 + k * 410 + tile.shape[1]] = tile
        label(side, text, org=(16 + k * 410, 62))
    save("stage2_layout_side.png", label(side, "layout 'side': the three views in one row, same size (illustration)"))


# ------------------------------------------------------------------------------------------ stage 3
def real_frame(ep, f, cam):
    path = os.path.join(REFS, f"ep{ep:03d}_f{f:04d}_{cam}.png")
    return cv2.imread(path) if os.path.exists(path) else None


def stage3():
    s = Scene(["--scene", "lab"])
    save("stage3_lab_overview.png", s.view([0.8, 0.0, 0.95], 2.7, -38, -20, (1280, 720)))

    # Lab photos next to the modelled room from about the same viewpoints.
    views = [("12.jpg", ([1.0, -0.15, 1.0], 3.1, 0, -8)), ("11.jpg", ([2.1, 0.35, 1.0], 1.4, 0, -5))]
    rows = []
    for photo, view in views:
        real = cv2.imread(os.path.join(ROOM_PHOTOS, photo))
        if real is None:
            continue
        real = fit(real, h=405)
        rows.append(np.hstack([label(real, "lab photo"), label(fit(s.view(*view, (720, 405)), h=405), "sim")]))
    if rows:
        w = min(r.shape[1] for r in rows)
        save("stage3_room_build.jpg", np.vstack([r[:, :w] for r in rows]))
        save("real_setup.jpg", fit(cv2.imread(os.path.join(ROOM_PHOTOS, "12.jpg")), w=1280))

    # Real recordings next to the sim at the same joint angles (wrapped box at its kinematic start pose).
    kin_path = os.path.join(REPO, "03_real2sim", "calib", "kinematic_wrap.json")
    if os.path.isdir(REFS) and os.path.exists(kin_path):
        kin = json.load(open(kin_path))
        wq = s.m.jnt_qposadr[s.m.body_jntadr[s.m.body("env/wrapped_box").id]]
        rows = []
        for ep, f in ((0, 0), (23, 0), (41, 0), (0, 300)):
            state = np.load(os.path.join(REFS, f"ep{ep:03d}.npz"))["state"]
            mujoco.mj_resetData(s.m, s.d)
            s.d.qpos[ARM_QPOS] = state[min(f, len(state) - 1)]
            if f == 0:
                s.d.qpos[wq:wq + 2] = kin[str(ep)][:2]
            else:  # after the drop: the wrapped box is inside the cardboard box
                s.d.qpos[wq:wq + 3] = [0.42, 0.40, 0.75]
            mujoco.mj_forward(s.m, s.d)
            tiles = []
            for cam in CAMS:
                real = real_frame(ep, f, cam)
                if real is None:
                    break
                tiles += [label(fit(real, w=320), f"real {cam[4:]}"), label(fit(s.camera(cam), w=320), "sim")]
            else:
                rows.append(np.hstack(tiles))
        if rows:
            save("stage3_real_vs_sim.jpg", np.vstack(rows), quality=90)
    else:
        print("  (no calib/real_refs: skipped stage3_real_vs_sim.jpg)")

    # Domain randomisation: every reset changes object poses, lights, tints and camera poses a little.
    s = Scene(["--scene", "lab", "--randomize", "--seed", "3"])
    frames = []
    for k in range(10):
        s.sim.reset(robot=True)
        mujoco.mj_forward(s.m, s.d)
        frame = np.hstack([fit(s.view([0.45, 0.15, 0.9], 1.5, 150, -28, (640, 480)), h=300), fit(s.camera("cam_head"), h=300)])
        frames += [label(frame, f"--randomize: reset {k + 1}")] * 4
    save_gif("stage3_randomize.gif", frames, 5)


# ------------------------------------------------------------------------------------------ recorded session
def session(log_path):
    log = np.load(log_path)
    s = Scene(["--scene", "lab"])
    t, success = log["t"], log["success"].astype(bool)
    rises = np.flatnonzero(success[1:] & ~success[:-1]) + 1
    if not len(rises):
        print("  (no successful episode in the log: skipped the session GIFs)")
        return
    end = min(len(t) - 1, rises[0] + 75)             # 2.5 s after the success flag
    start = max(0, rises[0] - int(15.5 * 30))
    idx = np.arange(start, end, 5)                   # 6 samples per second of sim time

    def at(i):
        s.d.qpos[:] = log["qpos"][i]
        s.d.mocap_pos[:], s.d.mocap_quat[:] = log["mocap_pos"][i], log["mocap_quat"][i]
        s.m.body_pos[:], s.m.body_quat[:] = log["body_pos"][i], log["body_quat"][i]
        mujoco.mj_forward(s.m, s.d)

    hero, cams_only, sim_rows = [], [], []
    for i in idx:
        at(i)
        cams = [s.camera(c, (320, 240)) for c in CAMS]
        over = s.view([0.5, 0.12, 0.92], 1.75, 148, -24, (640, 480))
        col = np.vstack([fit(c, h=160) for c in cams])
        frame = np.hstack([over, col])
        if success[i]:
            label(frame, "success: box is in the box", org=(8, 470))
        hero.append(frame)
        cams_only.append(np.hstack([label(c.copy(), n) for c, n in zip(cams, ("head", "left wrist", "right wrist"))]))
        sim_rows.append(np.hstack([fit(c, w=240) for c in cams]))
    save_gif("hero.gif", hero, 9)
    save_gif("stage4_scripted_episode.gif", [fit(f, w=720) for f in cams_only], 9)

    # A real teleoperated episode above the sim one, both stretched over the same number of frames.
    color = sorted(glob.glob(os.path.join(REFS, "ep*_color.npz")))
    if color:
        z = np.load(color[0])
        real = {c: z[c] for c in CAMS}
        n = min(len(real[c]) for c in CAMS)
        frames = []
        for k, row in enumerate(sim_rows):
            j = int(k / max(1, len(sim_rows) - 1) * (n - 1))
            top = np.hstack([fit(real[c][j], w=240) for c in CAMS])
            frames.append(np.vstack([label(top, "real robot (teleoperated)"), label(row, "simulation (scripted)")]))
        save_gif("stage4_real_vs_sim_episode.gif", frames, 9)
    else:
        print("  (no calib/real_refs: skipped stage4_real_vs_sim_episode.gif)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=["stage1", "stage2", "stage3", "session"], default=None)
    p.add_argument("--log", default=None, help="session log from g1_xr_sim.py --log-qpos (lab scene)")
    a = p.parse_args()
    os.makedirs(MEDIA, exist_ok=True)
    for name, fn in (("stage1", stage1), ("stage2", stage2), ("stage3", stage3)):
        if a.only is None or name in a.only:
            print(name)
            fn()
    if a.log and (a.only is None or "session" in a.only):
        print("session")
        session(a.log)


if __name__ == "__main__":
    main()
