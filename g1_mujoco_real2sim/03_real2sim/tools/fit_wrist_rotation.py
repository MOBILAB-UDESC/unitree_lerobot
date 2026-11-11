"""Self-calibrate a D405 wrist camera (focal length = FOV, and its orientation on the wrist) from the
real recordings: the wrist's rotation between two frames is known exactly from the recorded joints
(forward kinematics), so the background's image motion between those frames - a pure-rotation
homography H = K Rm^T (Rw2^T Rw1) Rm K^-1 - pins down K (focal length) and Rm (camera-in-wrist rotation).

Frames: tools/extract_real_refs.py --video-cams (gray 320x240). Run in the g1lab env (pinocchio, scipy).
    python 03_real2sim/tools/fit_wrist_rotation.py
Prints the vertical FOV at 640x480 and the camera rotation (OpenCV and MuJoCo conventions).
"""

import glob
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
ROOT = os.path.dirname(SIM)
sys.path.insert(0, os.path.join(ROOT, "04_recording"))
import logging_mp  # noqa: E402

logging_mp.basicConfig(level=logging_mp.ERROR)
import pinocchio as pin  # noqa: E402
from g1lab.robot_io import _in_teleop_dir  # noqa: E402
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK  # noqa: E402

REFS = os.path.join(SIM, "calib", "real_refs")
W, H = 320, 240
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])  # OpenCV camera (x right, y down, z fwd) <-> MuJoCo (x right, y up, z back)


def wrist_poses(model, data, frame, states):
    fid = model.getFrameId(frame)
    Rs, ps = [], []
    for q in states:
        pin.framesForwardKinematics(model, data, q.astype(float))
        Rs.append(data.oMf[fid].rotation.copy())
        ps.append(data.oMf[fid].translation.copy())
    return np.array(Rs), np.array(ps)


def matches(img1, img2, orb, bf):
    mask1 = (cv2.GaussianBlur(img1, (5, 5), 0) > 70).astype(np.uint8) * 255  # skip the black glove
    mask2 = (cv2.GaussianBlur(img2, (5, 5), 0) > 70).astype(np.uint8) * 255
    k1, d1 = orb.detectAndCompute(img1, mask1)
    k2, d2 = orb.detectAndCompute(img2, mask2)
    if d1 is None or d2 is None or len(k1) < 30 or len(k2) < 30:
        return None
    pairs = [m for m, n in (x for x in bf.knnMatch(d1, d2, k=2) if len(x) == 2) if m.distance < 0.75 * n.distance]
    if len(pairs) < 30:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in pairs])
    p2 = np.float32([k2[m.trainIdx].pt for m in pairs])
    Hm, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
    if Hm is None or inl.sum() < 30:
        return None
    inl = inl.ravel().astype(bool)
    # Reject near-identity homographies (glove / static features dominating).
    if np.median(np.linalg.norm(p2[inl] - p1[inl], axis=1)) < 3.0:
        return None
    return p1[inl], p2[inl]


SEPARATE_F = os.environ.get("SEPARATE_F") == "1"
# Datasheet-fixed intrinsics (focal length in px at 640x480, no distortion: RealSense 400-series): only fit
# the camera orientation. D405 640x480: fx = fy ~ 385 (87x58 deg at 1280x720 from a 1280x800 sensor, x0.6).
FIXED_F640 = float(os.environ["FIXED_F"]) if os.environ.get("FIXED_F") else None
LEVER = np.array([0.0214, 0.0077, 0.107])  # D405 cradle in the wrist yaw frame (right; y mirrored for left)
MIRROR = np.diag([1.0, -1.0, 1.0])         # robot left/right mirror (y)
# The mirrored right camera has its image flipped vertically relative to a pure mirror image (in the
# real videos the glove is at the top of the left image and at the bottom of the right one).
RIGHT_IMAGE_FLIP = np.diag([1.0, -1.0, 1.0])


def collect(side, model, data, orb, bf, max_cam_move=float(os.environ.get("MAX_CAM_MOVE", 0.015))):
    cam = f"cam_{side}_wrist"
    frame = "L_ee" if side == "left" else "R_ee"
    lever = LEVER * ([1, -1, 1] if side == "left" else [1, 1, 1])
    obs = []
    for f in sorted(glob.glob(os.path.join(REFS, f"ep*_{cam}_video.npz"))):
        ep = int(os.path.basename(f)[2:5])
        frames = np.load(f)["frames"]
        Rs, ps = wrist_poses(model, data, frame, np.load(os.path.join(REFS, f"ep{ep:03d}.npz"))["state"])
        cams = ps + np.einsum("nij,j->ni", Rs, lever)
        n = min(len(frames), len(Rs))
        for t in range(0, n - 8, 2):
            for k in (2, 3, 5, 8):
                Rrel = Rs[t + k].T @ Rs[t]
                ang = np.degrees(np.linalg.norm(Rotation.from_matrix(Rrel).as_rotvec()))
                if not 3.0 <= ang <= 15.0 or np.linalg.norm(cams[t + k] - cams[t]) > max_cam_move:
                    continue
                m = matches(frames[t], frames[t + k], orb, bf)
                if m is not None:
                    obs.append((Rrel, *m))
                break
    return obs


def main():
    with _in_teleop_dir():
        ik = G1_29_ArmIK()
    model, data = ik.reduced_robot.model, ik.reduced_robot.data
    orb = cv2.ORB_create(2000, fastThreshold=10)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    obs = {s: collect(s, model, data, orb, bf) for s in ("left", "right")}
    for s, o in obs.items():
        print(f"{s}: {len(o)} frame pairs, {sum(len(x[1]) for x in o)} matches", flush=True)
    c = np.array([W / 2, H / 2])

    def undistort(p, f, k1):  # one radial term, applied to normalised coordinates; f = (fx, fy)
        u = (p - c) / f
        r2 = (u ** 2).sum(1, keepdims=True)
        return u / (1 + k1 * r2)

    def distort(u, f, k1):
        r2 = (u ** 2).sum(1, keepdims=True)
        return u * (1 + k1 * r2) * f + c

    def residuals(x):
        f, k1 = np.exp(x[[0, 5]]) if SEPARATE_F else np.exp(x[[0, 0]]), x[1]
        if FIXED_F640:
            f, k1 = np.array([FIXED_F640 / 2] * 2), 0.0
        Rl = Rotation.from_rotvec(x[2:5]).as_matrix()
        Rr = MIRROR @ Rl @ RIGHT_IMAGE_FLIP  # mirrored mount; real right image is the left one rotated 180 deg
        res = []
        for Rm, o in ((Rl, obs["left"]), (Rr, obs["right"])):
            for Rrel, p1, p2 in o:
                u = undistort(p1, f, k1)
                rays = np.c_[u, np.ones(len(u))]
                r2 = (Rm.T @ Rrel @ Rm @ rays.T).T
                res.append((distort(r2[:, :2] / r2[:, 2:3], f, k1) - p2).ravel())
        return np.concatenate(res)

    best = None
    for vfov in (55, 64, 75):
        f0 = (H / 2) / np.tan(np.radians(vfov) / 2)
        for seed in range(16):
            x0 = np.r_[np.log(f0), 0.0, Rotation.random(random_state=seed).as_rotvec(), np.log(f0)]
            sol = least_squares(residuals, x0, loss="soft_l1", f_scale=2.0, max_nfev=300)
            cost = np.median(np.abs(sol.fun))
            if best is None or cost < best[0]:
                best = (cost, sol.x)
    cost, x = best
    fx, fy = np.exp(x[0]), np.exp(x[5] if SEPARATE_F else x[0])
    print(f"fx {fx * 2:.1f} px, fy {fy * 2:.1f} px at 640x480 -> horizontal FOV {np.degrees(2 * np.arctan((W / 2) / fx)):.1f}, "
          f"vertical FOV {np.degrees(2 * np.arctan((H / 2) / fy)):.1f} deg (aspect fy/fx {fy / fx:.2f})")
    f, k1 = fy, x[1]
    Rl = Rotation.from_rotvec(x[2:5]).as_matrix()
    Rr = MIRROR @ Rl @ RIGHT_IMAGE_FLIP
    vfov = np.degrees(2 * np.arctan((H / 2) / f))
    hfov = np.degrees(2 * np.arctan((W / 2) / f))
    print(f"median reprojection error {cost:.2f} px (320x240), k1 {k1:+.3f}")
    print(f"focal {f * 2:.1f} px at 640x480 -> vertical FOV {vfov:.1f} deg, horizontal {hfov:.1f} deg")
    for side, Rm in (("left", Rl), ("right", Rr)):
        q = Rotation.from_matrix(Rm @ CV_TO_MJ).as_quat()
        print(f"{side}: optical axis in wrist frame {np.round(Rm[:, 2], 3)}, image down {np.round(Rm[:, 1], 3)}")
        print(f"WRIST_CAM_CALIB['{side}'] = dict(fovy={vfov:.2f}, quat=[{q[3]:.5f}, {q[0]:.5f}, {q[1]:.5f}, {q[2]:.5f}])")


if __name__ == "__main__":
    main()
