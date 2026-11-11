"""Wrapped-box start position of every real episode from forward kinematics (robot frame = sim world xy).

At the squeeze the box sits between the palms, so its centre is the palms' midpoint. Writes
calib/kinematic_wrap.json: {episode: [x, y, grasp_z, palm_separation]}.
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(SIM), "04_recording"))
import logging_mp  # noqa: E402

logging_mp.basicConfig(level=logging_mp.WARNING)
import pinocchio as pin  # noqa: E402
from g1lab.robot_io import _in_teleop_dir  # noqa: E402
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK  # noqa: E402

PALM = 0.07  # palm centre ahead of the IK's L_ee/R_ee frames (wrist yaw joint + 5 cm) along wrist x

with _in_teleop_dir():
    ik = G1_29_ArmIK()
model, data = ik.reduced_robot.model, ik.reduced_robot.data
fl, fr = model.getFrameId("L_ee"), model.getFrameId("R_ee")
out = {}
for f in sorted(glob.glob(os.path.join(SIM, "calib", "real_refs", "ep*.npz"))):
    ep = int(os.path.basename(f)[2:5])
    pl, pr = [], []
    for q in np.load(f)["state"]:
        pin.framesForwardKinematics(model, data, q.astype(float))
        pl.append(data.oMf[fl].translation + data.oMf[fl].rotation[:, 0] * PALM)
        pr.append(data.oMf[fr].translation + data.oMf[fr].rotation[:, 0] * PALM)
    pl, pr = np.array(pl), np.array(pr)
    mid, sep = (pl + pr) / 2, np.linalg.norm(pl - pr, axis=1)
    z = mid[:, 2]
    k = max(0, int(np.argmax(z > z[: len(z) // 2].min() + 0.10)) - 15)  # ~0.5 s before the lift
    out[ep] = [round(float(v), 4) for v in (*mid[k], sep[k])]
json.dump(out, open(os.path.join(SIM, "calib", "kinematic_wrap.json"), "w"), indent=1)
print(len(out), "episodes ->", os.path.join(SIM, "calib", "kinematic_wrap.json"))
