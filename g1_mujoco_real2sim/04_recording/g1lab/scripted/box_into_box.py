"""Code-as-policy demo generator: put the bubble-wrapped box into the cardboard box (lab sim).

Closed loop on the sim's privileged object poses (rt/sim_state), moving the arms through
xr_teleoperate's own IK, in the rhythm of the real teleop episodes (~12 s):
    reach beside the box -> lower -> squeeze -> lift over the cardboard wall while turning the waist
    toward the cardboard box -> carry -> lower inside -> release -> back to rest, waist back to 0.
Turning the waist (hip yaw, legs still) moves both shoulders left: without it the right shoulder hits
the torso before the box is over the cardboard box. Recorded in the 15-joint "ArmsWaist" format
(14 arm joints + kWaistYaw). Only episodes ending with the sim's success check are kept.

Every episode is varied (sim2real): start arm/waist pose, speed, squeeze width, grasp height, lift
clearance, drop waist angle, waypoint jitter; the sim adds object poses and visuals with --randomize.

    scripts/run_sim.sh --scene lab --randomize --no-viewer
    python -m g1lab.scripted.box_into_box --episodes 20 --root datasets/sim_scripted_waist --repo-id local/sim_scripted_waist

The same structure is the template for Claude-written policies: everything goes through G1Arms, so a
policy written here runs unchanged against the real robot's channels (minus the privileged sim state).
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from g1lab.robot_io import G1Arms, rot_z  # noqa: E402

TASK = "put the wrapped box inside the cardboard box"
PELVIS_Z = 0.793          # sim world -> robot frame: z_robot = z_world - PELVIS_Z (pelvis fixed at x=y=0)
PALM_FORWARD = 0.06       # palm centre ahead of the IK wrist frame (L_ee/R_ee) along its x axis
PALM_INSET = 0.02         # palm surface ~2 cm from the wrist frame toward the palm
WRAP_HALF = 0.115         # wrapped box half width (23 cm)
WALL_H = 0.18             # cardboard box height
BOX_INNER_HALF = np.array([0.161, 0.241])
OPEN_WIDTH = 0.36
GRAVITY_SCALE = 1.0       # full IK gravity feed-forward (as xr_teleoperate sends): carrying the box needs it
TORSO_REACH_X = 0.50      # coarse reach bound (torso frame); the IK check below decides
TORSO_MAX_Y = 0.10        # further left the right shoulder hits the torso (torso frame)
REST_JITTER = 0.08        # rad, start-pose variation (real starts: std ~0.1)


def sample_params(rng):
    """Per-episode motion variation."""
    return dict(
        speed=rng.uniform(0.85, 1.15),
        squeeze=rng.uniform(0.13, 0.15),  # commanded palm gap: the arm PD (xr gains) stops ~6 cm short,
                                          # which leaves a firm ~1-2 cm press into the bubble wrap
        grasp_dz=rng.uniform(-0.015, 0.015),
        lift_extra=rng.uniform(0.03, 0.07),
        jitter=0.006,
        waist_scale=rng.uniform(0.9, 1.1),
    )


def poses(width, center, yaw=0.0):
    """Wrist targets in the pelvis frame: palms facing each other across the turned torso's y axis."""
    R = rot_z(yaw)
    out = []
    for side in (1, -1):
        T = R.copy()
        T[:3, 3] = np.asarray(center) + R[:3, :3] @ [-PALM_FORWARD, side * (width / 2 + PALM_INSET), 0.0]
        out.append(T)
    return out


class Episode:
    def __init__(self, robot, writer, params, rng, rate=30.0):
        self.robot, self.writer, self.p, self.rng, self.rate = robot, writer, params, rng, rate
        self.frames = 0

    def record(self, _i=None):
        if self.writer is None:
            return
        obs = self.robot.observe()
        if len(obs["images"]) == 3:
            self.writer.add_frame(obs["state"], self.robot.last_command(), obs["images"])
            self.frames += 1

    def go(self, width, center, duration, yaw=0.0):
        """Move both palms (and the waist) to a squeeze configuration, recording every control tick."""
        center = np.asarray(center, float) + self.rng.normal(0, self.p["jitter"], 3)
        L, R = poses(width, center, yaw)
        q, tau = self.robot.ik(L, R, waist=yaw)
        self.robot.move_to(np.append(q, yaw), duration / self.p["speed"], self.rate, on_step=self.record)
        self.robot.command(np.append(q, yaw), tau * GRAVITY_SCALE)

    def turn_waist(self, yaw, duration):
        """Rotate the waist with the arm joints frozen: the torso carries the box rigidly, so the grip
        geometry can't change (moving the arms while the waist turns makes the box roll in the grip)."""
        q = self.robot.last_command().astype(float)
        q[14] = yaw
        self.robot.move_to(q, duration / self.p["speed"], self.rate, on_step=self.record)

    def hold(self, seconds):
        t = time.perf_counter()
        for _ in range(int(seconds / self.p["speed"] * self.rate)):
            self.record()
            t += 1.0 / self.rate
            time.sleep(max(0.0, t - time.perf_counter()))


def object_frames(robot):
    st = None
    for _ in range(50):
        st = robot.sim_state()
        if st and "wrapped_box" in st and "cardboard_box" in st:
            break
        time.sleep(0.05)
    if not st:
        raise RuntimeError("no rt/sim_state from the sim - is it running with --scene lab on this domain?")
    wrap = np.array(st["wrapped_box"]["pos"]) - [0, 0, PELVIS_Z]   # bottom-face centre, pelvis frame
    box = np.array(st["cardboard_box"]["pos"]) - [0, 0, PELVIS_Z]  # floor centre, pelvis frame
    return wrap, box


def plan_drop(robot, box, z, width):
    """Drop point over the open cardboard box and the waist yaw that faces it, so the arms only extend
    straight ahead in the turned torso frame. Takes the deepest point (toward the box centre) that both
    squeezing arms reach per IK; returns (centre, waist yaw)."""
    x = float(np.clip(box[0] - BOX_INNER_HALF[0] + WRAP_HALF + 0.01, 0.38, 0.45))
    y_full = box[1] - BOX_INNER_HALF[1] + WRAP_HALF + 0.02    # fully inside the right wall
    for y in np.arange(y_full + 0.04, y_full - 0.14, -0.01):
        yaw = float(np.arctan2(y, x))
        L, R = poses(width, [x, y, z], yaw)
        q, _ = robot.ik(L, R, waist=yaw)
        FL, FR = robot.fk(q, waist=yaw)
        if max(np.linalg.norm(FL[:3, 3] - L[:3, 3]), np.linalg.norm(FR[:3, 3] - R[:3, 3])) < 0.015:
            return np.array([x, y, z]), yaw
    return np.array([x, y_full - 0.14, z]), float(np.arctan2(y_full - 0.14, x))


def go_to_start(robot, rest_q, rng):
    """Varied start pose (not recorded): rest arms + jitter, waist near 0."""
    q = np.asarray(rest_q, float).copy()
    q[:14] += rng.normal(0, REST_JITTER, 14)
    q[14] = rng.normal(0, 0.03)
    robot.move_to(q, 1.2)


def run_episode(robot, writer, rest_q, rng, task=TASK):
    p = sample_params(rng)
    wrap, box = object_frames(robot)
    # Palms near the box's centre of mass (13.5 cm up): little swing about the squeeze axis, and the
    # carry stays low enough for the arms to reach into the cardboard box. The left hand fits in the
    # ~9 cm gap between the boxes (measured layout).
    grasp_above = 0.15 + p["grasp_dz"]
    z_grasp = wrap[2] + grasp_above
    lift_z = box[2] + WALL_H + p["lift_extra"] + grasp_above
    target, yaw = plan_drop(robot, box, lift_z, p["squeeze"])
    yaw *= p["waist_scale"]
    inside = target[1] - WRAP_HALF > box[1] - BOX_INNER_HALF[1]
    drop_z = box[2] + (0.03 if inside else WALL_H + 0.02) + grasp_above
    cx = min(wrap[0], 0.42)
    ep = Episode(robot, writer, p, rng)
    y0 = float(robot.last_command()[14])
    if writer:
        writer.start_episode(task)
    ep.hold(0.4)
    ep.go(OPEN_WIDTH, [cx - 0.10, wrap[1], z_grasp + 0.16], 1.8, y0)   # reach: hands open, high and back
    ep.go(OPEN_WIDTH, [cx - 0.02, wrap[1], z_grasp + 0.10], 0.9, 0.0)  # come over the sides of the box
    ep.go(OPEN_WIDTH - 0.06, [cx, wrap[1], z_grasp], 1.0, 0.0)        # lower beside the box
    ep.go(p["squeeze"], [cx, wrap[1], z_grasp], 0.9, 0.0)             # squeeze
    ep.hold(0.3)
    ep.go(p["squeeze"], [cx - 0.02, wrap[1], lift_z], 1.6, 0.0)       # lift straight up, waist still
    ep.turn_waist(yaw, 2.6)                                            # turn the torso slowly, arms frozen
    ep.go(p["squeeze"], target, 1.8, yaw)                              # extend the arms over the open box
    ep.go(p["squeeze"], [target[0], target[1], drop_z], 1.0, yaw)     # lower inside
    ep.go(OPEN_WIDTH, [target[0], target[1], drop_z + 0.02], 0.7, yaw)  # release
    ep.go(OPEN_WIDTH, [target[0] - 0.06, target[1], lift_z + 0.05], 0.9, yaw)  # hands up and out
    robot.move_to(rest_q, 1.8 / p["speed"], ep.rate, on_step=ep.record)  # back to rest, waist to 0
    ep.hold(0.6)
    return ep.frames, dict(yaw=round(yaw, 2), target=np.round(target, 3).tolist(), inside=bool(inside))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=10, help="successful episodes to collect")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--root", default=None, help="LeRobot dataset root (omit to run without recording)")
    p.add_argument("--repo-id", default="local/sim_scripted_waist")
    p.add_argument("--task", default=TASK)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--domain", type=int, default=None)
    p.add_argument("--image-port", type=int, default=60000)
    a = p.parse_args()

    robot = G1Arms("sim", domain=a.domain, image_port=a.image_port, waist=True)
    rest_q = np.append(robot.state()[:14], 0.0)
    writer = None
    if a.root:
        from g1lab.lerobot_writer import LeRobotWriter
        writer = LeRobotWriter(a.root, a.repo_id, a.task, waist=True)
    rng = np.random.default_rng(a.seed)
    ok = attempts = 0
    max_attempts = a.max_attempts or 3 * a.episodes
    try:
        while ok < a.episodes and attempts < max_attempts:
            attempts += 1
            robot.reset_sim()
            time.sleep(0.8)
            go_to_start(robot, rest_q, rng)
            frames, info = run_episode(robot, writer, rest_q, rng, a.task)
            time.sleep(0.8)
            success = robot.success()
            if writer:
                (writer.save_episode if success else writer.discard_episode)()
            ok += success
            print(f"[scripted] attempt {attempts}: {'SUCCESS' if success else 'fail'} ({frames} frames, {info}) "
                  f"-> {ok}/{a.episodes}", flush=True)
    finally:
        if writer:
            writer.close()
    print(f"[scripted] {ok} successful of {attempts} attempts ({100 * ok / max(1, attempts):.0f}%)", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
