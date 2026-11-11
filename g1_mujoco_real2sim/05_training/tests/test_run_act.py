"""run_act without torch/lerobot/DDS: safety gate, ramp, feature parsing and the closed loop with fakes.

    source scripts/env.sh g1lab && python -m pytest 05_training/tests -v -p no:cacheprovider
"""

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_act as ra  # noqa: E402

CAMS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
IMG = np.zeros((480, 640, 3), np.uint8)


def feat(*shape):
    return types.SimpleNamespace(shape=shape)


def fake_config(n=15, cams=CAMS):
    inp = {"observation.state": feat(n), **{f"observation.images.{c}": feat(3, 480, 640) for c in cams}}
    return types.SimpleNamespace(input_features=inp, output_features={"action": feat(n)})


class FakeRobot:
    """Measured pose follows the command instantly (ideal tracking); success after `success_after` commands."""

    sim = True

    def __init__(self, n=15, q0=None, success_after=None, drop_cam_every=0):
        self.q = np.zeros(n) if q0 is None else np.asarray(q0, float)
        self.sent = []
        self.success_after, self.drop = success_after, drop_cam_every
        self.ticks = 0

    def state(self):
        return self.q.astype(np.float32)

    def observe(self):
        self.ticks += 1
        imgs = {c: IMG for c in CAMS}
        if self.drop and self.ticks % self.drop == 0:
            imgs.pop("cam_left_wrist")
        return {"state": self.state(), "images": imgs}

    def command(self, q):
        q = np.asarray(q, float)
        self.sent.append(q.copy())
        self.q[: q.size] = q

    def success(self):
        return self.success_after is not None and len(self.sent) >= self.success_after


class ConstPolicy:
    def __init__(self, target):
        self.target, self.calls, self.resets = np.asarray(target, float), 0, 0

    def reset(self):
        self.resets += 1

    def __call__(self, state, images):
        assert set(images) == set(CAMS) and state.dtype == np.float32
        self.calls += 1
        return self.target


class FakeWriter:
    def __init__(self):
        self.frames, self.saved, self.discarded = [], 0, 0

    @property
    def num_frames(self):
        return len(self.frames)

    def start_episode(self, task=None):
        self.frames = []

    def add_frame(self, s, a, imgs):
        self.frames.append((s, a))

    def save_episode(self):
        self.saved += 1
        self.frames = []
        return self.saved - 1

    def discard_episode(self):
        self.discarded += 1
        self.frames = []


def make_runner(robot, policy, n=15, **kw):
    gate = ra.SafetyGate(*ra.joint_limits(n), max_vel=kw.pop("max_vel", 1.0), max_lead=kw.pop("max_lead", 0.25))
    return ra.Runner(robot, policy, n, gate, CAMS, sleep=lambda s: None, log=lambda *a: None,
                     ramp_time=kw.pop("ramp_time", 0.5), **kw)


# ---------------------------------------------------------------------------------------- pieces
def test_policy_io():
    assert ra.policy_io(fake_config(15)) == (15, sorted(CAMS), 15)
    assert ra.policy_io(fake_config(14, cams=["cam_head"])) == (14, ["cam_head"], 14)
    bad = fake_config(15)
    bad.output_features = {"action": feat(16)}
    with pytest.raises(ValueError):
        ra.policy_io(bad)


def test_joint_limits_add_waist():
    lo14, hi14 = ra.joint_limits(14)
    lo, hi = ra.joint_limits(15)
    assert lo.shape == (15,) and np.allclose(lo[:14], lo14) and hi[14] == pytest.approx(ra.WAIST_YAW_LIMIT)


def test_gate_limits_lead_velocity_and_nan():
    g = ra.SafetyGate(-np.ones(2), np.ones(2), max_vel=3.0, max_lead=0.2, hz=30)   # 0.1 rad per tick
    prev = meas = np.zeros(2)
    out = g(np.array([5.0, 0.05]), prev, meas)
    assert np.allclose(out, [0.1, 0.05]) and g.counts["limit"] == 1 and g.counts["lead"] == 1 and g.counts["vel"] == 1
    out = g(np.array([0.5, 0.0]), np.array([0.15, 0.0]), np.array([0.0, 0.0]))     # lead: <= measured + 0.2
    assert out[0] == pytest.approx(0.2)
    assert np.allclose(g(np.array([np.nan, 0.0]), np.array([0.3, 0.3]), meas), [0.3, 0.3])
    assert g.counts["rejected"] == 1


def test_ramp_is_slow_smooth_and_ends_on_target():
    wp = ra.ramp(np.zeros(3), np.array([1.0, -0.5, 0.0]), hz=30, min_time=0.5, max_vel=0.5)
    assert np.allclose(wp[-1], [1.0, -0.5, 0.0])
    step = np.abs(np.diff(np.vstack([np.zeros(3), wp]), axis=0)).max()
    assert step <= 0.5 / 30 + 1e-9                          # peak speed respected
    assert len(wp) >= int(np.pi / 2 * 1.0 / 0.5 * 30)        # ~3.1 s for 1 rad at 0.5 rad/s
    assert len(ra.ramp(np.zeros(2), np.zeros(2), min_time=1.0)) == 30


def test_image_hold():
    h = ra.ImageHold(CAMS)
    assert h.update({"cam_head": IMG}) is None
    full = h.update({c: IMG for c in CAMS})
    assert set(full) == set(CAMS)
    assert set(h.update({"cam_head": IMG})) == set(CAMS)    # missing ones are held


# ------------------------------------------------------------------------------------------ loop
def test_episode_ramps_then_tracks_within_limits():
    target = np.full(15, 0.3)
    target[14] = 0.5
    robot, pol, w = FakeRobot(), ConstPolicy(target), FakeWriter()
    r = make_runner(robot, pol, writer=w)
    st = r.run_episode(max_steps=20)
    assert st["reason"] == "max_steps" and st["steps"] == 20 and len(w.frames) == 20
    sent = np.array(robot.sent)
    assert np.abs(np.diff(sent, axis=0)).max() <= 0.5 / 30 + 1e-6        # ramp (0.5 rad/s) then gate (1 rad/s)
    assert np.allclose(sent[-1], target, atol=1e-6) and pol.resets >= 2   # reset again after the ramp
    assert np.allclose(w.frames[-1][1], sent[-1])                         # recorded action = command sent


def test_success_stops_and_dry_run_sends_nothing():
    robot = FakeRobot(success_after=80)
    st = make_runner(robot, ConstPolicy(np.zeros(15))).run_episode(max_steps=500)
    assert st["reason"] == "success" and st["steps"] < 500
    robot = FakeRobot()
    st = make_runner(robot, ConstPolicy(np.ones(15)), send=False).run_episode(max_steps=10)
    assert robot.sent == [] and st["steps"] == 10


def test_motion_mode_does_not_drive_waist_and_stop_flag():
    robot = FakeRobot()
    r = make_runner(robot, ConstPolicy(np.full(15, 0.1)), drive_waist=False)
    r.run_episode(max_steps=5)
    assert all(q.size == 14 for q in robot.sent)
    r.stop = True
    st = r.run_episode(max_steps=5)
    assert st["reason"] == "interrupted"
    q = r.hold()
    assert np.allclose(q, robot.q[:15])


def test_missing_camera_frames_are_held():
    robot = FakeRobot(drop_cam_every=3)
    st = make_runner(robot, ConstPolicy(np.zeros(15))).run_episode(max_steps=30)
    assert st["steps"] == 30


def test_finish_recording_policy():
    w = FakeWriter()
    w.frames = [1]
    ra.finish_recording(w, {"reason": "max_steps"}, "success", log=lambda *a: None)
    assert w.discarded == 1
    w.frames = [1]
    ra.finish_recording(w, {"reason": "success"}, "success", log=lambda *a: None)
    assert w.saved == 1


def test_constant_state_joints():
    step = types.SimpleNamespace(stats={"observation.state": {"std": np.array([0.2, 0.0, 0.1])}})
    pre = types.SimpleNamespace(steps=[types.SimpleNamespace(), step])
    assert ra.constant_state_joints(pre) == [1]
    assert ra.constant_state_joints(types.SimpleNamespace(steps=[])) == []


def test_async_chunks_never_block_and_stay_aligned_in_time():
    import threading
    release, calls = threading.Event(), []

    def plan(state, images):          # chunk k holds the values 100 * k + step
        k = len(calls)
        calls.append(k)
        if k:                         # every plan after the first one waits until the test lets it finish
            release.wait(5)
        return 100.0 * k + np.arange(10.0)[:, None] * np.ones((1, 2))

    actor = ra.AsyncChunks(plan)
    s, im = np.zeros(2), {"cam": np.zeros((2, 2, 3), np.uint8)}
    assert [actor(s, im)[0] for _ in range(4)] == [0, 1, 2, 3]   # plays chunk 0 while plan 1 is being computed
    release.set()
    for _ in range(100):
        if actor.ready is not None:
            break
        import time
        time.sleep(0.01)
    assert actor(s, im)[0] == 104     # plan 1 saw tick 0, four ticks passed: enter it at its step 4
    actor.reset()
    assert actor(s, im)[0] % 100 == 0  # a new episode starts at step 0 of a fresh plan
