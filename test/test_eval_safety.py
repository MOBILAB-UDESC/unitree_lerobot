"""Robot-free checks of the eval_g1.py safety helpers: python -m pytest test/test_eval_safety.py"""

import threading
import time

import numpy as np

from unitree_lerobot.eval_robot.image_server.image_client import cameras_of
from unitree_lerobot.eval_robot.utils.utils import (
    AsyncChunks,
    format_start_pose,
    limit_command,
    ramp_to,
    release_arms,
)

CAMS = {"head_camera": {"enable_zmq": True, "zmq_port": 55555}, "left_wrist_camera": {"enable_zmq": False}}


def test_cameras_of_reads_teleimager_1_and_2_configs():
    assert cameras_of(CAMS) is CAMS  # 1.x: cameras at the top level
    assert cameras_of({"webrtc": {"bitrate": {}}, "camera": CAMS}) is CAMS  # 2.x: under "camera"
    assert cameras_of(None) is None


def run_ramp(start, target, max_vel, frequency, lag=0.0):
    """ramp_to against a fake robot whose joints follow the command, `lag` rad behind."""
    sent, measured = [], np.array(start, dtype=float)

    def send(q):
        nonlocal measured
        sent.append(q.copy())
        measured = q - lag * np.sign(q - measured)

    cmd = ramp_to(target, lambda: measured, send, max_vel, frequency, timeout=10.0, sleep=lambda _: None)
    return cmd, np.array(sent)


def test_ramp_to_never_changes_the_command_faster_than_max_vel():
    start, target = np.zeros(3), np.array([1.0, -0.5, 0.2])
    cmd, sent = run_ramp(start, target, max_vel=0.5, frequency=30)
    steps = np.abs(np.diff(np.vstack([start, sent]), axis=0))
    assert steps.max() <= 0.5 / 30 + 1e-12
    np.testing.assert_allclose(cmd, target)
    assert len(sent) >= int(np.ceil(1.0 / (0.5 / 30)))  # the 1 rad joint needs at least 2 s at 0.5 rad/s


def test_ramp_to_starts_from_the_measured_pose_not_zero():
    start = np.array([0.8, -0.3])
    _, sent = run_ramp(start, start + 0.01, max_vel=1.0, frequency=30)
    assert np.abs(sent[0] - start).max() <= 1.0 / 30 + 1e-12


def test_ramp_to_gives_up_after_timeout_when_joints_dont_arrive():
    def send(_):
        pass

    cmd = ramp_to(np.ones(2), lambda: np.zeros(2), send, max_vel=1.0, frequency=30, timeout=2.0, sleep=lambda _: None)
    np.testing.assert_allclose(cmd, np.ones(2))  # the command got there, the (stuck) joints did not


def test_limit_command_keeps_lead_and_speed():
    prev = measured = np.zeros(2)
    out = limit_command(np.array([1.0, -1.0]), prev, measured, max_vel=1.0, max_lead=0.25, frequency=10)
    np.testing.assert_allclose(out, [0.1, -0.1])


class FakeArms:
    """Stands in for G1_29_ArmController: the measured pose follows the command exactly."""

    def __init__(self, initial, now):
        self.initial_arm_q = np.array(initial, dtype=float)
        self.q = np.array(now, dtype=float)
        self.events = []  # ("q", command) and ("w", weight), in order

    def get_current_dual_arm_q(self):
        return self.q.copy()

    def ctrl_dual_arm(self, q, tau):
        self.q = np.array(q, dtype=float)
        self.events.append(("q", self.q.copy()))

    def set_arm_sdk_weight(self, weight):
        self.events.append(("w", weight))


class FakeIk:
    def solve_tau(self, q):
        return np.zeros_like(q)


def test_release_arms_returns_to_the_initial_pose_slowly_then_lowers_the_weight():
    initial = np.zeros(3)
    arms = FakeArms(initial, now=[0.6, -0.4, 0.2])
    release_arms(arms, FakeIk(), max_vel=0.5, frequency=30, release_seconds=2.0, sleep=lambda _: None)

    kinds = [k for k, _ in arms.events]
    assert kinds == sorted(kinds, key=lambda k: k != "q"), "all moves must come before the first weight change"
    moves = np.array([v for k, v in arms.events if k == "q"])
    weights = [v for k, v in arms.events if k == "w"]

    steps = np.abs(np.diff(np.vstack([[0.6, -0.4, 0.2], moves]), axis=0))
    assert steps.max() <= 0.5 / 30 + 1e-12  # never faster than max_vel
    np.testing.assert_allclose(moves[-1], initial)
    assert len(weights) == 60 and weights[-1] == 0.0
    assert all(a > b for a, b in zip(weights, weights[1:])) and weights[0] < 1.0


def test_release_arms_default_speed_when_max_vel_is_off():
    arms = FakeArms(np.zeros(2), now=[0.3, 0.3])
    release_arms(arms, FakeIk(), max_vel=0, frequency=30, release_seconds=1.0, sleep=lambda _: None)
    moves = np.array([v for k, v in arms.events if k == "q"])
    assert np.abs(np.diff(np.vstack([[0.3, 0.3], moves]), axis=0)).max() <= 0.5 / 30 + 1e-12


def test_async_chunks_close_waits_for_the_planning_thread_and_starts_no_more():
    calls, finished = [], threading.Event()

    def plan(observation):
        calls.append(len(calls))
        if len(calls) > 1:  # the first plan is computed in the caller, the later ones in the worker thread
            time.sleep(0.3)
            finished.set()
        return np.zeros((5, 2))

    actor = AsyncChunks(plan)
    actor({"x": 1})  # first call: plan() inline, then a worker thread starts planning the next chunk
    assert not finished.is_set()
    actor.close()
    assert finished.is_set(), "close() must wait for the planning thread"
    assert not actor.worker.is_alive()

    n = len(calls)
    actor({"x": 1})
    actor.close()
    assert len(calls) == n, "no new planning thread after close()"


def test_format_start_pose_lists_every_joint_with_its_move():
    text = format_start_pose(["kA", "kB"], np.array([0.1, 1.0]), np.array([0.0, -0.5]))
    assert "kA" in text and "kB" in text and "move +0.10" in text and "move +1.50" in text
