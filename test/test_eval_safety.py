"""Robot-free checks of the eval_g1.py safety helpers: python -m pytest test/test_eval_safety.py"""

import numpy as np

from unitree_lerobot.eval_robot.image_server.image_client import cameras_of
from unitree_lerobot.eval_robot.utils.utils import limit_command, ramp_to

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
