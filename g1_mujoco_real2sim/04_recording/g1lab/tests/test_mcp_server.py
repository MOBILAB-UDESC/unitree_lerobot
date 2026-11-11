"""Tests for g1lab/mcp_server.py.

    env -u PYTHONPATH -u LD_LIBRARY_PATH -u CYCLONEDDS_URI ~/miniforge3/envs/g1sim/bin/python -m pytest g1lab/tests/test_mcp_server.py -v

The unit tests use a fake robot. The sim tests drive a running sim over stdio through scripts/run_mcp.sh;
they run only when G1_MCP_TEST_DOMAIN is set, e.g. with a sim started as (env g1sim)
    scripts/run_sim.sh --scene lab --no-viewer --domain 4 --port-offset 2000
    G1_MCP_TEST_DOMAIN=4 G1_MCP_TEST_IMAGE_PORT=62000 python -m pytest g1lab/tests/test_mcp_server.py -v
"""

import asyncio
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from g1lab import mcp_server as M  # noqa: E402

RUN_MCP = ROOT / "scripts" / "run_mcp.sh"
SIM_DOMAIN = os.environ.get("G1_MCP_TEST_DOMAIN")
SIM_IMAGE_PORT = os.environ.get("G1_MCP_TEST_IMAGE_PORT", "60000")
needs_sim = pytest.mark.skipif(not SIM_DOMAIN, reason="set G1_MCP_TEST_DOMAIN (and G1_MCP_TEST_IMAGE_PORT) "
                                                      "to run against a running sim")


# ----------------------------------------------------------------------------- fake robot
class FakeRobot:
    """Joints 0-2 / 7-9 *are* the left / right wrist xyz; reach is limited to 0.7 m from the origin."""

    REACH = 0.7

    def __init__(self):
        self.q = np.zeros(14)
        self.q[0:3] = [0.2, 0.15, 0.1]
        self.q[7:10] = [0.2, -0.15, 0.1]
        self.commands = []
        self.resets = 0
        self.sim = True

    def state(self):
        return self.q.astype(np.float32)

    def last_command(self):
        return self.q.astype(np.float32)

    def command(self, q, tau=None):
        self.q = np.asarray(q, dtype=float).copy()
        self.commands.append(self.q)

    def images(self):
        return {c: np.full((480, 640, 3), 40 * i, np.uint8) for i, c in enumerate(M.CAMERAS)}

    def observe(self):
        return {"state": self.state(), "images": self.images()}

    def fk(self, q=None):
        q = self.q if q is None else np.asarray(q)
        return M.make_T(q[0:3], np.eye(3)), M.make_T(q[7:10], np.eye(3))

    def ik(self, left_T, right_T):
        q = self.q.copy()
        for sl, T in ((slice(0, 3), left_T), (slice(7, 10), right_T)):
            p = np.asarray(T)[:3, 3]
            n = np.linalg.norm(p)
            q[sl] = p if n <= self.REACH else p * self.REACH / n
        return q, np.zeros(14)

    def sim_state(self):
        return {"wrapped_box": {"pos": [0.43, -0.05, 0.7405], "quat": [1, 0, 0, 0]},
                "cardboard_box": {"pos": [0.445, 0.39, 0.7405], "quat": [1, 0, 0, 0]}, "success": False}

    def go_home(self):
        self.q = np.zeros(14)

    def reset_sim(self, category=1):
        self.resets += 1


@pytest.fixture
def session(tmp_path):
    robot = FakeRobot()
    s = M.G1Session("sim", robot_factory=lambda: robot, dataset_root=str(tmp_path / "ds"))
    s._settle = lambda seconds=0.0: None
    s.connect()
    s.fake = robot
    return s


# ----------------------------------------------------------------------------- pure helpers
def test_rpy_roundtrip():
    for rpy in ([0, 0, 0], [10, -20, 30], [-170, 45, 90], [5, 80, -120]):
        np.testing.assert_allclose(M.R_to_rpy(M.rpy_to_R(rpy)), rpy, atol=1e-6)
    # yaw +90 turns x into y
    np.testing.assert_allclose(M.rpy_to_R([0, 0, 90]) @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_quat_to_R():
    yaw = np.radians(30)
    R = M.quat_to_R([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    np.testing.assert_allclose(R, M.rpy_to_R([0, 0, 30]), atol=1e-12)


def test_clamp_workspace():
    p, clamped = M.clamp_workspace([0.9, -0.7, 0.0])
    np.testing.assert_allclose(p, [0.65, -0.5, 0.0])
    assert len(clamped) == 2
    p, clamped = M.clamp_workspace([0.3, 0.1, 0.1])
    assert clamped == [] and np.allclose(p, [0.3, 0.1, 0.1])


def test_squeeze_targets_symmetric_and_facing():
    L, R = M.squeeze_targets(0.23, [0.4, -0.05, 0.08])
    gap = L[1, 3] - R[1, 3]
    assert gap == pytest.approx(0.23 + 2 * M.PALM_INSET)
    assert (L[1, 3] + R[1, 3]) / 2 == pytest.approx(-0.05)
    assert L[0, 3] == pytest.approx(0.4 - M.PALM_FORWARD) and L[2, 3] == pytest.approx(0.08)
    np.testing.assert_allclose(L[:3, :3], np.eye(3))  # rpy 0 = palms facing each other


def test_joints_from_input():
    base = np.arange(14, dtype=float)
    q = M.joints_from_input({"kLeftElbow": -1.0, "right_wrist_yaw": 0.5, "RightShoulderPitch": 0.1}, base)
    assert q[3] == -1.0 and q[13] == 0.5 and q[7] == 0.1 and q[0] == 0.0
    with pytest.raises(ValueError):
        M.joints_from_input({"nose": 1.0}, base)
    with pytest.raises(ValueError):
        M.joints_from_input([0.0] * 13, base)
    assert M.joints_from_input(list(range(14)), base).shape == (14,)


def test_encode_jpeg_downscales():
    import cv2

    data = M.encode_jpeg(np.zeros((480, 640, 3), np.uint8), max_width=320)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (240, 320, 3)


# ----------------------------------------------------------------------------- session (fake robot)
def test_move_hands_relative(session):
    out = session.move_hands(left_xyz=[0.05, 0, 0], duration=0.1)
    L, R = session.fk()
    np.testing.assert_allclose(L[:3, 3], [0.25, 0.15, 0.1], atol=1e-9)
    np.testing.assert_allclose(R[:3, 3], [0.2, -0.15, 0.1], atol=1e-9)  # right arm kept its pose
    assert "achieved" in out


def test_move_hands_absolute_clamped(session):
    out = session.move_hands(right_xyz=[0.3, -0.9, 0.0], relative=False, duration=0.1)
    assert "clamped" in out
    np.testing.assert_allclose(session.fk()[1][:3, 3], [0.3, -0.5, 0.0], atol=1e-9)


def test_move_hands_rejects_unreachable(session):
    before = session.fake.q.copy()
    with pytest.raises(M.ToolFailure, match="IK cannot reach"):
        session.move_hands(left_xyz=[0.6, 0.45, 0.4], relative=False, duration=0.1)
    np.testing.assert_allclose(session.fake.q, before)  # did not move


def test_joint_step_limit(session):
    session.max_joint_step = 0.5
    with pytest.raises(M.ToolFailure, match="refused"):
        session.move_joints({"kLeftWristRoll": 1.0}, duration=0.1)
    session.move_joints({"kLeftWristRoll": 0.4}, duration=0.1)
    assert session.fake.q[4] == pytest.approx(0.4)


def test_squeeze_defaults_to_box(session):
    out = session.squeeze(0.30, duration=0.1)
    L, R = session.fk()
    assert L[1, 3] - R[1, 3] == pytest.approx(0.30 + 2 * M.PALM_INSET)
    assert (L[1, 3] + R[1, 3]) / 2 == pytest.approx(-0.05)
    assert "wrapped box" in out
    # now the hands are around the box: the next squeeze keeps their midpoint
    out = session.squeeze(0.20, duration=0.1)
    assert "palm midpoint" in out
    with pytest.raises(M.ToolFailure):
        session.squeeze(0.01)


def test_observe_and_reset(session):
    text, jpegs = session.observe()
    assert [c for c, _ in jpegs] == M.CAMERAS
    assert "kLeftElbow" in text and "wrapped_box" in text and "success: False" in text
    text, jpegs = session.observe(cameras=[])
    assert jpegs == []
    with pytest.raises(M.ToolFailure):
        session.observe(cameras=["nope"])
    session.reset_scene()
    assert session.fake.resets == 1
    np.testing.assert_allclose(session.fake.q, M.REST_POSE)


def test_recorder_without_writer(session, monkeypatch):
    monkeypatch.setitem(sys.modules, "g1lab.lerobot_writer", None)  # import fails
    with pytest.raises(M.ToolFailure, match="LeRobot recording unavailable"):
        session.recorder.start("task")


def test_recorder_with_fake_writer(session, monkeypatch):
    calls = []

    class FakeWriter:
        def __init__(self, root, repo_id, task, fps=30):
            calls.append(("init", repo_id, task, fps))
            self.frames, self.num_episodes = 0, 0

        def start_episode(self, task=None):
            calls.append(("start", task))

        def add_frame(self, state, action, images):
            assert np.shape(state) == (14,) and np.shape(action) == (14,) and set(images) == set(M.CAMERAS)
            self.frames += 1

        def save_episode(self):
            self.num_episodes += 1
            return 0

        def discard_episode(self):
            calls.append(("discard",))

        def close(self):
            calls.append(("close", self.frames))

    monkeypatch.setitem(sys.modules, "g1lab.lerobot_writer", types.SimpleNamespace(LeRobotWriter=FakeWriter))
    session.recorder.start("squeeze the box")
    with pytest.raises(M.ToolFailure):
        session.recorder.start("again")
    session.move_hands(left_xyz=[0, 0, 0.05], duration=0.3)
    r = session.recorder.stop(save=True)
    assert r["saved"] and r["frames"] >= 5 and r["episode_index"] == 0
    assert calls[0] == ("init", "local/sim_claude", "squeeze the box", 30) and calls[-1][0] == "close"
    with pytest.raises(M.ToolFailure):
        session.recorder.stop()


def test_mcp_wiring_in_process(session):
    mcp = pytest.importorskip("mcp")
    if not hasattr(mcp, "Client"):
        pytest.skip("in-process client needs mcp >= 2")
    server = M.build_server(session)

    async def go():
        async with mcp.Client(server) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names == {"observe", "move_hands", "squeeze", "move_joints", "go_home", "reset_scene",
                             "wait", "start_episode", "stop_episode", "get_status"}
            res = await c.call_tool("observe", {"max_width": 160})
            assert [b.type for b in res.content].count("image") == 3
            res = await c.call_tool("move_hands", {"left_xyz": [0.6, 0.45, 0.4], "relative": False})
            assert res.is_error and "IK cannot reach" in res.content[0].text

    asyncio.run(go())


# ----------------------------------------------------------------------------- against the sim (stdio)
def _err_cm(text):
    return [float(x) for x in re.findall(r"error (\d+\.\d) cm", text)]


@needs_sim
def test_stdio_against_sim():
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(command=str(RUN_MCP),
                                   args=["--domain", SIM_DOMAIN, "--image-port", SIM_IMAGE_PORT])

    async def go():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                assert {"observe", "move_hands", "squeeze", "reset_scene", "go_home"} <= names

                res = await s.call_tool("reset_scene", {})
                assert not res.is_error, res.content[0].text

                res = await s.call_tool("observe", {})
                assert not res.is_error, res.content[0].text
                images = [b for b in res.content if b.type == "image"]
                assert len(images) == 3 and all(b.mime_type == "image/jpeg" for b in images)

                res = await s.call_tool("move_hands", {"left_xyz": [0.05, 0, 0], "right_xyz": [0.05, 0, 0]})
                text = res.content[0].text
                assert not res.is_error, text
                assert all(e <= 1.0 for e in _err_cm(text)), text

                res = await s.call_tool("squeeze", {"width_m": 0.30})
                assert not res.is_error, res.content[0].text

                res = await s.call_tool("go_home", {})
                assert not res.is_error, res.content[0].text
                res = await s.call_tool("reset_scene", {})  # also puts the arms back at the rest pose
                assert not res.is_error, res.content[0].text

    asyncio.run(go())
