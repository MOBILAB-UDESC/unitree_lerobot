"""Tests for g1lab.remote_vla against the local mocks in g1lab.vla_mocks.

    # unit + protocol tests (no sim needed, ~15 s)
    env -u PYTHONPATH -u LD_LIBRARY_PATH -u CYCLONEDDS_URI ~/miniforge3/envs/g1sim/bin/python -m g1lab.tests.test_remote_vla

    # + closed-loop tests on a running sim, e.g. one started with --domain 3 --port-offset 1000
    G1LAB_SIM_DOMAIN=3 G1LAB_SIM_IMAGE_PORT=61000 ... python -m g1lab.tests.test_remote_vla

Run as a module (not via pytest) for the sim tests: DDS threads keep the process alive, so the
__main__ block ends with os._exit().
"""

import base64
import json
import os
import socket
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from g1lab import remote_vla as rv  # noqa: E402
from g1lab import vla_mocks as vm  # noqa: E402

MAX_STEP = 0.05
SIM_DOMAIN = os.environ.get("G1LAB_SIM_DOMAIN")
SIM_IMAGE_PORT = int(os.environ.get("G1LAB_SIM_IMAGE_PORT", "60000"))
SIM_STEPS = int(os.environ.get("G1LAB_SIM_STEPS", "150"))


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def make_backend(kind, port, **kw):
    if kind == "dream":
        return rv.DreamBackend(f"http://127.0.0.1:{port}", api_key=kw.pop("api_key", "test-key"), **kw)
    if kind == "openpi":
        return rv.OpenPIBackend(f"ws://127.0.0.1:{port}", **kw)
    return rv.Gr00tBackend(f"tcp://127.0.0.1:{port}", **kw)


def start_mock(kind, **kw):
    port = free_port()
    srv = vm.start_in_thread(kind, port, quiet=True, **kw)
    time.sleep(0.2)
    return srv, port


def rgb_frames(n=1, seed=0):
    rng = np.random.default_rng(seed)
    return [{c: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in rv.CAMERAS} for _ in range(n)]


def ideal_sine(wall, gen_t0, anchor):
    """The trajectory the sine mock commands, as a function of wall-clock time."""
    t = np.asarray(wall)[:, None] - gen_t0
    return np.asarray(anchor) + vm.SINE_AMP * (1 - np.cos(2 * np.pi * vm.SINE_HZ * t)) / 2


class FakeRobot:
    """Stands in for G1Arms: first-order joint tracking, constant camera images."""

    sim = True

    def __init__(self, q0=None):
        self.q = np.zeros(14) if q0 is None else np.asarray(q0, dtype=np.float64)
        self.cmd = self.q.copy()
        self.imgs = rgb_frames(1)[0]
        self.commands = []

    def observe(self):
        self.q += 0.5 * (self.cmd - self.q)
        return {"state": self.q.astype(np.float32), "images": self.imgs}

    def state(self):
        return self.q.astype(np.float32)

    def last_command(self):
        return self.cmd.astype(np.float32)

    def command(self, q):
        self.cmd = np.asarray(q, dtype=np.float64).copy()
        self.commands.append(self.cmd)

    def success(self):
        return False


# ------------------------------------------------------------------------------------- pure logic
class TestCore(unittest.TestCase):
    def test_joint_limits_from_urdf(self):
        lo, hi = rv.load_joint_limits()
        self.assertEqual(lo.shape, (14,))
        np.testing.assert_allclose(lo, rv._FALLBACK_LIMITS[:, 0])
        np.testing.assert_allclose(hi, rv._FALLBACK_LIMITS[:, 1])
        self.assertAlmostEqual(lo[8], -2.2515)  # right shoulder roll is mirrored
        self.assertAlmostEqual(hi[1], 2.2515)

    def test_safety_clip_and_rate(self):
        lo, hi = rv.load_joint_limits()
        f = rv.SafetyFilter(lo, hi, max_step=MAX_STEP)
        prev = np.zeros(14)
        out = f(np.full(14, 10.0), prev)
        np.testing.assert_allclose(out, np.full(14, MAX_STEP))
        q = prev
        for _ in range(200):
            q = f(np.full(14, 10.0), q)
        np.testing.assert_allclose(q, hi)  # never past the URDF limit
        np.testing.assert_allclose(f(np.full(14, np.nan), q), q)  # non-finite -> hold
        self.assertEqual(f.rejected, 1)

    def test_resample_20_to_30(self):
        H = 48
        t = np.arange(H) / 20.0
        chunk = np.stack([np.sin(t + j) for j in range(14)], 1)
        out = rv.resample_chunk(chunk, 20.0, 30.0)
        self.assertEqual(out.shape, (71, 14))  # (48-1)/20 s * 30 Hz + 1
        t30 = np.arange(71) / 30.0
        np.testing.assert_allclose(out[:, 0], np.sin(t30), atol=2e-3)
        np.testing.assert_allclose(rv.resample_chunk(chunk[:41], 20.0, 30.0)[-1], chunk[40], atol=1e-9)  # 2 s: exact end
        self.assertIs(type(rv.resample_chunk(chunk, 30.0, 30.0)), np.ndarray)

    def test_player_latency_skip_and_seam_blend(self):
        p = rv.ChunkPlayer(blend_steps=5, execute_ratio=30 / 48)
        a = np.zeros((71, 14))
        p.add_chunk(a, 0, current=np.zeros(14))
        for _ in range(45):
            p.next()
        self.assertTrue(p.needs_request())
        played_at_request = p.played
        for _ in range(3):  # 3 ticks of latency
            p.next()
        b = np.ones((71, 14))
        p.add_chunk(b, p.played - played_at_request)
        self.assertEqual(p.remaining(), 68)
        seq = np.array([p.next()[0][0] for _ in range(8)])
        np.testing.assert_allclose(seq[:5], [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6])  # linear seam over 5 ticks
        np.testing.assert_allclose(seq[5:], 1.0)

    def test_player_starvation_holds(self):
        p = rv.ChunkPlayer()
        p.add_chunk(np.ones((3, 14)), 0)
        for _ in range(3):
            p.next()
        a, starved = p.next()
        self.assertTrue(starved)
        np.testing.assert_allclose(a, 1.0)

    def test_msgpack_roundtrip(self):
        obs = {"a": np.arange(6, dtype=np.float32).reshape(2, 3), "img": np.zeros((4, 5, 3), np.uint8), "s": "x",
               "l": ["p"], "n": np.float64(1.5)}
        for style in ("openpi", "gr00t"):
            d = rv.unpackb(rv.packb(obs, style))
            np.testing.assert_array_equal(d["a"], obs["a"])
            self.assertEqual(d["a"].dtype, np.float32)
            self.assertEqual(d["img"].shape, (4, 5, 3))
            self.assertEqual((d["s"], d["l"], float(d["n"])), ("x", ["p"], 1.5))

    def test_frame_history_5fps(self):
        h = rv.FrameHistory(9, 5.0, 30.0)
        for step in range(120):
            h.push(step, {"cam_head": step})
        self.assertEqual([f["cam_head"] for f in h.frames()], list(range(66, 120, 6)))


# ---------------------------------------------------------------------------------- protocol tests
class TestDream(unittest.TestCase):
    def setUp(self):
        self.srv, self.port = start_mock("dream")

    def tearDown(self):
        self.srv.stop()

    def test_schema_session_and_reset(self):
        b = make_backend("dream", self.port)
        frames = [b.frame_transform(f) for f in rgb_frames(9)]
        st = np.linspace(-0.3, 0.3, 14).astype(np.float32)
        chunk, hz = b.infer(st, frames, rv.DEFAULT_TASK)
        self.assertEqual((chunk.shape, hz), ((48, 14), 20.0))
        req = b.last_request
        self.assertEqual(set(req), {"checkpoint", "session_id", "reset", "observation"})
        self.assertIsNone(req["session_id"])
        obs = req["observation"]
        self.assertEqual(set(obs), {"state.left_arm", "state.right_arm", "state.left_gripper", "state.right_gripper",
                                    "video.overhead", "video.wrist_l", "video.wrist_r", "prompt"})
        np.testing.assert_allclose(obs["state.left_arm"] + obs["state.right_arm"], st, atol=1e-6)
        for key in ("video.overhead", "video.wrist_l", "video.wrist_r"):
            self.assertEqual(len(obs[key]), 9)
            import cv2
            img = cv2.imdecode(np.frombuffer(base64.b64decode(obs[key][0]), np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(img.shape, (176, 320, 3))
        json.dumps(req)  # pure JSON
        sid = b.session_id
        self.assertTrue(sid)
        b.infer(st, frames, rv.DEFAULT_TASK)
        self.assertEqual(b.last_request["session_id"], sid)  # reused
        b.reset()
        self.assertIsNone(b.session_id)
        b.infer(st, frames, rv.DEFAULT_TASK)
        self.assertIsNone(b.last_request["session_id"])
        reqs = self.srv.mock.requests
        self.assertTrue(all(r["ok"] for r in reqs), [r["errors"] for r in reqs])
        self.assertEqual([r["endpoint"] for r in reqs], ["get_action", "get_action", "end_session", "get_action"])

    def test_bad_request_is_reported(self):
        b = make_backend("dream", self.port, api_key="")
        with self.assertRaises(rv.PolicyError) as cm:
            b.infer(np.zeros(14), [b.frame_transform(rgb_frames(1)[0])], "x")
        self.assertTrue(cm.exception.fatal)
        self.assertIn("Bearer", str(cm.exception))

    def test_field_override(self):
        b = make_backend("dream", self.port, keys={"cameras": {"cam_head": "video.top"}, "state_left_gripper": None})
        req = b.build_request(np.zeros(14), [b.frame_transform(rgb_frames(1)[0])], "x")["observation"]
        self.assertIn("video.top", req)
        self.assertNotIn("video.overhead", req)
        self.assertNotIn("state.left_gripper", req)
        self.assertIn("video.wrist_l", req)


class TestOpenPI(unittest.TestCase):
    def setUp(self):
        self.srv, self.port = start_mock("openpi")

    def tearDown(self):
        self.srv.stop()

    def test_schema(self):
        b = make_backend("openpi", self.port)
        b.connect()
        self.assertEqual(b.metadata["action_dim"], 14)
        chunk, hz = b.infer(np.zeros(14, np.float32), rgb_frames(1), rv.DEFAULT_TASK)
        self.assertEqual((chunk.shape, hz), ((50, 14), 30.0))
        req = b.last_request
        self.assertEqual(set(req), {"observation/exterior_image_1_left", "observation/wrist_image_left",
                                    "observation/wrist_image_right", "observation/joint_position", "prompt"})
        self.assertEqual(req["observation/exterior_image_1_left"].shape, (224, 224, 3))
        self.assertTrue(all(r["ok"] for r in self.srv.mock.requests))
        b.close()

    def test_server_error_string(self):
        b = make_backend("openpi", self.port, keys={"state": "observation/wrong"})
        with self.assertRaises(rv.PolicyError) as cm:
            b.infer(np.zeros(14), rgb_frames(1), "x")
        self.assertIn("observation/joint_position", str(cm.exception))
        b.close()


class TestGr00t(unittest.TestCase):
    def setUp(self):
        self.srv, self.port = start_mock("gr00t", api_token="secret")

    def tearDown(self):
        self.srv.stop()

    def test_schema_and_token(self):
        b = make_backend("gr00t", self.port, api_token="secret")
        b.connect()
        b.reset()
        chunk, hz = b.infer(np.zeros(14), rgb_frames(1), rv.DEFAULT_TASK)
        self.assertEqual((chunk.shape, hz), ((16, 14), 30.0))
        req = b.last_request
        self.assertEqual(req["video.cam_head"].shape, (1, 480, 640, 3))
        self.assertEqual(req["state.left_arm"].shape, (1, 7))
        self.assertEqual(req["annotation.human.task_description"], [rv.DEFAULT_TASK])
        self.assertTrue(all(r["ok"] for r in self.srv.mock.requests))
        b.close()
        bad = make_backend("gr00t", self.port, api_token="wrong", timeout=3)
        with self.assertRaises(rv.PolicyError) as cm:
            bad.connect()
        self.assertTrue(cm.exception.fatal)
        bad.close()


# ------------------------------------------------------------------------ closed loop, fake robot
def check_run(tc, runner, stats, srv, steps, min_chunks):
    tr = runner.trace
    cmd = np.array([r["cmd"] for r in tr])
    tc.assertEqual(stats["steps"], steps)
    tc.assertEqual(stats["failures"], 0)
    tc.assertEqual(stats["starved"], 0)
    tc.assertGreaterEqual(stats["chunks"], min_chunks)
    tc.assertLessEqual(np.abs(np.diff(cmd, axis=0)).max(), MAX_STEP + 1e-9)
    tc.assertTrue(all(r["ok"] for r in srv.mock.requests), [r["errors"] for r in srv.mock.requests if not r["ok"]])
    # The commanded trajectory follows the mock's sine (resampling + latency alignment + seams).
    first = next(r for r in srv.mock.requests if "gen_t0" in r)
    wall = np.array([r["wall"] for r in tr])
    err = np.abs(cmd - ideal_sine(wall, first["gen_t0"], first["anchor"]))
    tc.assertLess(np.sqrt((err[10:] ** 2).mean()), 0.01)
    tc.assertLess(err[10:].max(), 0.03)
    # Smooth: no jerk spikes at the seams.
    tc.assertLess(np.abs(np.diff(cmd[5:], 2, axis=0)).max(), 0.005)
    return cmd, err


class TestClosedLoopFake(unittest.TestCase):
    STEPS = 90

    def _run(self, kind, min_chunks):
        srv, port = start_mock(kind)
        try:
            b = make_backend(kind, port)
            b.connect()
            runner = rv.Runner(FakeRobot(), b, max_step=MAX_STEP, verbose=False)
            stats = runner.run_episode(self.STEPS)
            check_run(self, runner, stats, srv, self.STEPS, min_chunks)
            self.assertGreater(stats["hz"], 28.0)
            runner.close()
        finally:
            srv.stop()

    def test_dream(self):
        self._run("dream", 2)

    def test_openpi(self):
        self._run("openpi", 2)

    def test_gr00t(self):
        self._run("gr00t", 6)

    def test_stops_on_success(self):
        srv, port = start_mock("gr00t")
        try:
            robot = FakeRobot()
            robot.success = lambda: len(robot.commands) > 40  # (only polled after min_success_steps)
            b = make_backend("gr00t", port)
            runner = rv.Runner(robot, b, max_step=MAX_STEP, verbose=False)
            stats = runner.run_episode(300)
            self.assertEqual((stats["reason"], stats["steps"]), ("success", 41))
            runner.close()
        finally:
            srv.stop()


# ------------------------------------------------------------------------------ closed loop, sim
@unittest.skipUnless(SIM_DOMAIN, "set G1LAB_SIM_DOMAIN (and G1LAB_SIM_IMAGE_PORT) to run against a live sim")
class TestClosedLoopSim(unittest.TestCase):
    robot = None
    results = {}

    @classmethod
    def setUpClass(cls):
        from g1lab.robot_io import G1Arms
        cls.robot = G1Arms("sim", domain=int(SIM_DOMAIN), image_port=SIM_IMAGE_PORT)
        t_end = time.time() + 10
        while time.time() < t_end and len(cls.robot.images()) < 3:
            time.sleep(0.1)

    @classmethod
    def tearDownClass(cls):
        if cls.robot is not None:
            cls.robot.go_home()
        for k, v in cls.results.items():
            print(f"\n[sim {k}] " + ", ".join(f"{n}={x}" for n, x in v.items()), flush=True)

    def _run(self, kind, min_chunks):
        robot = self.robot
        self.assertEqual(sorted(robot.images()), sorted(rv.CAMERAS))
        robot.move_to(np.zeros(14), duration=1.0)
        time.sleep(0.5)
        srv, port = start_mock(kind)
        try:
            b = make_backend(kind, port)
            b.connect()
            runner = rv.Runner(robot, b, max_step=MAX_STEP, verbose=False)
            stats = runner.run_episode(SIM_STEPS)
            cmd, err = check_run(self, runner, stats, srv, SIM_STEPS, min_chunks)
            state = np.array([r["state"] for r in runner.trace])
            # the sim joints really move as commanded: shoulder pitch / elbow swing, tracked within a few ticks
            lag = 4
            track = np.abs(state[lag:] - cmd[:-lag])[10:]
            moved = state.max(0) - state.min(0)
            self.assertGreater(moved[0], 0.15)
            self.assertGreater(moved[3], 0.15)
            self.assertLess(np.sqrt((track ** 2).mean()), 0.03)
            self.assertGreater(stats["hz"], 27.0)
            self.results[kind] = dict(
                steps=stats["steps"], hz=round(stats["hz"], 1), chunks=stats["chunks"],
                latency_ms=round(stats["latency_ms_mean"], 1), starved=stats["starved"],
                max_step=round(float(np.abs(np.diff(cmd, axis=0)).max()), 4),
                max_jerk=round(float(np.abs(np.diff(cmd[5:], 2, axis=0)).max()), 5),
                cmd_vs_mock_rms=round(float(np.sqrt((err[10:] ** 2).mean())), 4),
                track_rms=round(float(np.sqrt((track ** 2).mean())), 4),
                moved_j0=round(float(moved[0]), 3), requests=len(srv.mock.requests))
            runner.close()
        finally:
            srv.stop()

    def test_dream(self):
        self._run("dream", 3)

    def test_openpi(self):
        self._run("openpi", 3)

    def test_gr00t(self):
        self._run("gr00t", 10)


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    sys.stdout.flush()
    os._exit(0 if result.wasSuccessful() else 1)
