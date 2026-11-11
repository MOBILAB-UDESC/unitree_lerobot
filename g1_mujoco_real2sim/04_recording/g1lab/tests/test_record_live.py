"""Pure-logic tests for g1lab.record_live (no DDS, no sim, no lerobot).

    cd 04_recording && python -m pytest g1lab/tests/test_record_live.py -v -p no:cacheprovider
"""

import cv2
import numpy as np
import pytest

from g1lab import record_live as rl


class FakeWriter:
    def __init__(self):
        self.log, self.episodes, self.frames = [], [], []
        self.closed = False

    def start_episode(self, task=None):
        self.log.append(("start", task))
        self.frames = []

    def add_frame(self, state, action, images):
        if set(images) != set(rl.CAMERAS):
            raise ValueError("bad cams")
        self.frames.append((np.asarray(state), np.asarray(action)))

    def save_episode(self):
        self.episodes.append(self.frames)
        self.frames = []
        self.log.append(("save",))
        return len(self.episodes) - 1

    def discard_episode(self):
        self.frames = []
        self.log.append(("discard",))

    def close(self):
        self.closed = True


def no_decode(jpgs, binocular=()):
    return {c: None for c in jpgs}


def q29(arms=0.0, waist=0.0):
    q = np.zeros(29)
    q[15:29] = arms
    q[12] = waist
    return q


JPGS = {c: b"x" for c in rl.CAMERAS}


def make_session(**kw):
    w = FakeWriter()
    sink = rl.EpisodeSink(w, decode=no_decode, threaded=False, log=lambda *a: None)
    return w, sink, rl.Session(sink, "task", log=lambda *a: None, min_frames=kw.pop("min_frames", 2), **kw)


# ----------------------------------------------------------------------------- command sources
def test_active_source_prefers_lowcmd_and_times_out():
    assert rl.active_source(None, None, 10.0) is None
    assert rl.active_source(9.9, 9.9, 10.0) == "lowcmd"
    assert rl.active_source(9.0, 9.9, 10.0) == "arm_sdk"   # lowcmd stale
    assert rl.active_source(9.0, 9.0, 10.0) is None


def test_command_for_sources_and_waist_modes():
    meas, low, sdk = q29(0.1, 0.3), q29(0.2, 0.4), q29(0.5, 0.0)
    arms, w = rl.command_for("lowcmd", meas, low, sdk)
    assert np.allclose(arms, 0.2) and w == pytest.approx(0.4)
    arms, w = rl.command_for("arm_sdk", meas, low, sdk)          # motion mode: waist from the next tick
    assert np.allclose(arms, 0.5) and w is None
    arms, w = rl.command_for(None, meas, None, None)             # nobody commands: hold = measured
    assert np.allclose(arms, 0.1) and w == pytest.approx(0.3)
    assert rl.command_for("arm_sdk", meas, low, sdk, "command")[1] == pytest.approx(0.0)
    assert rl.command_for("lowcmd", meas, low, sdk, "measured")[1] == pytest.approx(0.3)
    assert rl.command_for("lowcmd", meas, low, sdk, "next")[1] is None
    with pytest.raises(ValueError):
        rl.command_for("lowcmd", meas, low, sdk, "bogus")


def test_state_from():
    s = rl.state_from(q29(0.1, 0.3))
    assert s.shape == (15,) and s.dtype == np.float32 and s[14] == pytest.approx(0.3)
    assert rl.state_from(q29(0.1, 0.3), waist=False).shape == (14,)


def test_lookahead_uses_next_measured_waist():
    a = rl.LookaheadAssembler(waist=True)
    st = lambda w: np.append(np.zeros(14), w).astype(np.float32)  # noqa: E731
    assert a.push(st(0.0), np.ones(14), None, "p0") is None
    s, act, p = a.push(st(0.1), np.ones(14), 0.7, "p1")
    assert p == "p0" and act.shape == (15,) and act[14] == pytest.approx(0.1)   # next tick's measured
    s, act, p = a.push(st(0.2), np.ones(14), None, "p2")
    assert p == "p1" and act[14] == pytest.approx(0.7)                          # explicit command kept
    s, act, p = a.flush()
    assert p == "p2" and act[14] == pytest.approx(0.2)                          # last: own measured
    assert a.flush() is None


def test_lookahead_without_waist():
    a = rl.LookaheadAssembler(waist=False)
    a.push(np.zeros(14, np.float32), np.ones(14), None, 0)
    s, act, _ = a.flush()
    assert act.shape == (14,)


def test_decode_jpgs_crops_stereo_and_resizes():
    left = np.zeros((480, 640, 3), np.uint8)
    left[..., 2] = 255                                   # red in BGR
    stereo = np.concatenate([left, np.zeros_like(left)], axis=1)
    small = cv2.resize(left, (320, 240))
    jpgs = {"cam_head": cv2.imencode(".jpg", stereo)[1].tobytes(),
            "cam_left_wrist": cv2.imencode(".jpg", small)[1].tobytes()}
    out = rl.decode_jpgs(jpgs, binocular={"cam_head"})
    for img in out.values():
        assert img.shape == (480, 640, 3) and img.dtype == np.uint8
        assert img[..., 0].mean() > 200 and img[..., 2].mean() < 50   # RGB, left eye kept


# ------------------------------------------------------------------------------------- session
def run_ticks(sess, n, t0=0.0, success_at=None, waist=0.0):
    t = t0
    for i in range(n):
        st = np.append(np.zeros(14), waist).astype(np.float32)
        sess.tick(t, st, np.zeros(14), None, JPGS, success=(success_at is not None and i >= success_at))
        t += 1 / 30
    return t


def test_keys_start_stop_save_discard_quit():
    w, sink, s = make_session()
    s.key("s")
    run_ticks(s, 10)
    s.key("s")
    assert len(w.episodes) == 1 and len(w.episodes[0]) == 10      # lookahead flushed on stop
    s.key("s")
    run_ticks(s, 5)
    s.key("d")
    assert w.log[-1] == ("discard",) and len(w.episodes) == 1
    s.key("s")
    run_ticks(s, 5)
    s.key("q")
    assert s.quit and w.log[-1] == ("discard",) and len(w.episodes) == 1
    sink.close()
    assert w.closed


def test_auto_success_needs_rising_edge_and_hold():
    w, sink, s = make_session(auto_success=True, success_hold=0.1)
    s.key("s")
    run_ticks(s, 20, success_at=10)
    assert not s.recording and len(w.episodes) == 1
    assert 12 <= len(w.episodes[0]) <= 16          # stopped once success held ~0.1 s
    # a success flag that is already up when the episode starts is stale: no auto stop
    s.key("s")
    run_ticks(s, 10, success_at=0)
    assert s.recording
    s.key("d")


def test_flickering_success_does_not_count():
    w, sink, s = make_session(auto_success=True, only_success=True, success_hold=0.2)
    s.key("s")
    t = 0.0
    for i in range(30):                             # success up for 2 ticks out of every 5
        s.tick(t, np.zeros(15, np.float32), np.zeros(14), None, JPGS, success=(i % 5) in (3, 4))
        t += 1 / 30
    assert s.recording and not s.success_seen
    assert s.stop("x") == "discarded"


def test_only_success_discards_failures_and_short_episodes():
    w, sink, s = make_session(only_success=True, min_frames=5, success_hold=0.1)
    s.key("s")
    run_ticks(s, 10)
    assert s.stop("x") == "discarded"
    s.key("s")
    run_ticks(s, 2, success_at=1)
    assert s.stop("x") == "discarded"              # too short (and success not held long enough)
    s.key("s")
    run_ticks(s, 10, success_at=5)
    assert s.stop("x") == "saved" and len(w.episodes) == 1


def test_auto_start_on_reset_with_delay():
    w, sink, s = make_session(auto_start=True, start_delay=0.5, auto_success=True, success_hold=0.0,
                              only_success=True)
    s.on_reset(0.0)
    t = run_ticks(s, 10)                            # 0.33 s: not yet
    assert not s.recording
    t = run_ticks(s, 30, t0=t, success_at=25)       # starts at 0.5 s, success later -> saved
    assert len(w.episodes) == 1 and not s.recording
    s.on_reset(t)                                   # next attempt: fails, then a reset ends it
    t = run_ticks(s, 30, t0=t + 0.5)
    assert s.recording
    s.on_reset(t)
    assert w.log[-1] == ("discard",) and len(w.episodes) == 1


def test_sink_counts_bad_frames_and_threaded_close():
    w = FakeWriter()
    sink = rl.EpisodeSink(w, decode=lambda j, b: {"cam_head": None}, threaded=True, log=lambda *a: None)
    sink.start("t")
    sink.frame(np.zeros(15), np.zeros(15), JPGS)
    sink.save()
    sink.close()
    assert sink.frame_errors == 1 and w.closed and sink.saved == [0]


def test_episode_time_limit():
    w, sink, s = make_session(episode_seconds=0.5)
    s.key("s")
    run_ticks(s, 30)
    assert not s.recording and len(w.episodes) == 1 and 15 <= len(w.episodes[0]) <= 17


def test_success_already_up_at_launch_is_stale():
    w, sink, s = make_session(only_success=True, success_hold=0.0)
    s.key("s")
    run_ticks(s, 20, success_at=0)               # up from the very first tick the recorder saw
    assert s.stop("x") == "discarded"
