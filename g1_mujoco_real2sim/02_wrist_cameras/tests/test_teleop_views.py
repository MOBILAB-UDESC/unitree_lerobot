"""Display logic of teleop_views.py against fake frames and a fake vuer session (no server, no headset).

    source scripts/env.sh g1sim; python -m pytest 02_wrist_cameras/tests -q -p no:cacheprovider
"""

import asyncio
import math
import os
import sys
from multiprocessing import Array

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import teleop_views as tv  # noqa: E402

CFG_WRISTS = {
    "head_camera": {"enable_zmq": True, "zmq_port": 55555, "enable_webrtc": False, "image_shape": [480, 1280],
                    "binocular": True},
    "left_wrist_camera": {"enable_zmq": True, "zmq_port": 55556, "image_shape": [480, 640], "binocular": False},
    "right_wrist_camera": {"enable_zmq": True, "zmq_port": 55557, "image_shape": [480, 640], "binocular": False},
}
CFG_NO_WRISTS = {
    "head_camera": CFG_WRISTS["head_camera"],
    "left_wrist_camera": {"enable_zmq": False, "image_shape": [480, 640]},
    "right_wrist_camera": {"enable_zmq": False, "image_shape": [480, 640]},
}


class FakeSession:
    def __init__(self):
        self.calls = []  # (to, [serialized nodes])

    def upsert(self, nodes, to=None):
        nodes = nodes if isinstance(nodes, (list, tuple)) else [nodes]
        self.calls.append((to, [n.serialize() for n in nodes]))


class StopLoop(Exception):
    pass


def run_frames(coro_fn, session, frames=1, monkeypatch=None):
    """Run a display coroutine for `frames` iterations (asyncio.sleep raises after that)."""
    count = {"n": 0}

    async def fake_sleep(_):
        count["n"] += 1
        if count["n"] >= frames:
            raise StopLoop

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    with pytest.raises(StopLoop):
        asyncio.run(coro_fn(session))


def head_pose(x=0.0, y=1.6, z=0.0, yaw_deg=0.0):
    a = math.radians(yaw_deg)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    m[:3, 3] = [x, y, z]
    return m


@pytest.fixture
def views():
    streams = tv.wrist_streams(CFG_WRISTS, "immersive", True, False)
    v = tv.WristViews(streams, "pip", "immersive", 640 / 480)
    v.debounce_s = 0.0
    for side in v.sides:
        v.write_frame(side, np.full((480, 640, 3), 100 if side == "left" else 200, np.uint8))
    yield v
    v.close()


def visible(nodes):
    return [n for n in nodes if not n.get("hide")]


def make_tvuer(views_obj, binocular=True, display_mode="immersive"):
    """WristViewsTeleVuer without its __init__ (which would start the vuer server)."""
    t = object.__new__(tv.WristViewsTeleVuer)
    t._views = views_obj
    t.use_hand_tracking = False
    t.display_fps = 30.0
    t.display_mode = display_mode
    t.binocular = binocular
    t.img_height = 480
    t.img_width = 640
    t.aspect_ratio = 640 / 480
    t.img2display = np.zeros((480, 1280 if binocular else 640, 3), np.uint8)
    t.head_pose_shared = Array("d", [float(v) for v in head_pose().flatten(order="F")])
    return t


# ---------------------------------------------------------------- stream selection / geometry
def test_wrist_streams_selection():
    assert set(tv.wrist_streams(CFG_WRISTS, "immersive", True, False)) == {"left", "right"}
    assert tv.wrist_streams(CFG_WRISTS, "immersive", True, False)["left"] == (480, 640, False)
    assert tv.wrist_streams(CFG_NO_WRISTS, "immersive", True, False) == {}
    assert tv.wrist_streams(CFG_WRISTS, "pass-through", True, False) == {}
    assert tv.wrist_streams(CFG_WRISTS, "immersive", False, True) == {}  # WebRTC head: untouched
    assert tv.wrist_streams(None, "immersive", True, False) == {}
    stereo = dict(CFG_WRISTS, left_wrist_camera={"enable_zmq": True, "image_shape": [480, 1280], "binocular": True})
    assert tv.wrist_streams(stereo, "ego", True, False)["left"] == (480, 640, True)


@pytest.mark.parametrize("mode", ["immersive", "ego"])
def test_side_fits_span_and_matches_head(mode):
    plan = tv.plan_layout("side", mode, 4 / 3, {"left": 4 / 3, "right": 4 / 3})
    head = plan["head"]
    for side, sign in (("left", -1), ("right", 1)):
        w = plan[side]
        assert w["height"] == pytest.approx(head["height"])  # same size as the head view
        yaw = -math.degrees(w["rotation"][1])
        assert sign * yaw > 0
        half_w = math.degrees(math.atan(w["height"] * w["aspect"] / 2 / head["distanceToCamera"]))
        assert abs(yaw) + half_w <= tv.SIDE_SPAN_DEG / 2 + 1e-6  # inside the FOV budget
    assert plan["left"]["position"][0] == pytest.approx(-plan["right"]["position"][0])
    if mode == "immersive":
        assert head["height"] < 1.0  # shrunk to fit
    else:
        assert head["height"] == pytest.approx(0.75)  # ego already fits


def test_pip_small_low_and_off_has_no_wrists():
    plan = tv.plan_layout("pip", "immersive", 4 / 3, {"left": 4 / 3, "right": 4 / 3})
    assert plan["head"] == {"height": 1.0, "distanceToCamera": 1.0}  # head unchanged
    for side in ("left", "right"):
        assert plan[side]["height"] < 0.3
        assert plan[side]["position"][1] < 0  # low
        assert math.hypot(plan[side]["position"][0], plan[side]["distanceToCamera"]) < 1.0  # in front of head
    assert set(tv.plan_layout("off", "immersive", 4 / 3, {"left": 4 / 3})) == {"head"}


def test_place_faces_eye():
    p = tv.place(30.0, -20.0, 0.2, 4 / 3, 0.9)
    x, y = p["position"][:2]
    z = -p["distanceToCamera"]
    assert math.sqrt(x * x + y * y + z * z) == pytest.approx(0.9)
    pitch, yaw, roll, order = p["rotation"]
    # plane normal R_y(yaw) R_x(pitch) [0,0,1] must point back at the eye
    n = np.array([math.cos(pitch) * math.sin(yaw), -math.sin(pitch), math.cos(pitch) * math.cos(yaw)])
    assert np.allclose(n, -np.array([x, y, z]) / 0.9)
    assert order == "YXZ" and roll == 0.0


# ---------------------------------------------------------------- nodes per layout
@pytest.mark.parametrize("layout", ["pip", "side"])
@pytest.mark.parametrize("binocular", [True, False])
def test_nodes_head_plus_two_wrists(views, layout, binocular):
    views.set_layout(layout)
    img = np.zeros((480, 1280 if binocular else 640, 3), np.uint8)
    head = [n.serialize() for n in views.head_nodes(img, binocular, 640, 4 / 3)]
    wrists = [n.serialize() for n in views.wrist_nodes(views.new_session())]
    assert [n["key"] for n in head] == (["background-left", "background-right"] if binocular else ["background-mono"])
    if binocular:
        assert [n["layers"] for n in head] == [1, 2]
    assert [n["key"] for n in wrists] == ["wrist-left", "wrist-right"]
    for n in wrists:
        assert n["tag"] == "ImageBackground"
        assert "layers" not in n  # both eyes see them
        assert isinstance(n["src"], bytes) and n["src"][:2] == b"\xff\xd8"
        assert n["hide"] is False
    assert head[0]["height"] == pytest.approx(views.plans[layout]["head"]["height"])


def test_off_only_head(views):
    views.set_layout("off")
    st = views.new_session()
    for _ in range(tv.HIDE_FRAMES):
        assert visible([n.serialize() for n in views.wrist_nodes(st)]) == []  # hide markers only
    assert views.wrist_nodes(st) == []  # then nothing at all: plain televuer frames
    head = views.head_nodes(np.zeros((480, 1280, 3), np.uint8), True, 640, 4 / 3)
    assert [n.serialize()["height"] for n in head] == [1.0, 1.0]


def test_toggle_cycle_and_inputs(views):
    assert views.layout == "pip"
    seen = []
    for _ in range(4):
        views.toggle()
        seen.append(views.layout)
    assert seen == ["side", "off", "pip", "side"]
    # left X: rising edge only
    views.set_layout("pip")
    assert views.on_left_x(True) and views.layout == "side"
    assert not views.on_left_x(True) and views.layout == "side"  # held
    assert not views.on_left_x(False)
    assert views.on_left_x(True) and views.layout == "off"
    # clicks
    class Ev:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    assert views.on_click(Ev(etype="ON_CLICK", value={"key": tv.KEY_BUTTON})) and views.layout == "pip"
    assert views.on_click(Ev(etype="CLICK", key=tv.KEY_HTML_BUTTON, value=None)) and views.layout == "side"
    assert not views.on_click(Ev(etype="ON_CLICK", value={"key": "something-else"}))
    views.debounce_s = 10.0
    assert not views.toggle()  # debounced


def test_button_anchor_follows_lazily(views):
    st = views.new_session()
    aux = dict((to, n) for n, to in views.aux_upserts(st, np.zeros((4, 4))))
    assert "children" not in aux and "htmlChildren" in aux  # no head pose yet: page button only
    aux = dict((to, n.serialize()) for n, to in views.aux_upserts(st, head_pose()))
    b = aux["children"]
    assert b["tag"] == "Plane" and b["key"] == tv.KEY_BUTTON and b["materialType"] == "basic"
    assert b["material"]["map"].startswith("data:image/png;base64,")
    x, y, z = b["position"]
    assert x < 0 and y > 1.6 and z < 0  # up-left in front of the operator
    assert views.aux_upserts(st, head_pose(yaw_deg=10)) == []  # small turn: stays put, nothing resent
    aux = dict((to, n.serialize()) for n, to in views.aux_upserts(st, head_pose(yaw_deg=90)))
    x2, _, z2 = aux["children"]["position"]
    assert x2 < 0 and abs(z2) < abs(x2)  # re-anchored to the left-facing gaze (forward = -x)
    views.toggle()
    aux = dict((to, n.serialize()) for n, to in views.aux_upserts(st, head_pose(yaw_deg=90)))
    assert "SIDE" in aux["htmlChildren"]["value"]  # label follows the layout at once


# ---------------------------------------------------------------- display coroutines
@pytest.mark.parametrize("binocular", [True, False])
def test_display_loop_single_upsert(views, monkeypatch, binocular):
    t = make_tvuer(views, binocular)
    s = FakeSession()
    coro = t.main_image_binocular_zmq if binocular else t.main_image_monocular_zmq
    run_frames(coro, s, frames=2, monkeypatch=monkeypatch)
    assert s.calls[0][1][0]["tag"] == "MotionControllers"
    frames = [nodes for to, nodes in s.calls[1:] if to == "bgChildren"]
    assert len(frames) == 2
    heads = ["background-left", "background-right"] if binocular else ["background-mono"]
    for nodes in frames:  # head + wrists in ONE message
        assert [n["key"] for n in nodes] == heads + ["wrist-left", "wrist-right"]
    assert {to for to, _ in s.calls[1:]} == {"bgChildren", "children", "htmlChildren"}


def test_display_loop_ego(monkeypatch):
    views = tv.WristViews(tv.wrist_streams(CFG_WRISTS, "ego", True, False), "side", "ego", 640 / 480)
    try:
        t = make_tvuer(views, binocular=False, display_mode="ego")
        s = FakeSession()
        run_frames(t.main_image_monocular_zmq_ego, s, monkeypatch=monkeypatch)
        nodes = [nodes for to, nodes in s.calls if to == "bgChildren"][-1]
        assert [n["key"] for n in nodes] == ["background-mono", "wrist-left", "wrist-right"]
        assert nodes[0]["distanceToCamera"] == 2.0 and nodes[0]["height"] == pytest.approx(0.75)
    finally:
        views.close()


@pytest.mark.parametrize("binocular", [True, False])
def test_no_wrists_is_plain_televuer(monkeypatch, binocular):
    t = make_tvuer(None, binocular)
    s = FakeSession()
    coro = t.main_image_binocular_zmq if binocular else t.main_image_monocular_zmq
    run_frames(coro, s, monkeypatch=monkeypatch)
    keys = [[n["key"] for n in nodes] for _, nodes in s.calls]
    assert keys == [["motionControllers"],
                    ["background-left", "background-right"] if binocular else ["background-mono"]]
    assert all(to == "bgChildren" for to, _ in s.calls)


def test_install_patches_upstream_names():
    import teleimager.image_client as ic
    import televuer.tv_wrapper as tw
    old = (ic.ImageClient, tw.TeleVuer)
    try:
        tv.install("side", 100.0)
        assert tw.TeleVuer is tv.WristViewsTeleVuer and ic.ImageClient is tv.TrackedImageClient
        assert tv.WristViewsTeleVuer.initial_layout == "side"
    finally:
        ic.ImageClient, tw.TeleVuer = old
        tv.WristViewsTeleVuer.initial_layout = "pip"
        tv.WristViewsTeleVuer.side_span_deg = tv.SIDE_SPAN_DEG
