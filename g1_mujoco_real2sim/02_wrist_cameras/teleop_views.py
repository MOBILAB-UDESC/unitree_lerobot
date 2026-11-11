"""Run Unitree's xr_teleoperate unmodified, with the two wrist camera views added to the headset display.

    python 02_wrist_cameras/teleop_views.py [--wrist-layout pip|side|off] [--wrist-side-span DEG]
                                            <teleop_hand_and_arm.py flags...>
    (scripts/run_teleop.sh runs it with the sim flags; --plain there runs teleop_hand_and_arm.py as is)

Why a launcher instead of a patch: xr_teleoperate must stay a clean upstream clone. This file
  * subclasses televuer's TeleVuer and swaps it into televuer.tv_wrapper (TeleVuerWrapper looks the name
    up when it is called), so only the headset *display* changes;
  * subclasses teleimager's ImageClient only to remember the instance teleop_hand_and_arm.py creates; its
    subscribers already receive every enabled topic (wrists included), we just read their latest frames;
  * then runs teleop_hand_and_arm.py with runpy as __main__ (same argv, same cwd as a plain run).
The head image xr_teleoperate records is untouched (we never write to it), so --record is unaffected.

How the display works (televuer v4.0.0, vuer 0.0.60): TeleVuer forks a child process that runs the vuer
web server; the head frame reaches it through shared memory (img2display) and a per-session coroutine
upserts ImageBackground planes at display_fps. We create the wrist shared memory and the shared layout
value BEFORE TeleVuer.__init__ forks, copy wrist frames into it from the parent (like the head writer
thread), and override the display coroutines so the wrist planes go out in the SAME upsert as the head
plane (vuer's per-session send queue keeps only 3 messages, separate image messages get dropped).

Layouts (cycle pip -> side -> off):
  pip   head view unchanged; wrists small (13 deg tall), low in the head view's bottom corners
  side  head + wrists in one row, all the same angular size, wrists turned to face the eye; the row is
        kept within --wrist-side-span degrees (default 94, i.e. +-47 deg, inside the Quest 3 FOV), which in
        immersive mode means shrinking the head plane to ~0.41 of its normal size (ego mode fits as is)
  off   exactly the plain xr_teleoperate display
Toggle with: left controller X (controller input mode), the floating "WRIST VIEW" button in VR (point and
click/pinch), the button at the bottom-left of the web page, or the start option --wrist-layout.

If the image server enables no wrist topic (demo scene without the wrist_cameras add-on), or the display
is pass-through / WebRTC, nothing is changed: the display is plain xr_teleoperate.
"""

import argparse
import asyncio
import base64
import math
import os
import runpy
import sys
import threading
import time
from multiprocessing import Value, shared_memory

import cv2
import logging_mp
import numpy as np

if __name__ == "__main__":
    # teleop_hand_and_arm.py starts with logging_mp.basicConfig(INFO), which raises once any logger
    # exists - and importing teleimager/televuer below creates some. Do its call first (main() then
    # turns its own call into a no-op).
    logging_mp.basicConfig(level=logging_mp.INFO)
    if "-h" in sys.argv or "--help" in sys.argv:  # importing vuer prints its own help for -h and exits
        print(__doc__)

from vuer.schemas import Button, Hands, ImageBackground, MotionControllers, Plane

import teleimager.image_client as image_client_mod
import televuer.tv_wrapper as tv_wrapper_mod
from televuer.televuer import TeleVuer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
THIRD_PARTY = os.environ.get("G1_THIRD_PARTY", os.path.join(REPO, "third_party"))
TELEOP_DIR = os.path.join(THIRD_PARTY, "xr_teleoperate", "teleop")

LAYOUTS = ("pip", "side", "off")  # toggle order
SIDES = ("left", "right")

# Head plane as televuer draws it: (height m, distance m) per display mode (televuer.py main_image_*_zmq).
HEAD_PLANE = {"immersive": (1.0, 1.0), "ego": (0.75, 2.0)}
HEAD_QUALITY = 80             # same JPEG quality televuer uses for the head

SIDE_SPAN_DEG = 94.0          # side: total horizontal angle of head + both wrists (Quest 3 FOV ~110 deg)
SIDE_GAP_DEG = 1.0            # side: gap between neighbouring views
PIP_HEIGHT_DEG = 13.0         # pip: wrist view height ...
PIP_HEAD_FRACTION = 0.35      # ... but at most this fraction of the head view height (ego mode is small)
PIP_DISTANCE_SCALE = 0.9      # pip: a bit closer than the head plane so they are drawn on top of it
PIP_MAX_WIDTH = 320           # px; pip frames are downscaled before JPEG (less encode time / bandwidth)
WRIST_QUALITY = 70

# In-VR button: world-fixed (not head-locked, so it can be pointed at), placed up-left of where the
# operator looks - away from the hands' working area so a gripper trigger pull doesn't hit it by accident.
# It re-anchors when the operator turns/moves away from it.
BUTTON_YAW_DEG = -28.0        # left of the gaze direction
BUTTON_PITCH_DEG = 22.0       # above eye level
BUTTON_DISTANCE = 0.7         # m
BUTTON_SIZE = (0.15, 0.065)   # m
BUTTON_REANCHOR_YAW_DEG = 45.0
BUTTON_REANCHOR_DIST = 0.4    # m
BUTTON_COLORS = {"pip": (200, 110, 20), "side": (40, 150, 40), "off": (90, 90, 90)}  # BGR

KEY_WRIST = {"left": "wrist-left", "right": "wrist-right"}
KEY_BUTTON = "wrist-layout-button"        # 3D button (ON_CLICK)
KEY_HTML_BUTTON = "wrist-layout-html"     # page button (CLICK)
AUX_REFRESH_S = 1.0           # buttons go in their own upserts; resend periodically in case one is dropped
HIDE_FRAMES = 30              # frames to keep sending "hide" for wrist planes after switching to off
TOGGLE_DEBOUNCE_S = 0.3


# ----------------------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------------------
def place(yaw_deg, pitch_deg, height, aspect, distance):
    """Head-locked plane centred at (yaw right+, pitch up+) seen from the eye, turned to face the eye.

    vuer's ImagePlane puts a plane at camera * (position + [0, 0, -distanceToCamera]) with rotation
    relative to the camera; the Euler order "YXZ" (yaw, then pitch) keeps the plane's rows horizontal.
    """
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return {
        "position": [distance * math.cos(p) * math.sin(y), distance * math.sin(p), 0.0],
        "distanceToCamera": distance * math.cos(p) * math.cos(y),
        "rotation": [p, -y, 0.0, "YXZ"],
        "height": height,
        "aspect": aspect,
    }


def _half_angle(size, distance):
    return math.degrees(math.atan(size / 2.0 / distance))


def plan_layout(layout, display_mode, head_aspect, wrist_aspects, side_span_deg=SIDE_SPAN_DEG):
    """Plane geometry for one layout: {"head": {height, distanceToCamera}, "left"/"right": place(...)}."""
    h0, d = HEAD_PLANE[display_mode]
    out = {"head": {"height": h0, "distanceToCamera": d}}
    if layout == "off" or not wrist_aspects:
        return out
    wide = max(wrist_aspects.values())
    if layout == "side":
        # Largest head scale s <= 1 whose row (head + gap + wrist on each side) fits in +-span/2.
        def outer_edge(s):
            return _half_angle(s * h0 * head_aspect, d) + SIDE_GAP_DEG + 2 * _half_angle(s * h0 * wide, d)

        s, limit = 1.0, side_span_deg / 2.0
        if outer_edge(1.0) > limit:
            lo, hi = 0.0, 1.0
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if outer_edge(mid) <= limit else (lo, mid)
            s = lo
        h = s * h0
        out["head"]["height"] = h
        head_half_w = _half_angle(h * head_aspect, d)
        for side, sign in (("left", -1), ("right", 1)):
            if side in wrist_aspects:
                a = wrist_aspects[side]
                yaw = sign * (head_half_w + SIDE_GAP_DEG + _half_angle(h * a, d))
                out[side] = place(yaw, 0.0, h, a, d)
    elif layout == "pip":
        head_half_v = _half_angle(h0, d)
        head_half_w = _half_angle(h0 * head_aspect, d)
        pip_v = min(PIP_HEIGHT_DEG, PIP_HEAD_FRACTION * 2 * head_half_v)
        dd = d * PIP_DISTANCE_SCALE
        h = 2 * dd * math.tan(math.radians(pip_v / 2))
        for side, sign in (("left", -1), ("right", 1)):
            if side in wrist_aspects:
                a = wrist_aspects[side]
                # half inside the head view's bottom corner, bottom edge level with the head view's
                yaw = sign * (head_half_w - 0.5 * _half_angle(h * a, dd))
                out[side] = place(yaw, -(head_half_v - pip_v / 2), h, a, dd)
    else:
        raise ValueError(f"unknown wrist layout {layout!r}")
    return out


def wrist_streams(cam_config, display_mode, zmq, webrtc):
    """{side: (height, width, binocular)} of the wrist topics we can show; {} -> plain televuer display."""
    if not cam_config or display_mode not in HEAD_PLANE or not zmq or webrtc:
        return {}
    out = {}
    for side in SIDES:
        cfg = cam_config.get(f"{side}_wrist_camera") or {}
        if cfg.get("enable_zmq"):
            h, w = cfg["image_shape"][:2]
            binocular = bool(cfg.get("binocular"))
            out[side] = (int(h), int(w) // 2 if binocular else int(w), binocular)
    return out


def head_frame(head_pose):
    """(position, yaw) of a 4x4 OpenXR head pose (y up, -z forward); None if no pose arrived yet."""
    m = np.asarray(head_pose, dtype=float)
    if m.shape != (4, 4) or abs(np.linalg.det(m[:3, :3])) < 0.5:
        return None
    fwd = -m[:3, 2]
    return m[:3, 3].copy(), math.atan2(-fwd[0], -fwd[2])


def _label_png(layout):
    """Data URL of the 3D button's face (vuer's Plane loads `material.map` with a TextureLoader)."""
    img = np.zeros((128, 296, 3), np.uint8)
    img[:] = BUTTON_COLORS[layout]
    cv2.rectangle(img, (3, 3), (292, 124), (255, 255, 255), 3)
    cv2.putText(img, "WRIST VIEW", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, layout.upper(), (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, "click / X", (190, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    ok, png = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()


# ----------------------------------------------------------------------------------------------
# Display state shared by the parent (frame copy) and the vuer child (drawing, toggles)
# ----------------------------------------------------------------------------------------------
class WristViews:
    """Wrist frames in shared memory + the current layout; builds the vuer nodes for each frame.

    Create it before TeleVuer forks: the child inherits the shared memory and the layout Value.
    """

    def __init__(self, streams, layout="pip", display_mode="immersive", head_aspect=4 / 3,
                 side_span_deg=SIDE_SPAN_DEG):
        self.streams = dict(streams)
        self.sides = [s for s in SIDES if s in self.streams]
        self.display_mode = display_mode
        self._layout = Value("i", LAYOUTS.index(layout))
        self._shm, self.frames = {}, {}
        for side in self.sides:
            h, w, _ = self.streams[side]
            shm = shared_memory.SharedMemory(create=True, size=h * w * 3)
            self._shm[side] = shm
            self.frames[side] = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf)
            self.frames[side][:] = 0
        aspects = {s: self.streams[s][1] / self.streams[s][0] for s in self.sides}
        self.plans = {name: plan_layout(name, display_mode, head_aspect, aspects, side_span_deg) for name in LAYOUTS}
        self.labels = {name: _label_png(name) for name in LAYOUTS}
        self.debounce_s = TOGGLE_DEBOUNCE_S
        self._last_toggle = -1e9
        self._x_down = False
        self._anchor = None          # (position, yaw) the 3D button is placed relative to
        self.anchor_version = 0

    # --- layout ---
    @property
    def layout(self):
        return LAYOUTS[self._layout.value]

    def set_layout(self, name, source=""):
        with self._layout.get_lock():
            old = LAYOUTS[self._layout.value]
            self._layout.value = LAYOUTS.index(name)
        if name != old:
            print(f"[wrist views] layout {old} -> {name}" + (f" ({source})" if source else ""), flush=True)

    def toggle(self, source=""):
        now = time.monotonic()
        if now - self._last_toggle < self.debounce_s:
            return False
        self._last_toggle = now
        self.set_layout(LAYOUTS[(LAYOUTS.index(self.layout) + 1) % len(LAYOUTS)], source)
        return True

    def on_left_x(self, pressed):
        """Rising edge of the left controller's X button (televuer's left_ctrl_aButton) toggles."""
        pressed = bool(pressed)
        rising = pressed and not self._x_down
        self._x_down = pressed
        return self.toggle("left X") if rising else False

    def on_click(self, event):
        """vuer CLICK (page Button: event.key) and ON_CLICK (3D mesh: event.value['key'])."""
        key = getattr(event, "key", None)
        value = getattr(event, "value", None)
        if key is None and isinstance(value, dict):
            key = value.get("key")
        if key == KEY_BUTTON:
            return self.toggle("VR button")
        if key == KEY_HTML_BUTTON:
            return self.toggle("page button")
        return False

    # --- frames (parent) ---
    def write_frame(self, side, bgr):
        """BGR frame from teleimager -> RGB in shared memory (left eye only if the topic is stereo)."""
        dst = self.frames[side]
        h, w, binocular = self.streams[side]
        if binocular:
            bgr = bgr[:, : bgr.shape[1] // 2]
        if bgr.shape[:2] != (h, w):
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=dst)

    def close(self):
        for shm in self._shm.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shm.clear()

    # --- vuer nodes (child) ---
    def head_nodes(self, img, binocular, img_width, aspect):
        """televuer's head plane(s), at the size the current layout wants."""
        g = self.plans[self.layout]["head"]
        common = dict(aspect=aspect, height=g["height"], distanceToCamera=g["distanceToCamera"],
                      format="jpeg", quality=HEAD_QUALITY, interpolate=True)
        if binocular:
            return [ImageBackground(img[:, :img_width], layers=1, key="background-left", **common),
                    ImageBackground(img[:, img_width:], layers=2, key="background-right", **common)]
        return [ImageBackground(img, key="background-mono", **common)]

    @staticmethod
    def new_session():
        return {"hide": HIDE_FRAMES, "aux_t": -1e9, "aux_layout": None, "aux_anchor": None}

    def wrist_nodes(self, st):
        """Wrist planes (no `layers`: both eyes see the same mono image) or, after switching to off,
        hide markers for a while (vuer's upsert merges props, so planes must be hidden explicitly)."""
        layout = self.layout
        if layout == "off":
            if st["hide"] <= 0:
                return []
            st["hide"] -= 1
            return [ImageBackground(key=KEY_WRIST[s], hide=True) for s in self.sides]
        st["hide"] = HIDE_FRAMES
        plan = self.plans[layout]
        nodes = []
        for side in self.sides:
            img = self.frames[side]
            if layout == "pip" and img.shape[1] > PIP_MAX_WIDTH:
                img = cv2.resize(img, (PIP_MAX_WIDTH, round(img.shape[0] * PIP_MAX_WIDTH / img.shape[1])),
                                 interpolation=cv2.INTER_AREA)
            nodes.append(ImageBackground(img, key=KEY_WRIST[side], format="jpeg", quality=WRIST_QUALITY,
                                         interpolate=True, hide=False, **plan[side]))
        return nodes

    def update_anchor(self, head_pose):
        """Re-place the 3D button when there is no anchor yet or the operator turned/moved away."""
        hf = head_frame(head_pose)
        if hf is None:
            return
        pos, yaw = hf
        if self._anchor is not None:
            apos, ayaw = self._anchor
            dyaw = abs((yaw - ayaw + math.pi) % (2 * math.pi) - math.pi)
            if math.degrees(dyaw) < BUTTON_REANCHOR_YAW_DEG and np.linalg.norm(pos - apos) < BUTTON_REANCHOR_DIST:
                return
        self._anchor = (pos, yaw)
        self.anchor_version += 1

    def button_node(self):
        """World-fixed, clickable Plane (vuer mesh primitives publish ON_CLICK) facing the anchor pose."""
        if self._anchor is None:
            return None
        pos, psi = self._anchor
        local = place(BUTTON_YAW_DEG, BUTTON_PITCH_DEG, 0, 0, BUTTON_DISTANCE)
        x, y = local["position"][:2]
        z = -local["distanceToCamera"]
        world = [pos[0] + x * math.cos(psi) + z * math.sin(psi), pos[1] + y,
                 pos[2] - x * math.sin(psi) + z * math.cos(psi)]
        pitch, yaw = local["rotation"][:2]
        return Plane(key=KEY_BUTTON, args=list(BUTTON_SIZE), position=world, rotation=[pitch, psi + yaw, 0.0, "YXZ"],
                     materialType="basic", material={"map": self.labels[self.layout], "side": 2, "toneMapped": False})

    def html_node(self):
        return Button(key=KEY_HTML_BUTTON, value=f"Wrist views: {self.layout.upper()} (click to change)",
                      style={"position": "fixed", "left": "16px", "bottom": "16px", "zIndex": 1000,
                             "padding": "10px 16px", "fontSize": "16px", "borderRadius": "6px"})

    def aux_upserts(self, st, head_pose):
        """[(node, to), ...] for the buttons, when something changed or AUX_REFRESH_S passed."""
        self.update_anchor(head_pose)
        now = time.monotonic()
        if (now - st["aux_t"] < AUX_REFRESH_S and st["aux_layout"] == self.layout
                and st["aux_anchor"] == self.anchor_version):
            return []
        st.update(aux_t=now, aux_layout=self.layout, aux_anchor=self.anchor_version)
        out = [(self.html_node(), "htmlChildren")]
        button = self.button_node()
        if button is not None:
            # "children" sits inside a Suspense boundary: loading the label texture can't blank the scene.
            out.append((button, "children"))
        return out


# ----------------------------------------------------------------------------------------------
# TeleVuer with wrist views
# ----------------------------------------------------------------------------------------------
class TrackedImageClient(image_client_mod.ImageClient):
    """teleimager's ImageClient, remembering the instance teleop_hand_and_arm.py creates."""

    latest = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        TrackedImageClient.latest = self


class WristViewsTeleVuer(TeleVuer):
    """TeleVuer that also draws the wrist cameras (see module docstring). Same constructor as TeleVuer."""

    initial_layout = "pip"
    side_span_deg = SIDE_SPAN_DEG

    def __init__(self, use_hand_tracking, binocular=True, img_shape=None, display_fps=30.0,
                 display_mode="immersive", zmq=False, webrtc=False, webrtc_url=None, cert_file=None, key_file=None):
        self._views = None
        client = TrackedImageClient.latest
        streams = wrist_streams(client.get_cam_config() if client else None, display_mode, zmq, webrtc)
        if streams and img_shape is not None:
            head_w = img_shape[1] // 2 if binocular else img_shape[1]
            self._views = WristViews(streams, self.initial_layout, display_mode, head_w / img_shape[0],
                                     self.side_span_deg)
        # TeleVuer forks the vuer process at the end of __init__: shared state must exist before this.
        super().__init__(use_hand_tracking, binocular=binocular, img_shape=img_shape, display_fps=display_fps,
                         display_mode=display_mode, zmq=zmq, webrtc=webrtc, webrtc_url=webrtc_url,
                         cert_file=cert_file, key_file=key_file)
        if self._views is None:
            print("[wrist views] no wrist camera stream to show (or not a ZMQ immersive/ego display): "
                  "plain xr_teleoperate display", flush=True)
            return
        self._client = client
        self._wrist_stop = threading.Event()
        self._wrist_thread = threading.Thread(target=self._wrist_copy_loop, daemon=True)
        self._wrist_thread.start()
        desc = ", ".join(f"{s} {w}x{h}" for s, (h, w, _) in self._views.streams.items())
        toggles = "page button, VR button" + ("" if use_hand_tracking else ", left X")
        print(f"[wrist views] {desc}; layout {self._views.layout}; toggle: {toggles}", flush=True)

    def _wrist_copy_loop(self):
        """Parent: latest wrist frames from teleimager's subscribers -> shared memory."""
        getters = {"left": self._client.get_left_wrist_frame, "right": self._client.get_right_wrist_frame}
        while not self._wrist_stop.wait(1.0 / self.display_fps):
            if self._views.layout == "off":
                continue
            for side in self._views.sides:
                try:
                    bgr = getters[side]().bgr
                except Exception:
                    bgr = None
                if bgr is not None:
                    self._views.write_frame(side, bgr)

    def _vuer_run(self):
        # Runs in the forked child: handlers registered in the parent after the fork would never get here.
        if self._views is not None:
            self.vuer.add_handler("CLICK")(self._on_click)
            self.vuer.add_handler("ON_CLICK")(self._on_click)
        super()._vuer_run()

    async def _on_click(self, event, session):
        self._views.on_click(event)

    async def on_controller_move(self, event, session, fps=60):
        await super().on_controller_move(event, session, fps)
        if self._views is not None:
            self._views.on_left_x(self.left_ctrl_aButton_shared.value)

    def close(self):
        super().close()
        if self._views is not None:
            self._wrist_stop.set()
            self._wrist_thread.join(timeout=0.5)
            self._views.close()

    # --- display coroutines (vuer child, one per websocket session) ---
    async def main_image_binocular_zmq(self, session):
        if self._views is None:
            return await super().main_image_binocular_zmq(session)
        await self._display_loop(session)

    async def main_image_monocular_zmq(self, session):
        if self._views is None:
            return await super().main_image_monocular_zmq(session)
        await self._display_loop(session)

    async def main_image_binocular_zmq_ego(self, session):
        if self._views is None:
            return await super().main_image_binocular_zmq_ego(session)
        await self._display_loop(session)

    async def main_image_monocular_zmq_ego(self, session):
        if self._views is None:
            return await super().main_image_monocular_zmq_ego(session)
        await self._display_loop(session)

    async def _display_loop(self, session):
        if self.use_hand_tracking:
            session.upsert(Hands(stream=True, key="hands", hideLeft=True, hideRight=True), to="bgChildren")
        else:
            session.upsert(MotionControllers(stream=True, key="motionControllers", left=True, right=True),
                           to="bgChildren")
        views, st = self._views, self._views.new_session()
        while True:
            session.upsert(views.head_nodes(self.img2display, self.binocular, self.img_width, self.aspect_ratio)
                           + views.wrist_nodes(st), to="bgChildren")
            for node, to in views.aux_upserts(st, self.head_pose):
                session.upsert(node, to=to)
            await asyncio.sleep(1.0 / self.display_fps)


# ----------------------------------------------------------------------------------------------
def install(layout="pip", side_span_deg=SIDE_SPAN_DEG):
    """Swap our classes into the (unmodified) televuer / teleimager modules teleop_hand_and_arm.py uses."""
    WristViewsTeleVuer.initial_layout = layout
    WristViewsTeleVuer.side_span_deg = side_span_deg
    image_client_mod.ImageClient = TrackedImageClient
    tv_wrapper_mod.TeleVuer = WristViewsTeleVuer


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False,
                                usage="teleop_views.py [--wrist-layout L] [--wrist-side-span DEG] "
                                      "<teleop_hand_and_arm.py args>")
    p.add_argument("--wrist-layout", choices=LAYOUTS, default="pip", help="wrist views at start (default pip)")
    p.add_argument("--wrist-side-span", type=float, default=SIDE_SPAN_DEG,
                   help=f"side layout: horizontal degrees the three views may cover (default {SIDE_SPAN_DEG:g})")
    ours, rest = p.parse_known_args(argv)

    script = os.path.join(TELEOP_DIR, "teleop_hand_and_arm.py")
    if not os.path.isfile(script):
        sys.exit(f"xr_teleoperate not found: {script} (set G1_THIRD_PARTY)")
    install(ours.wrist_layout, ours.wrist_side_span)
    logging_mp.basicConfig = lambda *a, **k: None  # already configured identically above
    # Same environment as `cd teleop && python teleop_hand_and_arm.py ...`.
    sys.argv = [script] + rest
    os.chdir(TELEOP_DIR)
    sys.path.insert(0, TELEOP_DIR)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
