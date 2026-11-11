"""Run a remote vision-language-action (VLA) policy on the G1 arms, closed loop at 30 Hz.

    python -m g1lab.remote_vla --backend dream  --prompt "put the wrapped box inside the cardboard box"
    python -m g1lab.remote_vla --backend openpi --url ws://gpu-box:8000
    python -m g1lab.remote_vla --backend gr00t  --url tcp://gpu-box:5555 --episodes 5 \
        --record datasets/vla_rollouts --repo-id local/vla_rollouts

Every control tick (1/30 s):  observe -> (when the action buffer runs low) ask the policy server for
an action chunk in a background thread -> play the buffered actions through G1Arms.command().

Chunks arrive at the server's rate (Dream-1: 20 Hz) and are resampled to 30 Hz. Rows the robot already
played while the request was in flight are skipped (latency alignment), and the seam between the old
and the new chunk is blended linearly over --blend-steps ticks. Every command then passes a safety
filter: clip to the URDF joint limits and limit the per-tick joint change (--max-step-rad).

The loop stops on --max-steps, on robot.success() (sim), or on Ctrl+C, and then calls go_home().

Backends
  dream   Dream Labs Dream-1, HTTPS JSON (POST /v1/get_action, Bearer $DREAM_API_KEY).
  openpi  Physical Intelligence openpi policy server, websocket + msgpack-numpy.
  gr00t   NVIDIA Isaac GR00T inference server, ZMQ REQ/REP + msgpack (np.save-encoded arrays).
Wire key names live in DREAM_FIELDS / OPENPI_KEYS / GR00T_KEYS and can be overridden with
--keys '{"name": "value", ...}' (a null value drops that field). Local mock servers for all three
protocols are in g1lab/vla_mocks.py.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
XR_TELEOP = os.path.join(os.environ.get("G1_THIRD_PARTY", os.path.join(os.path.dirname(ROOT), "third_party")),
                         "xr_teleoperate")

CONTROL_HZ = 30.0
DEFAULT_TASK = "put the wrapped box inside the cardboard box"
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
NUM_JOINTS = 14

# ------------------------------------------------------------------------------------------ limits
URDF_PATH = os.path.join(XR_TELEOP, "assets", "g1", "g1_body29_hand14.urdf")
URDF_JOINTS = [f"{side}_{j}_joint" for side in ("left", "right") for j in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
# Copied from g1_body29_hand14.urdf; only used if the URDF cannot be read.
_FALLBACK_LIMITS = np.array([
    [-3.0892, 2.6704], [-1.5882, 2.2515], [-2.618, 2.618], [-1.0472, 2.0944],
    [-1.972222054, 1.972222054], [-1.614429558, 1.614429558], [-1.614429558, 1.614429558],
    [-3.0892, 2.6704], [-2.2515, 1.5882], [-2.618, 2.618], [-1.0472, 2.0944],
    [-1.972222054, 1.972222054], [-1.614429558, 1.614429558], [-1.614429558, 1.614429558],
])


def load_joint_limits(urdf=URDF_PATH, margin=0.0):
    """(lower, upper) arrays (14,) for the arm joints in G1Arms order, read from the URDF."""
    lims = _FALLBACK_LIMITS.copy()
    try:
        joints = {j.get("name"): j for j in ET.parse(urdf).getroot().iter("joint")}
        for i, name in enumerate(URDF_JOINTS):
            lim = joints[name].find("limit")
            lims[i] = float(lim.get("lower")), float(lim.get("upper"))
    except (OSError, ET.ParseError, KeyError, AttributeError, TypeError) as e:
        print(f"[safety] could not read {urdf} ({e}); using built-in G1 limits", flush=True)
    return lims[:, 0] + margin, lims[:, 1] - margin


class SafetyFilter:
    """Clip to joint limits, then limit the change from the previous command to max_step per tick."""

    def __init__(self, lower, upper, max_step=0.05):
        self.lower, self.upper, self.max_step = np.asarray(lower), np.asarray(upper), float(max_step)
        self.clipped = 0      # ticks where the limit clip was active
        self.rate_limited = 0  # ticks where the per-step limit was active
        self.rejected = 0     # non-finite targets (held instead)

    def __call__(self, target, prev):
        prev = np.asarray(prev, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if target.shape != prev.shape or not np.all(np.isfinite(target)):
            self.rejected += 1
            return prev.copy()
        t = np.clip(target, self.lower, self.upper)
        self.clipped += bool(np.any(t != target))
        d = t - prev
        dc = np.clip(d, -self.max_step, self.max_step)
        self.rate_limited += bool(np.any(dc != d))
        return np.clip(prev + dc, self.lower, self.upper)


# ----------------------------------------------------------------------------------- msgpack-numpy
def _pack_openpi(obj):
    """openpi_client.msgpack_numpy wire format."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "VOc":
            raise ValueError(f"unsupported dtype {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _pack_gr00t(obj):
    """Isaac-GR00T gr00t/eval/service.py MsgSerializer wire format (np.save bytes)."""
    import io
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _unpack_any(obj):
    import io
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    if "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


def packb(obj, style="openpi"):
    import msgpack
    return msgpack.packb(obj, default=_pack_openpi if style == "openpi" else _pack_gr00t, use_bin_type=True)


def unpackb(data):
    import msgpack
    return msgpack.unpackb(data, object_hook=_unpack_any, raw=False, strict_map_key=False)


# ---------------------------------------------------------------------------------------- images
def resize_rgb(img, width, height):
    import cv2
    if img.shape[1] == width and img.shape[0] == height:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def resize_with_pad(img, height, width):
    """openpi_client.image_tools.resize_with_pad: keep aspect ratio, pad with black."""
    import cv2
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    out = np.zeros((height, width, 3), dtype=np.uint8)
    y0, x0 = (height - nh) // 2, (width - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def jpeg_b64(rgb, quality=85):
    import cv2
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class FrameHistory:
    """Last `length` camera frames sampled every 1/fps s of control time (oldest first)."""

    def __init__(self, length=1, fps=CONTROL_HZ, control_hz=CONTROL_HZ, transform=None):
        self.period = max(1, int(round(control_hz / fps)))
        self.buf = deque(maxlen=max(1, int(length)))
        self.transform = transform
        self._last_step = None

    def clear(self):
        self.buf.clear()
        self._last_step = None

    def push(self, step, images):
        if not images:
            return
        if self._last_step is None or step - self._last_step >= self.period:
            self.buf.append(self.transform(images) if self.transform else images)
            self._last_step = step

    def frames(self):
        return list(self.buf)


# --------------------------------------------------------------------------------- chunk handling
def resample_chunk(chunk, src_hz, dst_hz=CONTROL_HZ):
    """Linear resample of an [H, D] action chunk whose row k is at t = k / src_hz."""
    chunk = np.asarray(chunk, dtype=np.float64)
    if len(chunk) <= 1 or abs(src_hz - dst_hz) < 1e-9:
        return chunk.copy()
    t_src = np.arange(len(chunk)) / float(src_hz)
    n = int(np.floor(t_src[-1] * dst_hz + 1e-9)) + 1
    t_dst = np.arange(n) / float(dst_hz)
    return np.stack([np.interp(t_dst, t_src, chunk[:, j]) for j in range(chunk.shape[1])], axis=1)


class ChunkPlayer:
    """30 Hz action buffer with latency alignment and linear seam blending."""

    def __init__(self, blend_steps=5, execute_ratio=0.625):
        self.blend_steps = int(blend_steps)
        self.execute_ratio = float(execute_ratio)
        self.reset()

    def reset(self):
        self.plan = np.zeros((0, NUM_JOINTS))
        self.idx = 0
        self.chunk_len = 0     # length of the current chunk before the latency skip
        self.chunk_played = 0  # rows of the current chunk consumed (skip + played)
        self.played = 0        # total rows played from plans (monotonic, for latency alignment)
        self.last = None
        self.chunks = 0

    def remaining(self):
        return len(self.plan) - self.idx

    def needs_request(self):
        return self.chunk_len == 0 or self.chunk_played >= self.execute_ratio * self.chunk_len or self.remaining() == 0

    def add_chunk(self, chunk30, skip=0, current=None):
        """Install a resampled chunk. skip = rows played since its observation was taken."""
        chunk30 = np.asarray(chunk30, dtype=np.float64)
        new = chunk30[skip:].copy()
        if len(new) == 0:
            return False
        old = self.plan[self.idx:]
        ref = self.last if self.last is not None else (current if current is not None else new[0])
        n = min(self.blend_steps, len(new))
        for i in range(n):
            o = old[i] if i < len(old) else (old[-1] if len(old) else ref)
            w = (i + 1) / (n + 1)
            new[i] = (1.0 - w) * o + w * new[i]
        self.plan, self.idx = new, 0
        self.chunk_len, self.chunk_played = len(chunk30), skip
        self.chunks += 1
        return True

    def next(self):
        """(action, starved). Holds the last action when the buffer is empty."""
        if self.idx < len(self.plan):
            a = self.plan[self.idx]
            self.idx += 1
            self.played += 1
            self.chunk_played += 1
            self.last = a
            return a, False
        return self.last, True


# -------------------------------------------------------------------------------------- backends
class PolicyError(RuntimeError):
    def __init__(self, msg, fatal=False):
        super().__init__(msg)
        self.fatal = fatal


class Backend:
    name = "base"
    default_url = ""
    default_hz = CONTROL_HZ
    history_len = 1
    history_fps = CONTROL_HZ

    def frame_transform(self, images):
        return images

    def connect(self):
        pass

    def infer(self, state, frames, prompt):
        """-> (chunk [H, 14] float64 joint targets, chunk rate in Hz)."""
        raise NotImplementedError

    def reset(self):
        pass

    def close(self):
        pass


def _override(base, keys):
    out = dict(base)
    for k, v in (keys or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


# Dream Labs Dream-1. Field names follow the public client (dream_library/dream_client/src/dream/
# client.py + transport.py) at https://github.com/dreamlabs-co/dreamlabs and the API docs at
# https://platform.dreamlabs.ai/api-documentation. They may change: everything that goes on the wire
# is named here, and --keys '{...}' overrides any entry (null drops an optional field).
DREAM_FIELDS = {
    "path_get_action": "/v1/get_action",
    "path_end_session": "/v1/end_session",
    # request
    "checkpoint": "checkpoint",
    "session_id": "session_id",
    "reset": "reset",
    "observation": "observation",          # null -> observation keys go at the top level
    "prompt": "prompt",
    "embodiment": "embodiment",
    "cameras": {"cam_head": "video.overhead", "cam_left_wrist": "video.wrist_l", "cam_right_wrist": "video.wrist_r"},
    "state_left_arm": "state.left_arm",
    "state_right_arm": "state.right_arm",
    "state_left_gripper": "state.left_gripper",    # G1 rubber hands: constant --gripper value
    "state_right_gripper": "state.right_gripper",
    # response
    "resp_status": "status",
    "resp_code": "code",
    "resp_retry_after": "retry_after_s",
    "resp_session_id": "session_id",
    "resp_actions": "actions",             # chunks are looked up here, then at the top level
    "resp_left_chunk": "q_l_chunk",        # [H, 7 joints + 1 gripper]
    "resp_right_chunk": "q_r_chunk",
    "resp_action_hz": "action_hz",         # optional; else --action-hz, else 20
}
DREAM_OBS = dict(frames=9, fps=5.0, width=320, height=176, jpeg_quality=85)
DREAM_OK = ("DA-0000", "DA-0001")         # OK / OK_NEW_SESSION
DREAM_RETRYABLE = ("DA-2000",)            # rate_limited / pool_warming
DREAM_FATAL = ("DA-1000", "DA-4000")      # unauthorized / bad request


class DreamBackend(Backend):
    name = "dream"
    default_url = "https://api.dreamlabs.ai"
    default_hz = 20.0
    history_len = DREAM_OBS["frames"]
    history_fps = DREAM_OBS["fps"]

    def __init__(self, url=None, api_key=None, checkpoint="dreamlabs/dream-1-beta", embodiment=None,
                 keys=None, action_hz=None, timeout=15.0, gripper=0.0):
        self.fields = _override(DREAM_FIELDS, keys)
        url = url or self.default_url
        u = urllib.parse.urlsplit(url if "://" in url else "https://" + url)
        self.scheme, self.host, self.port = u.scheme, u.hostname, u.port
        self.base_path = u.path.rstrip("/")
        for p in (self.fields["path_get_action"], self.fields["path_end_session"]):
            if self.base_path.endswith(p):  # accept the full endpoint URL as --url
                self.base_path = self.base_path[: -len(p)]
        self.api_key = api_key
        self.checkpoint, self.embodiment = checkpoint, embodiment
        self.action_hz, self.timeout, self.gripper = action_hz, float(timeout), float(gripper)
        self.session_id = None
        self.last_response = None
        self.last_request = None
        self._conn = None

    def frame_transform(self, images):
        o = DREAM_OBS
        return {k: jpeg_b64(resize_rgb(v, o["width"], o["height"]), o["jpeg_quality"]) for k, v in images.items()}

    def _new_conn(self, timeout):
        cls = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=timeout)

    def _post(self, path, payload, timeout=None):
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
                   "User-Agent": "g1lab-remote-vla/1.0"}
        for attempt in range(2):  # retry once on a fresh connection (idle keep-alive closed by the gateway)
            if self._conn is None:
                self._conn = self._new_conn(timeout or self.timeout)
            try:
                if timeout:
                    self._conn.timeout = timeout
                self._conn.request("POST", self.base_path + path, body=body, headers=headers)
                r = self._conn.getresponse()
                raw = r.read()
                break
            except (http.client.HTTPException, ConnectionError, BrokenPipeError, socket.timeout, OSError):
                self._conn.close()
                self._conn = None
                if attempt:
                    raise
        try:
            data = json.loads(raw.decode() or "{}")
        except ValueError:
            data = {"_raw": raw[:500].decode(errors="replace")}
        return r.status, data

    def build_request(self, state, frames, prompt):
        F = self.fields
        q = np.asarray(state, dtype=np.float64).reshape(NUM_JOINTS)
        obs = {F["state_left_arm"]: q[:7].tolist(), F["state_right_arm"]: q[7:].tolist()}
        for k in ("state_left_gripper", "state_right_gripper"):
            if F.get(k):
                obs[F[k]] = [self.gripper]
        for cam, key in F["cameras"].items():
            if key:
                obs[key] = [f[cam] for f in frames if cam in f]
        obs[F["prompt"]] = prompt
        if self.embodiment and F.get("embodiment"):
            obs[F["embodiment"]] = self.embodiment
        payload = {F["checkpoint"]: self.checkpoint, F["session_id"]: self.session_id}
        if F.get("reset"):
            payload[F["reset"]] = False
        if F.get("observation"):
            payload[F["observation"]] = obs
        else:
            payload.update(obs)
        return payload

    def infer(self, state, frames, prompt):
        F = self.fields
        payload = self.build_request(state, frames, prompt)
        self.last_request = payload
        for attempt in range(4):
            status, resp = self._post(F["path_get_action"], payload)
            st = resp.get(F["resp_status"]) if isinstance(resp, dict) else None
            st = st if isinstance(st, dict) else {}
            code = st.get(F["resp_code"])
            if code in DREAM_RETRYABLE or (code is None and status in (429, 503)):
                time.sleep(float(st.get(F["resp_retry_after"]) or 0.7))
                continue
            break
        self.last_response = resp
        if code in DREAM_FATAL or status in (401, 403):
            raise PolicyError(f"Dream {status} {code}: {st.get('reason')} {st.get('message', resp)}", fatal=True)
        if (code is not None and code not in DREAM_OK) or status >= 400:
            raise PolicyError(f"Dream {status} {code}: {st.get('reason')} {st.get('message', resp)}")
        sid = resp.get(F["resp_session_id"])
        if code == "DA-0001" and self.session_id and sid != self.session_id:
            print(f"[dream] session replaced ({st.get('reason')}): {self.session_id} -> {sid}", flush=True)
        if sid:
            self.session_id = sid
        acts = resp.get(F["resp_actions"]) if F.get("resp_actions") else None
        if not isinstance(acts, dict) or F["resp_left_chunk"] not in acts:
            acts = resp
        try:
            ql = np.asarray(acts[F["resp_left_chunk"]], dtype=np.float64)
            qr = np.asarray(acts[F["resp_right_chunk"]], dtype=np.float64)
        except KeyError as e:
            raise PolicyError(f"Dream response has no {e} (keys: {sorted(resp)})")
        if ql.ndim != 2 or qr.ndim != 2 or ql.shape[1] < 7 or qr.shape[1] < 7 or len(ql) != len(qr):
            raise PolicyError(f"Dream chunk shapes {ql.shape} / {qr.shape}, expected [H, 7(+1)]")
        hz = acts.get(F["resp_action_hz"]) or resp.get(F["resp_action_hz"]) or self.action_hz or self.default_hz
        return np.concatenate([ql[:, :7], qr[:, :7]], axis=1), float(hz)

    def reset(self):
        """New episode: release the session; the next get_action sends session_id null."""
        if self.session_id:
            F = self.fields
            try:
                self._post(F["path_end_session"], {F["checkpoint"]: self.checkpoint, F["session_id"]: self.session_id},
                           timeout=2.0)
            except Exception as e:  # never raises, like the official client
                print(f"[dream] end_session failed (ignored): {e}", flush=True)
        self.session_id = None

    def close(self):
        self.reset()
        if self._conn is not None:
            self._conn.close()
            self._conn = None


OPENPI_KEYS = {
    "cam_head": "observation/exterior_image_1_left",
    "cam_left_wrist": "observation/wrist_image_left",
    "cam_right_wrist": "observation/wrist_image_right",
    "state": "observation/joint_position",
    "prompt": "prompt",
    "actions": "actions",
}


class OpenPIBackend(Backend):
    """openpi serve_policy.py: websocket, msgpack-numpy, metadata on connect, {"actions": [H, D]} replies."""
    name = "openpi"
    default_url = "ws://localhost:8000"

    def __init__(self, url=None, keys=None, image_size=224, action_hz=None, timeout=15.0, api_key=None):
        self.url = url or self.default_url
        if "://" not in self.url:
            self.url = "ws://" + self.url
        self.keys = _override(OPENPI_KEYS, keys)
        self.image_size = int(image_size or 0)
        self.action_hz = action_hz or self.default_hz
        self.timeout = float(timeout)
        self.api_key = api_key
        self.ws = None
        self.metadata = None
        self.last_request = None

    def connect(self):
        from websockets.sync.client import connect
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        self.ws = connect(self.url, compression=None, max_size=None, open_timeout=self.timeout,
                          additional_headers=headers)
        self.metadata = unpackb(self.ws.recv(timeout=self.timeout))
        print(f"[openpi] connected to {self.url}, server metadata: {self.metadata}", flush=True)

    def build_request(self, state, frames, prompt):
        k = self.keys
        obs = {}
        latest = frames[-1] if frames else {}
        for cam in CAMERAS:
            if k.get(cam) and cam in latest:
                img = latest[cam]
                if self.image_size:
                    img = resize_with_pad(img, self.image_size, self.image_size)
                obs[k[cam]] = np.ascontiguousarray(img, dtype=np.uint8)
        obs[k["state"]] = np.asarray(state, dtype=np.float32).reshape(NUM_JOINTS)
        if k.get("prompt"):
            obs[k["prompt"]] = prompt
        return obs

    def infer(self, state, frames, prompt):
        if self.ws is None:
            self.connect()
        obs = self.build_request(state, frames, prompt)
        self.last_request = obs
        try:
            self.ws.send(packb(obs, "openpi"))
            resp = self.ws.recv(timeout=self.timeout)
        except Exception:
            self.close()  # reconnect on the next call
            raise
        if isinstance(resp, str):
            raise PolicyError(f"openpi server error:\n{resp}")
        d = unpackb(resp)
        a = np.asarray(d[self.keys["actions"]], dtype=np.float64)
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 2 or a.shape[1] < NUM_JOINTS:
            raise PolicyError(f"openpi actions shape {a.shape}, expected [H, >=14]")
        return a[:, :NUM_JOINTS], float(self.action_hz)

    def close(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None


GR00T_KEYS = {
    "cam_head": "video.cam_head",
    "cam_left_wrist": "video.cam_left_wrist",
    "cam_right_wrist": "video.cam_right_wrist",
    "state_left_arm": "state.left_arm",
    "state_right_arm": "state.right_arm",
    "prompt": "annotation.human.task_description",
    "action_left_arm": "action.left_arm",
    "action_right_arm": "action.right_arm",
}


class Gr00tBackend(Backend):
    """Isaac-GR00T inference service: ZMQ REQ {"endpoint", "data", ["api_token"]} -> dict (or {"error"})."""
    name = "gr00t"
    default_url = "tcp://localhost:5555"

    def __init__(self, url=None, keys=None, image_size=0, action_hz=None, timeout=15.0, api_token=None):
        self.url = url or self.default_url
        if "://" not in self.url:
            self.url = "tcp://" + self.url
        self.keys = _override(GR00T_KEYS, keys)
        self.image_size = int(image_size or 0)
        self.action_hz = action_hz or self.default_hz
        self.timeout = float(timeout)
        self.api_token = api_token
        self._ctx = self._sock = None
        self.last_request = None

    def _socket(self):
        import zmq
        if self._ctx is None:
            self._ctx = zmq.Context.instance()
        if self._sock is None:
            s = self._ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
            s.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
            s.connect(self.url)
            self._sock = s
        return self._sock

    def call(self, endpoint, data=None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        if self.api_token:
            req["api_token"] = self.api_token
        s = self._socket()
        try:
            s.send(packb(req, "gr00t"))
            resp = unpackb(s.recv())
        except Exception:
            s.close()  # a REQ socket is stuck after a timeout: rebuild it
            self._sock = None
            raise
        if isinstance(resp, dict) and "error" in resp:
            raise PolicyError(f"gr00t server error on '{endpoint}': {resp['error']}",
                              fatal="token" in str(resp["error"]).lower())
        return resp

    def connect(self):
        print(f"[gr00t] ping {self.url}: {self.call('ping')}", flush=True)

    def build_request(self, state, frames, prompt):
        k = self.keys
        q = np.asarray(state, dtype=np.float64).reshape(NUM_JOINTS)
        obs = {}
        latest = frames[-1] if frames else {}
        for cam in CAMERAS:
            if k.get(cam) and cam in latest:
                img = latest[cam]
                if self.image_size:
                    img = resize_rgb(img, self.image_size, self.image_size)
                obs[k[cam]] = np.ascontiguousarray(img, dtype=np.uint8)[None]
        obs[k["state_left_arm"]] = q[None, :7]
        obs[k["state_right_arm"]] = q[None, 7:]
        if k.get("prompt"):
            obs[k["prompt"]] = [prompt]
        return obs

    def infer(self, state, frames, prompt):
        obs = self.build_request(state, frames, prompt)
        self.last_request = obs
        resp = self.call("get_action", obs)
        if isinstance(resp, (list, tuple)):  # newer servers reply (action, info)
            resp = resp[0]
        k = self.keys
        parts = []
        for key in (k["action_left_arm"], k["action_right_arm"]):
            v = resp.get(key, resp.get(key.split(".", 1)[-1]))
            if v is None:
                raise PolicyError(f"gr00t reply has no '{key}' (keys: {sorted(resp)})")
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 3 and v.shape[0] == 1:
                v = v[0]
            parts.append(v)
        if parts[0].ndim != 2 or parts[0].shape[1] != 7 or parts[0].shape != parts[1].shape:
            raise PolicyError(f"gr00t action shapes {parts[0].shape} / {parts[1].shape}, expected [H, 7]")
        return np.concatenate(parts, axis=1), float(self.action_hz)

    def reset(self):
        try:
            self.call("reset")
        except PolicyError as e:
            print(f"[gr00t] reset not supported by server (ignored): {e}", flush=True)

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


BACKENDS = {"dream": DreamBackend, "openpi": OpenPIBackend, "gr00t": Gr00tBackend}


# ---------------------------------------------------------------------------------------- runner
def open_writer(root, repo_id, task, fps=int(CONTROL_HZ)):
    """Lazy LeRobotWriter; None (with a clear message) if it or lerobot is not importable here."""
    try:
        from g1lab.lerobot_writer import LeRobotWriter
    except Exception as e:  # ImportError, or lerobot missing inside the writer module
        print(f"[record] g1lab.lerobot_writer is not usable in this env ({type(e).__name__}: {e}).\n"
              f"[record] Continuing WITHOUT recording. Run in an env with lerobot (e.g. G1LAB_ENV=g1lab "
              f"scripts/run_vla.sh ...) to record.", flush=True)
        return None
    return LeRobotWriter(root, repo_id, task, fps=fps)


class Runner:
    def __init__(self, robot, backend, prompt=DEFAULT_TASK, max_step=0.05, blend_steps=5, execute_ratio=0.625,
                 control_hz=CONTROL_HZ, writer=None, save_on="all", min_success_steps=15, verbose=True):
        self.robot, self.backend, self.prompt = robot, backend, prompt
        self.dt = 1.0 / control_hz
        self.control_hz = control_hz
        self.safety = SafetyFilter(*load_joint_limits(), max_step=max_step)
        self.player = ChunkPlayer(blend_steps, execute_ratio)
        self.history = FrameHistory(backend.history_len, backend.history_fps, control_hz, backend.frame_transform)
        self.writer, self.save_on = writer, save_on
        self.min_success_steps = min_success_steps
        self.verbose = verbose
        self.stop = False
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla-infer")
        self.trace = []

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def _request(self, step, obs):
        fut = self.pool.submit(self.backend.infer, obs["state"].copy(), self.history.frames(), self.prompt)
        return dict(fut=fut, step=step, played=self.player.played, t=time.perf_counter())

    def _first_chunk(self, timeout=60.0):
        obs = self.robot.observe()
        self.history.push(0, obs["images"])
        req = self._request(0, obs)
        chunk, hz = req["fut"].result(timeout=timeout)
        lat = time.perf_counter() - req["t"]
        self.player.add_chunk(resample_chunk(chunk, hz, self.control_hz), 0, current=obs["state"])
        self.log(f"[vla] first chunk {chunk.shape} @ {hz:g} Hz in {lat * 1e3:.0f} ms")
        return lat

    def run_episode(self, max_steps=900, episode=0):
        """-> dict(steps, reason, ...). reason in success|max_steps|interrupted|error."""
        self.backend.reset()
        self.history.clear()
        self.player.reset()
        stats = dict(episode=episode, steps=0, chunks=0, starved=0, failures=0, latency_ms=[], reason="max_steps")
        if self.writer is not None:
            self.writer.start_episode()
        try:
            stats["latency_ms"].append(self._first_chunk() * 1e3)
        except Exception as e:
            stats["reason"] = "error"
            self.log(f"[vla] first request failed: {type(e).__name__}: {e}")
            return self._finish(stats)
        # Start from the measured pose: last_command() can be stale (e.g. captured at G1Arms() init while the
        # arms were still settling), and jumping to it would be an unfiltered step.
        prev = np.asarray(self.robot.state(), dtype=np.float64)
        pending, retry_at, consecutive_fail = None, 0, 0
        t_next = time.perf_counter()
        t0 = t_next
        step = 0
        while step < max_steps and not self.stop:
            obs = self.robot.observe()
            self.history.push(step, obs["images"])
            if pending is not None and pending["fut"].done():
                try:
                    chunk, hz = pending["fut"].result()
                except Exception as e:
                    stats["failures"] += 1
                    consecutive_fail += 1
                    self.log(f"[vla] step {step}: request failed: {type(e).__name__}: {e}")
                    if getattr(e, "fatal", False) or consecutive_fail >= 5:
                        stats["reason"] = "error"
                        break
                    retry_at = step + int(self.control_hz * 0.5)
                else:
                    consecutive_fail = 0
                    stats["latency_ms"].append((time.perf_counter() - pending["t"]) * 1e3)
                    skip = self.player.played - pending["played"]
                    if not self.player.add_chunk(resample_chunk(chunk, hz, self.control_hz), skip, current=prev):
                        self.log(f"[vla] step {step}: chunk arrived too late (skip {skip} >= its length), dropped")
                pending = None
            if pending is None and step >= retry_at and self.player.needs_request():
                pending = self._request(step, obs)

            target, starved = self.player.next()
            stats["starved"] += starved
            cmd = self.safety(prev if target is None else target, prev)
            self.robot.command(cmd)
            if self.writer is not None:
                try:
                    self.writer.add_frame(obs["state"], cmd.astype(np.float32), obs["images"])
                except ValueError as e:
                    self.log(f"[record] frame skipped: {e}")
            self.trace.append(dict(episode=episode, step=step, t=time.perf_counter() - t0, wall=time.time(),
                                   state=obs["state"].copy(),
                                   cmd=cmd.copy(), target=(prev if target is None else np.asarray(target)).copy(),
                                   chunk=self.player.chunks, starved=starved))
            prev = cmd
            step += 1
            if step >= self.min_success_steps and self.robot.success():
                stats["reason"] = "success"
                break
            if self.verbose and step % int(self.control_hz) == 0:
                lat = stats["latency_ms"][-1] if stats["latency_ms"] else float("nan")
                self.log(f"[vla] ep {episode} step {step:4d}  chunks {self.player.chunks:3d}  buffer "
                         f"{self.player.remaining():3d}  latency {lat:6.0f} ms  starved {stats['starved']}")
            t_next += self.dt
            sleep = t_next - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.5:  # fell far behind (e.g. a stall): don't try to catch up
                t_next = time.perf_counter()
        if self.stop and stats["reason"] == "max_steps":
            stats["reason"] = "interrupted"
        if pending is not None:
            pending["fut"].cancel()
        stats["steps"] = step
        stats["chunks"] = self.player.chunks
        stats["hz"] = step / max(1e-9, time.perf_counter() - t0)
        return self._finish(stats)

    def _finish(self, stats):
        if self.writer is not None:
            keep = stats["reason"] == "success" or (self.save_on == "all" and stats["reason"] == "max_steps")
            if keep and getattr(self.writer, "num_frames", 1):
                idx = self.writer.save_episode()
                self.log(f"[record] saved episode {idx}")
            else:
                self.writer.discard_episode()
                self.log(f"[record] discarded episode ({stats['reason']})")
        lat = np.asarray(stats["latency_ms"]) if stats["latency_ms"] else np.zeros(1)
        stats["latency_ms_mean"] = float(lat.mean())
        stats["latency_ms_max"] = float(lat.max())
        stats.update(clipped=self.safety.clipped, rate_limited=self.safety.rate_limited, rejected=self.safety.rejected)
        self.log(f"[vla] episode {stats['episode']} ended: {stats['reason']} after {stats['steps']} steps, "
                 f"{stats['chunks']} chunks, latency mean {stats['latency_ms_mean']:.0f} / max "
                 f"{stats['latency_ms_max']:.0f} ms, starved {stats['starved']}, failures {stats['failures']}")
        return stats

    def save_trace(self, path):
        if not self.trace:
            return
        arr = {k: np.array([r[k] for r in self.trace]) for k in self.trace[0]}
        np.savez(path, **arr)
        self.log(f"[vla] trace -> {path}")

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.backend.close()


def wait_sim_reset(robot, timeout=3.0):
    """After reset_sim(): wait until the sim reports success == False again (stale flag otherwise)."""
    robot.reset_sim()
    t_end = time.time() + timeout
    time.sleep(0.3)
    while time.time() < t_end and robot.success():
        time.sleep(0.05)


def make_backend(args):
    keys = json.loads(args.keys) if args.keys else None
    if args.backend == "dream":
        key = os.environ.get("DREAM_API_KEY")
        host = urllib.parse.urlsplit(args.url if args.url and "://" in args.url else "https://" + (args.url or "")).hostname
        if not key:
            if host in ("localhost", "127.0.0.1", "::1"):
                key = "local-mock-key"
            else:
                sys.exit("DREAM_API_KEY is not set (export DREAM_API_KEY=dl_live_...)")
        return DreamBackend(args.url, api_key=key, checkpoint=args.checkpoint, embodiment=args.embodiment,
                            keys=keys, action_hz=args.action_hz, timeout=args.timeout, gripper=args.gripper)
    if args.backend == "openpi":
        return OpenPIBackend(args.url, keys=keys, image_size=224 if args.image_size is None else args.image_size,
                             action_hz=args.action_hz, timeout=args.timeout,
                             api_key=args.api_token or os.environ.get("OPENPI_API_KEY"))
    return Gr00tBackend(args.url, keys=keys, image_size=args.image_size or 0, action_hz=args.action_hz,
                        timeout=args.timeout, api_token=args.api_token or os.environ.get("GR00T_API_TOKEN"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    p.add_argument("--url", default=None, help="server URL (dream: https://api.dreamlabs.ai, openpi: ws://host:8000, "
                                               "gr00t: tcp://host:5555)")
    p.add_argument("--prompt", default=DEFAULT_TASK)
    p.add_argument("--target", default="sim", choices=["sim", "real"])
    p.add_argument("--domain", type=int, default=None, help="DDS domain override (isolated sim: --domain 3)")
    p.add_argument("--image-port", type=int, default=60000, help="teleimager request port (sim --port-offset 1000 -> 61000)")
    p.add_argument("--image-host", default=None)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=900, help="per episode, at 30 Hz (900 = 30 s)")
    p.add_argument("--max-step-rad", type=float, default=0.05, help="max joint change per 30 Hz tick")
    p.add_argument("--blend-steps", type=int, default=5, help="ticks of linear blending at chunk seams")
    p.add_argument("--execute-ratio", type=float, default=30 / 48,
                   help="request the next chunk after this fraction of the current one has played")
    p.add_argument("--action-hz", type=float, default=None,
                   help="chunk rate if the server does not say (default: dream 20, openpi/gr00t 30)")
    p.add_argument("--timeout", type=float, default=15.0, help="request timeout [s]")
    p.add_argument("--keys", default=None, help="JSON overrides for the backend's wire key names")
    p.add_argument("--image-size", type=int, default=None,
                   help="openpi: square pad-resize (default 224, 0 = native); gr00t: square resize (default native)")
    p.add_argument("--checkpoint", default="dreamlabs/dream-1-beta", help="dream: model / fine-tune id")
    p.add_argument("--embodiment", default=None, help="dream: optional embodiment string (e.g. other_embodiment)")
    p.add_argument("--gripper", type=float, default=0.0, help="dream: constant state.*_gripper value")
    p.add_argument("--api-token", default=None, help="gr00t api_token / openpi Api-Key (or GR00T_API_TOKEN / OPENPI_API_KEY)")
    p.add_argument("--record", metavar="DATASET_ROOT", default=None)
    p.add_argument("--repo-id", default=None)
    p.add_argument("--save-on", choices=["all", "success"], default="all",
                   help="which finished episodes to save (interrupted/error episodes are always discarded)")
    p.add_argument("--trace", default=None, help="save a per-step .npz trace (state, cmd, target, chunk index)")
    p.add_argument("--no-home", action="store_true", help="don't go_home() at the end of each episode")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.record and not args.repo_id:
        sys.exit("--record needs --repo-id")
    if args.target == "real":
        print(f"About to run a REMOTE POLICY ({args.backend}) on the REAL G1 arms, up to {args.episodes} x "
              f"{args.max_steps} steps, max {args.max_step_rad} rad/tick. Keep the e-stop in reach.")
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            sys.exit("aborted")

    backend = make_backend(args)
    backend.connect()

    from g1lab.robot_io import G1Arms
    robot = G1Arms(args.target, domain=args.domain, image_port=args.image_port, image_host=args.image_host)
    t_end = time.time() + 5.0
    while time.time() < t_end and len(robot.images()) < len(CAMERAS):
        time.sleep(0.1)
    print(f"[vla] robot ready ({args.target}), cameras: {sorted(robot.images())}", flush=True)

    writer = open_writer(args.record, args.repo_id, args.prompt) if args.record else None
    runner = Runner(robot, backend, prompt=args.prompt, max_step=args.max_step_rad, blend_steps=args.blend_steps,
                    execute_ratio=args.execute_ratio, writer=writer, save_on=args.save_on)

    def on_sigint(sig, frame):
        if runner.stop:
            raise KeyboardInterrupt
        print("\n[vla] Ctrl+C: stopping after this tick (again to force)", flush=True)
        runner.stop = True

    signal.signal(signal.SIGINT, on_sigint)
    results = []
    try:
        for ep in range(args.episodes):
            if robot.sim:
                wait_sim_reset(robot)
            results.append(runner.run_episode(args.max_steps, episode=ep))
            if not args.no_home:
                print("[vla] go_home()", flush=True)
                robot.go_home()
            if runner.stop or results[-1]["reason"] == "error":
                break
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        if args.trace:
            runner.save_trace(args.trace)
        runner.close()
        if writer is not None:
            writer.close()
    ok = sum(r["reason"] == "success" for r in results)
    print(f"[vla] done: {ok}/{len(results)} successful episodes", flush=True)
    sys.stdout.flush()
    os._exit(0)  # DDS / teleimager threads would otherwise keep the process alive


if __name__ == "__main__":
    main()
