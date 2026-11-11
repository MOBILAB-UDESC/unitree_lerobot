"""Local mock VLA servers for testing g1lab.remote_vla without keys or GPUs.

    python -m g1lab.vla_mocks dream  --port 8642   # HTTP JSON, POST /v1/get_action + /v1/end_session
    python -m g1lab.vla_mocks openpi --port 8000   # websocket, msgpack-numpy, metadata on connect
    python -m g1lab.vla_mocks gr00t  --port 5555   # ZMQ REP, msgpack (np.save arrays), ping/get_action/reset

Each server validates the request shape against the schema remote_vla sends (and the public protocol it
mimics), prints a summary of what it received, and answers with a smooth action chunk:
  --mode sine   anchor (first state of the session) + a slow sinusoid, phase = wall-clock time, so
                consecutive chunks continue the same trajectory (seams should be invisible)
  --mode reach  from the received state toward a fixed "reach" pose
Invalid requests get the protocol's error reply (Dream DA-4000 envelope, openpi error string,
GR00T {"error": ...}).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from g1lab.remote_vla import CAMERAS, DREAM_FIELDS, DREAM_OBS, GR00T_KEYS, OPENPI_KEYS, packb, unpackb

# Per-joint sinusoid amplitude pattern (rad): shoulder pitch / roll / elbow / wrist pitch, mirrored.
SINE_AMP = np.array([-0.25, 0.12, 0.0, 0.25, 0.0, 0.15, 0.0,
                     -0.25, -0.12, 0.0, 0.25, 0.0, 0.15, 0.0])
SINE_HZ = 0.25
REACH_POSE = np.array([-0.6, 0.25, 0.0, 0.9, 0.0, 0.2, 0.0,
                       -0.6, -0.25, 0.0, 0.9, 0.0, 0.2, 0.0])


class ChunkGen:
    def __init__(self, mode="sine", horizon=48, hz=20.0, amp_scale=1.0):
        self.mode, self.horizon, self.hz, self.amp = mode, int(horizon), float(hz), float(amp_scale)
        self.anchor, self.t0 = None, None

    def reset(self):
        self.anchor, self.t0 = None, None

    def __call__(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(14)
        if self.anchor is None:
            self.anchor, self.t0 = state.copy(), time.time()
        k = np.arange(self.horizon)[:, None]
        if self.mode == "sine":
            t = time.time() - self.t0 + k / self.hz
            # 1 - cos so the trajectory starts at the anchor with zero velocity
            return self.anchor + self.amp * SINE_AMP * (1 - np.cos(2 * np.pi * SINE_HZ * t)) / 2
        s = np.minimum(1.0, (k + 1) / (self.horizon * 2.0))  # cover half the way per chunk
        s = s * s * (3 - 2 * s)
        return state + (REACH_POSE - state) * s


def _summ(v):
    if isinstance(v, np.ndarray):
        return f"ndarray{list(v.shape)} {v.dtype}"
    if isinstance(v, list):
        if v and isinstance(v[0], str) and len(v[0]) > 200:
            return f"list[{len(v)}] of b64 str (~{len(v[0]) // 1024} KiB)"
        return f"list[{len(v)}]" + (f" {np.round(v, 3).tolist()}" if v and isinstance(v[0], (int, float)) else
                                    (f" {v}" if len(str(v)) < 80 else ""))
    if isinstance(v, str):
        return repr(v if len(v) < 80 else v[:77] + "...")
    return repr(v)


class Mock:
    """Shared bookkeeping: request log + printing."""

    def __init__(self, name, gen, quiet=False):
        self.name, self.gen, self.quiet = name, gen, quiet
        self.requests = []  # dicts: {"ok", "errors", "summary", ...}
        self.lock = threading.Lock()
        self._seen = set()  # endpoints already printed in full

    def record(self, entry):
        with self.lock:
            entry["n"] = len(self.requests)
            entry["t"] = time.time()
            self.requests.append(entry)
        if not self.quiet:
            first = entry.get("endpoint") not in self._seen
            self._seen.add(entry.get("endpoint"))
            if first or not entry["ok"]:
                print(f"[mock-{self.name}] request #{entry['n']} ({'OK' if entry['ok'] else 'INVALID'}):", flush=True)
                for k, v in entry["summary"].items():
                    print(f"    {k}: {v}", flush=True)
                for e in entry["errors"]:
                    print(f"    ERROR: {e}", flush=True)
            else:
                print(f"[mock-{self.name}] request #{entry['n']} OK  {entry.get('brief', '')}", flush=True)


# ----------------------------------------------------------------------------------------- dream
def validate_dream(payload, auth, session_ids, fields=DREAM_FIELDS):
    F, errors = fields, []
    if not (auth or "").startswith("Bearer ") or len(auth) <= 7:
        errors.append("missing 'Authorization: Bearer <key>' header")
    if not isinstance(payload, dict):
        return ["body is not a JSON object"], {}
    if not isinstance(payload.get(F["checkpoint"]), str) or not payload.get(F["checkpoint"]):
        errors.append(f"'{F['checkpoint']}' must be a non-empty string")
    sid = payload.get(F["session_id"], "MISSING")
    if sid == "MISSING":
        errors.append(f"'{F['session_id']}' missing (null on the first call)")
    elif sid is not None and sid not in session_ids:
        errors.append(f"unknown {F['session_id']} {sid!r}")
    obs = payload.get(F["observation"]) if F.get("observation") else payload
    if not isinstance(obs, dict):
        return errors + [f"'{F['observation']}' must be an object"], {}
    if not isinstance(obs.get(F["prompt"]), str) or not obs.get(F["prompt"]):
        errors.append(f"observation '{F['prompt']}' must be a non-empty string")
    for k, n in ((F["state_left_arm"], 7), (F["state_right_arm"], 7), (F["state_left_gripper"], 1),
                 (F["state_right_gripper"], 1)):
        v = obs.get(k)
        if not (isinstance(v, list) and len(v) == n and all(isinstance(x, (int, float)) for x in v)):
            errors.append(f"observation '{k}' must be a list of {n} floats, got {_summ(v)}")
    import cv2
    for cam, key in F["cameras"].items():
        frames = obs.get(key)
        if not (isinstance(frames, list) and 1 <= len(frames) <= DREAM_OBS["frames"]):
            errors.append(f"observation '{key}' must be a list of 1..{DREAM_OBS['frames']} frames, got {_summ(frames)}")
            continue
        for i, f in enumerate(frames):
            try:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(f, validate=True), np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                img = None
            if img is None:
                errors.append(f"'{key}'[{i}] is not a base64 JPEG")
                break
            if img.shape != (DREAM_OBS["height"], DREAM_OBS["width"], 3):
                errors.append(f"'{key}'[{i}] is {img.shape[1]}x{img.shape[0]}, expected "
                              f"{DREAM_OBS['width']}x{DREAM_OBS['height']}")
                break
    summary = {k: _summ(v) for k, v in payload.items() if k != F.get("observation")}
    summary.update({f"{F['observation']}.{k}": _summ(v) for k, v in obs.items()} if F.get("observation") else {})
    return errors, summary


def make_dream_server(port=8642, mode="sine", horizon=48, hz=20.0, quiet=False, host="127.0.0.1"):
    F = DREAM_FIELDS
    mock = Mock("dream", None, quiet)
    sessions = {}  # id -> ChunkGen

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _reply(self, code, body):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):
            t_in = time.perf_counter()
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = None
            if self.path == F["path_end_session"]:
                sid = (payload or {}).get(F["session_id"])
                existed = sessions.pop(sid, None) is not None
                if not quiet:
                    print(f"[mock-dream] end_session {sid} ({'released' if existed else 'unknown'})", flush=True)
                mock.record(dict(ok=existed, errors=[] if existed else ["unknown session"], endpoint="end_session",
                                 summary={"session_id": sid}))
                return self._reply(200, {"status": {"code": "DA-0000", "reason": "ok"}})
            if self.path != F["path_get_action"]:
                return self._reply(404, {"status": {"code": "DA-4000", "reason": "not_found", "message": self.path}})
            errors, summary = validate_dream(payload, self.headers.get("Authorization"), sessions)
            entry = dict(ok=not errors, errors=errors, summary=summary, endpoint="get_action", bytes=len(raw),
                         session_id=(payload or {}).get(F["session_id"]) if isinstance(payload, dict) else None)
            if errors:
                mock.record(entry)
                return self._reply(400, {"status": {"code": "DA-4000", "reason": "bad_request", "retryable": False,
                                                    "message": "; ".join(errors)}})
            obs = payload[F["observation"]]
            sid = payload[F["session_id"]]
            new = sid is None
            if new:
                sid = "mock-" + uuid.uuid4().hex[:12]
                sessions[sid] = ChunkGen(mode, horizon, hz)
            state = np.array(obs[F["state_left_arm"]] + obs[F["state_right_arm"]])
            q = sessions[sid](state)
            entry.update(gen_t0=sessions[sid].t0, anchor=sessions[sid].anchor.tolist())
            grip = np.zeros((len(q), 1))
            ql, qr = np.hstack([q[:, :7], grip]), np.hstack([q[:, 7:], grip])
            entry["frames"] = {k: len(obs[k]) for k in F["cameras"].values()}
            entry["state"] = state.tolist()
            entry["brief"] = f"session {sid} frames {list(entry['frames'].values())} state[0]={state[0]:+.3f}"
            mock.record(entry)
            self._reply(200, {
                "status": {"code": "DA-0001" if new else "DA-0000", "reason": "new" if new else "ok"},
                "session_id": sid,
                "actions": {"q_l_chunk": ql.tolist(), "q_r_chunk": qr.tolist(), "q_l": ql[0, :7].tolist(),
                            "q_r": qr[0, :7].tolist(), "per_key": {}},
                "timing": {"serve_infer_ms": 0.0, "gateway_ms": (time.perf_counter() - t_in) * 1e3},
            })

    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    srv.mock = mock
    srv.stop = srv.shutdown
    srv.serve = srv.serve_forever
    return srv


# ---------------------------------------------------------------------------------------- openpi
def validate_openpi(obs, keys=OPENPI_KEYS):
    errors = []
    if not isinstance(obs, dict):
        return ["observation is not a dict"], {}
    for cam in CAMERAS:
        v = obs.get(keys[cam])
        if not (isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 3 and v.shape[2] == 3):
            errors.append(f"'{keys[cam]}' must be uint8 [H, W, 3], got {_summ(v)}")
    s = obs.get(keys["state"])
    if not (isinstance(s, np.ndarray) and s.shape == (14,) and np.all(np.isfinite(s))):
        errors.append(f"'{keys['state']}' must be a finite (14,) array, got {_summ(s)}")
    if not isinstance(obs.get(keys["prompt"]), str):
        errors.append(f"'{keys['prompt']}' must be a string")
    return errors, {k: _summ(v) for k, v in obs.items()}


def make_openpi_server(port=8000, mode="sine", horizon=50, hz=30.0, quiet=False, host="127.0.0.1"):
    from websockets.sync.server import serve
    mock = Mock("openpi", None, quiet)
    metadata = {"mock": True, "policy": "g1lab.vla_mocks", "action_horizon": horizon, "action_dim": 14}

    def handler(ws):
        gen = ChunkGen(mode, horizon, hz)  # one session per connection
        ws.send(packb(metadata, "openpi"))
        for msg in ws:
            try:
                obs = unpackb(msg)
            except Exception as e:
                obs = None
                errors, summary = [f"msgpack decode failed: {e}"], {}
            else:
                errors, summary = validate_openpi(obs)
            entry = dict(ok=not errors, errors=errors, summary=summary, endpoint="infer", bytes=len(msg))
            if errors:
                mock.record(entry)
                ws.send("Traceback (mock): invalid observation\n" + "\n".join(errors))  # openpi sends str on error
                continue
            state = obs[OPENPI_KEYS["state"]]
            entry["state"] = state.tolist()
            entry["brief"] = f"state[0]={state[0]:+.3f} images {[obs[OPENPI_KEYS[c]].shape for c in CAMERAS]}"
            a = gen(state).astype(np.float32)
            entry.update(gen_t0=gen.t0, anchor=gen.anchor.tolist())
            mock.record(entry)
            ws.send(packb({"actions": a, "policy_timing": {"infer_ms": 0.0}}, "openpi"))

    srv = serve(handler, host, port, compression=None, max_size=None)
    srv.mock = mock
    srv.stop = srv.shutdown
    srv.serve = srv.serve_forever
    return srv


# ----------------------------------------------------------------------------------------- gr00t
def validate_gr00t(obs, keys=GR00T_KEYS):
    errors = []
    if not isinstance(obs, dict):
        return ["data is not a dict"], {}
    for cam in CAMERAS:
        v = obs.get(keys[cam])
        if not (isinstance(v, np.ndarray) and v.dtype == np.uint8 and v.ndim == 4 and v.shape[0] == 1
                and v.shape[3] == 3):
            errors.append(f"'{keys[cam]}' must be uint8 [1, H, W, 3], got {_summ(v)}")
    for k in ("state_left_arm", "state_right_arm"):
        v = obs.get(keys[k])
        if not (isinstance(v, np.ndarray) and v.shape == (1, 7) and np.all(np.isfinite(v))):
            errors.append(f"'{keys[k]}' must be a finite [1, 7] array, got {_summ(v)}")
    p = obs.get(keys["prompt"])
    if not (isinstance(p, list) and len(p) == 1 and isinstance(p[0], str)):
        errors.append(f"'{keys['prompt']}' must be a 1-element list of str, got {_summ(p)}")
    return errors, {k: _summ(v) for k, v in obs.items()}


class Gr00tServer:
    def __init__(self, port=5555, mode="sine", horizon=16, hz=30.0, quiet=False, host="127.0.0.1", api_token=None):
        import zmq
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, 200)
        self.sock.bind(f"tcp://{host}:{port}")
        self.mock = Mock("gr00t", None, quiet)
        self.gen = ChunkGen(mode, horizon, hz)
        self.api_token, self.quiet = api_token, quiet
        self._run = True

    def _handle(self, req):
        if not isinstance(req, dict):
            return {"error": "request must be a dict"}
        if self.api_token and req.get("api_token") != self.api_token:
            return {"error": "Unauthorized: invalid api_token"}
        ep = req.get("endpoint", "get_action")
        if ep == "ping":
            return {"status": "ok", "message": "Server is running (mock)"}
        if ep == "reset":
            self.gen.reset()
            self.mock.record(dict(ok=True, errors=[], summary={"endpoint": "reset"}, endpoint="reset", brief="reset"))
            return {"status": "ok"}
        if ep != "get_action":
            return {"error": f"Unknown endpoint: {ep}"}
        obs = req.get("data")
        errors, summary = validate_gr00t(obs)
        entry = dict(ok=not errors, errors=errors, summary=summary, endpoint="get_action")
        if errors:
            self.mock.record(entry)
            return {"error": "; ".join(errors)}
        state = np.concatenate([obs[GR00T_KEYS["state_left_arm"]][0], obs[GR00T_KEYS["state_right_arm"]][0]])
        entry["state"] = state.tolist()
        entry["brief"] = f"state[0]={state[0]:+.3f} video {[obs[GR00T_KEYS[c]].shape for c in CAMERAS]}"
        a = self.gen(state).astype(np.float32)
        entry.update(gen_t0=self.gen.t0, anchor=self.gen.anchor.tolist())
        self.mock.record(entry)
        return {"action.left_arm": a[:, :7], "action.right_arm": a[:, 7:]}

    def serve(self):
        import zmq
        while self._run:
            try:
                msg = self.sock.recv()
            except zmq.Again:
                continue
            try:
                reply = self._handle(unpackb(msg))
            except Exception as e:
                reply = {"error": f"{type(e).__name__}: {e}"}
            self.sock.send(packb(reply, "gr00t"))
        self.sock.close()
        self.ctx.term()

    def stop(self):
        self._run = False


def make_gr00t_server(port=5555, mode="sine", horizon=16, hz=30.0, quiet=False, host="127.0.0.1", api_token=None):
    return Gr00tServer(port, mode, horizon, hz, quiet, host, api_token)


FACTORIES = {"dream": (make_dream_server, 8642, 48, 20.0),
             "openpi": (make_openpi_server, 8000, 50, 30.0),
             "gr00t": (make_gr00t_server, 5555, 16, 30.0)}


def start_in_thread(kind, port, **kw):
    """Start a mock in a daemon thread; returns the server (server.mock.requests, server.stop())."""
    make, _, horizon, hz = FACTORIES[kind]
    kw.setdefault("horizon", horizon)
    kw.setdefault("hz", hz)
    srv = make(port, **kw)
    threading.Thread(target=srv.serve, daemon=True, name=f"mock-{kind}").start()
    return srv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("protocol", choices=sorted(FACTORIES))
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--mode", choices=["sine", "reach"], default="sine")
    p.add_argument("--horizon", type=int, default=None, help="chunk rows (dream 48, openpi 50, gr00t 16)")
    p.add_argument("--hz", type=float, default=None, help="chunk rate (dream 20, openpi/gr00t 30)")
    p.add_argument("--api-token", default=None, help="gr00t: require this api_token")
    p.add_argument("--log-jsonl", default=None, help="append one JSON line per request (for tests)")
    a = p.parse_args(argv)
    make, port, horizon, hz = FACTORIES[a.protocol]
    kw = dict(mode=a.mode, horizon=a.horizon or horizon, hz=a.hz or hz, host=a.host)
    if a.protocol == "gr00t":
        kw["api_token"] = a.api_token
    srv = make(a.port or port, **kw)
    if a.log_jsonl:
        orig = srv.mock.record

        def record(entry, _orig=orig, _path=a.log_jsonl):
            _orig(entry)
            with open(_path, "a") as f:
                f.write(json.dumps({k: v for k, v in entry.items() if k != "summary"} | {"summary": entry["summary"]},
                                   default=str) + "\n")
        srv.mock.record = record
    print(f"[mock-{a.protocol}] listening on port {a.port or port} ({a.mode}, {kw['horizon']} rows @ {kw['hz']:g} Hz)",
          flush=True)
    try:
        srv.serve()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            srv.stop()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
