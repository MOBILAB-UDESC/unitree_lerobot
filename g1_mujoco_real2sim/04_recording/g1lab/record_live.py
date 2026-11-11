"""Record LeRobot episodes at 30 Hz next to ANY controller (xr_teleoperate, MCP, VLA, scripted, ACT).

A passive observer: it only subscribes (DDS rt/lowstate, rt/lowcmd, rt/arm_sdk, rt/rewards_state,
rt/reset_pose/cmd + the teleimager image streams) and never publishes, so it cannot move the robot and
can run beside whatever is driving it, in the sim or on the real G1.

    # terminal 1: sim (or the real robot), terminal 2: the controller, terminal 3:
    python -m g1lab.record_live --root datasets/teleop_waist --task "put the wrapped box inside the cardboard box"
    #   keys:  s = start / stop+save   d = discard the current episode   q = quit (an unsaved episode is dropped)

    # unattended (sim): an episode starts on every rt/reset_pose/cmd (+ --start-delay) and is saved once the
    # sim's success flag (rt/rewards_state) has held for --success-hold s; attempts that never succeed are dropped
    python -m g1lab.record_live --root /tmp/rec --auto-start --start-delay 2 --auto-success --only-success \
        --domain 7 --image-port 61000      # sim started with --domain 7 --port-offset 1000

Why not xr_teleoperate's own --record: with --input-mode controller --motion the operator twists the waist
with the right joystick (LocoClient.Move vyaw; the sim emulates Unitree's motion controller), but Unitree's
recorder only stores the arm (and hand) joints - with --ee none its "body" fields stay empty, and with the
gripper/controller modes the body action is just the joystick velocities - so those episodes silently lose
the torso rotation and a policy trained on them can't reproduce it. This recorder stores 15 joints (14 arms + kWaistYaw, robot_type
Unitree_G1_ArmsWaist_3Cam) by default; --no-waist writes the 14-joint Unitree_G1_ArmsOnly_3Cam format.

What is recorded each 1/30 s tick:
  observation.state  measured joints from rt/lowstate (motors 15-28, + waist yaw = motor 12).
  action (arms)      the COMMANDED arm targets as they are on the wire: rt/lowcmd if it is live (debug mode:
                     xr_teleoperate without --motion, G1Arms / MCP / scripted / run_act), else rt/arm_sdk
                     (--motion). Live = a message within 0.5 s; rt/lowcmd wins, as on the sim. With no live
                     controller the measured pose is recorded as the action (the robot holds still).
  action (waist)     --waist-action auto (default):
                       rt/lowcmd live -> its motor-12 target (whoever drives lowcmd also drives the waist);
                       rt/arm_sdk live -> the waist MEASURED ONE TICK LATER. With --motion the motion
                         controller (the sim's emulation / Unitree's on the robot) owns the waist and turns it
                         from the joystick; arm_sdk's motor-12 slot only carries the pose xr_teleoperate latched
                         at start-up, so it is not the operator's intent. The next measured waist is the
                         target the motion controller actually reached, i.e. the best available "command",
                         and replaying it through G1Arms.command_waist() reproduces the motion.
                       none   -> measured.
                     Override with command | next | measured.
  images             the three 480x640 RGB streams (cam_head, cam_left_wrist, cam_right_wrist); a stereo head
                     stream keeps its left eye, like the real dataset. Frames are buffered as the JPEG bytes
                     the image server sent and decoded/written by a writer thread, so the 30 Hz sampling never
                     stalls while an episode's videos are encoded (save takes ~15-20 s per episode).

Note on the real robot: xr_teleoperate velocity-clips the arm target before publishing, and this records the
clipped value (what the robot was told), whereas its --record stores the unclipped IK output.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import select
import signal
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from g1lab.robot_io import CAMERA_KEYS, TARGETS, WAIST_YAW_INDEX, init_dds  # noqa: E402

DEFAULT_TASK = "put the wrapped box inside the cardboard box"
CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
IMAGE_HW = (480, 640)
NUM_MOTORS = 29
ARMS = slice(15, 29)
ARM_SDK_WEIGHT = 29        # xr_teleoperate puts the arm_sdk blend weight in motor_cmd[29].q
CMD_TIMEOUT = 0.5          # s; same freshness rule the sim uses to pick its command source
WAIST_MODES = ("auto", "command", "next", "measured")


# ------------------------------------------------------------------------------------- pure logic
def active_source(t_lowcmd, t_arm_sdk, now, timeout=CMD_TIMEOUT):
    """'lowcmd' | 'arm_sdk' | None: which command stream is driving the robot (lowcmd wins, like the sim)."""
    if t_lowcmd is not None and now - t_lowcmd < timeout:
        return "lowcmd"
    if t_arm_sdk is not None and now - t_arm_sdk < timeout:
        return "arm_sdk"
    return None


def command_for(source, measured, lowcmd=None, arm_sdk=None, waist_mode="auto"):
    """Arm action (14,) and waist action for one tick from 29-motor q vectors.

    The waist action is a float, or None = "use the NEXT tick's measured waist" (see the module docstring).
    """
    measured = np.asarray(measured, dtype=np.float64)
    if waist_mode not in WAIST_MODES:
        raise ValueError(f"waist_mode must be one of {WAIST_MODES}")
    if source == "lowcmd":
        cmd = np.asarray(lowcmd, dtype=np.float64)
    elif source == "arm_sdk":
        cmd = np.asarray(arm_sdk, dtype=np.float64)
    else:
        cmd = measured
    arms = cmd[ARMS].copy()
    if waist_mode == "measured":
        waist = float(measured[WAIST_YAW_INDEX])
    elif waist_mode == "next":
        waist = None
    elif waist_mode == "command":
        waist = float(cmd[WAIST_YAW_INDEX])
    else:  # auto
        waist = None if source == "arm_sdk" else float(cmd[WAIST_YAW_INDEX])
    return arms, waist


def state_from(measured, waist=True):
    m = np.asarray(measured, dtype=np.float64)
    s = m[ARMS]
    return (np.append(s, m[WAIST_YAW_INDEX]) if waist else s).astype(np.float32)


class LookaheadAssembler:
    """Holds each frame back one tick, so a waist action of None can become the next tick's measured waist."""

    def __init__(self, waist=True):
        self.waist = waist
        self.pending = None

    def clear(self):
        self.pending = None

    def _finish(self, frame, next_waist):
        state, arms, waist, payload = frame
        if not self.waist:
            return state, arms.astype(np.float32), payload
        w = (next_waist if waist is None else waist)
        return state, np.append(arms, w).astype(np.float32), payload

    def push(self, state, arms, waist, payload):
        """Add a tick; returns the completed previous frame (state, action, payload) or None."""
        out = None
        if self.pending is not None:
            out = self._finish(self.pending, float(state[-1]) if self.waist else None)
        self.pending = (np.asarray(state, dtype=np.float32), np.asarray(arms, dtype=np.float64), waist, payload)
        return out

    def flush(self):
        """The last frame of an episode: no next tick, so its own measured waist is used."""
        if self.pending is None:
            return None
        out = self._finish(self.pending, float(self.pending[0][-1]) if self.waist else None)
        self.pending = None
        return out


def decode_jpgs(jpgs, binocular=(), size=IMAGE_HW):
    """{cam: jpeg bytes} -> {cam: RGB uint8 (480, 640, 3)}; stereo streams keep the left eye."""
    import cv2
    out = {}
    for cam, data in jpgs.items():
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"{cam}: undecodable JPEG")
        if cam in binocular:
            bgr = bgr[:, : bgr.shape[1] // 2]
        if bgr.shape[:2] != tuple(size):
            bgr = cv2.resize(bgr, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        out[cam] = np.ascontiguousarray(bgr[:, :, ::-1])
    return out


class EpisodeSink:
    """Serialises everything that touches the (not thread-safe) LeRobotWriter onto one worker thread.

    Events are applied in order, so frames of the next episode simply queue up while the previous episode's
    videos are being encoded. threaded=False applies events immediately (tests).
    """

    def __init__(self, writer, decode=decode_jpgs, binocular=(), threaded=True, log=print):
        self.writer, self.decode, self.binocular, self.log = writer, decode, tuple(binocular), log
        self.q = queue.Queue()
        self.saved, self.discarded, self.frame_errors = [], 0, 0
        self.busy = None   # what the worker is doing right now (for the status line)
        self.error = None
        self.threaded = threaded
        if threaded:
            self.thread = threading.Thread(target=self._run, name="record-writer", daemon=True)
            self.thread.start()

    def backlog(self):
        return self.q.qsize()

    def _put(self, *ev):
        if self.threaded:
            self.q.put(ev)
        else:
            self._apply(ev)

    def start(self, task=None):
        self._put("start", task)

    def frame(self, state, action, jpgs):
        self._put("frame", state, action, jpgs)

    def save(self, note=""):
        self._put("save", note)

    def discard(self, note=""):
        self._put("discard", note)

    def close(self):
        """Apply everything queued, finalize the dataset, and wait for the worker."""
        self._put("close")
        if self.threaded:
            self.thread.join()

    def _apply(self, ev):
        kind = ev[0]
        if kind == "start":
            self.writer.start_episode(ev[1])
        elif kind == "frame":
            try:
                self.writer.add_frame(ev[1], ev[2], self.decode(ev[3], self.binocular))
            except ValueError as e:
                self.frame_errors += 1
                if self.frame_errors <= 5:
                    self.log(f"[record] frame skipped: {e}")
        elif kind == "save":
            self.busy = "saving"
            idx = self.writer.save_episode()
            self.busy = None
            if idx is not None:
                self.saved.append(idx)
            self.log(f"[record] saved episode {idx} {ev[1]}".rstrip())
        elif kind == "discard":
            self.writer.discard_episode()
            self.discarded += 1
            self.log(f"[record] discarded episode {ev[1]}".rstrip())
        elif kind == "close":
            self.writer.close()
            return False
        return True

    def _run(self):
        while True:
            ev = self.q.get()
            try:
                if not self._apply(ev):
                    return
            except Exception as e:  # keep draining so close() can't hang; report once
                self.error = e
                self.log(f"[record] writer error on {ev[0]}: {type(e).__name__}: {e}")


class Session:
    """Episode state machine: keys, auto-start on sim resets, auto-stop on success. Time is passed in."""

    def __init__(self, sink, task=DEFAULT_TASK, waist=True, auto_success=False, only_success=False,
                 success_hold=1.0, auto_start=False, start_delay=0.0, min_frames=15, episode_seconds=0.0,
                 log=print):
        self.sink, self.task, self.log = sink, task, log
        self.asm = LookaheadAssembler(waist)
        self.auto_success, self.only_success, self.success_hold = auto_success, only_success, success_hold
        self.auto_start, self.start_delay, self.min_frames = auto_start, start_delay, min_frames
        self.episode_seconds = episode_seconds
        self.started_at = None
        self.recording = False
        self.frames = 0
        self.episodes = 0          # started
        self.success_seen = False  # success held for success_hold s during this episode
        self.success_at = None     # start of the current success streak (None while not successful)
        self.start_at = None       # scheduled auto-start time
        self.quit = False
        self._prev_success = None  # unknown until the first tick: a flag already up then is not a new success

    # ---------------------------------------------------------------- control
    def start(self):
        self.asm.clear()
        self.sink.start(self.task)
        self.recording, self.frames, self.success_seen, self.success_at = True, 0, False, None
        self.started_at = None
        self.episodes += 1
        self.log(f"[record] episode {self.episodes} started")

    def stop(self, reason="stop"):
        """End the episode: save it, unless it is too short or (--only-success) never succeeded."""
        if not self.recording:
            return None
        last = self.asm.flush()
        if last is not None:
            self.sink.frame(*last)
            self.frames += 1
        self.recording = False
        self.success_at = None
        note = f"({reason}, {self.frames} frames{', success' if self.success_seen else ''})"
        if self.frames < self.min_frames:
            self.sink.discard(note + " - too short")
            return "discarded"
        if self.only_success and not self.success_seen:
            self.sink.discard(note + " - no success")
            return "discarded"
        self.sink.save(note)
        return "saved"

    def discard(self, reason="d"):
        if not self.recording:
            return
        self.asm.clear()
        self.recording = False
        self.success_at = None
        self.sink.discard(f"({reason}, {self.frames} frames)")

    def key(self, ch, now=None):
        if ch == "s":
            if self.recording:
                self.stop("key s")
            else:
                self.start_at = None
                self.start()
        elif ch == "d":
            self.discard("key d")
        elif ch == "q":
            if self.recording:
                self.log("[record] quitting: dropping the unsaved episode (press s before q to keep it)")
                self.discard("quit")
            self.quit = True

    def on_reset(self, now):
        """A sim reset (rt/reset_pose/cmd): ends the running episode (a new attempt begins) and, with
        --auto-start, schedules the next one."""
        if not self.auto_start:
            return
        if self.recording:
            self.stop("reset")
        self.start_at = now + self.start_delay

    # ------------------------------------------------------------------ ticks
    def tick(self, now, state, arms, waist, payload, success=False):
        """One sample. payload = the frame's images (JPEG bytes per camera)."""
        if self.start_at is not None and now >= self.start_at and not self.recording:
            self.start_at = None
            self.start()
        rising = bool(success) and self._prev_success is False
        self._prev_success = bool(success)
        if not self.recording:
            return
        if self.started_at is None:
            self.started_at = now
        done = self.asm.push(state, arms, waist, payload)
        if done is not None:
            self.sink.frame(*done)
            self.frames += 1
        # A success only counts once it has held for success_hold s: the sim's check can flicker true while
        # the box is still moving (and a flag that is already up when the episode starts is stale).
        if rising:
            self.success_at = now
        elif not success:
            self.success_at = None
        if self.success_at is not None and now - self.success_at >= self.success_hold:
            self.success_seen = True
            if self.auto_success:
                self.stop("success")
                return
        if self.episode_seconds and now - self.started_at >= self.episode_seconds:
            self.stop("time limit")


# --------------------------------------------------------------------------------------- live I/O
class LiveObserver:
    """Read-only view of a G1 (sim or real): DDS subscriptions + teleimager streams. Publishes nothing.

    Also offers a G1Arms-like state()/images()/observe()/success() for read-only users (run_act --dry-run).
    """

    def __init__(self, target="sim", domain=None, iface=None, image_host=None, image_port=60000, cameras=True,
                 waist=True):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

        cfg = dict(TARGETS[target])
        if iface:
            cfg["iface"] = iface
        if image_host:
            cfg["image_host"] = image_host
        if domain is not None:
            cfg["domain"] = domain
        self.cfg, self.sim, self.waist = cfg, cfg["simulation"], waist
        init_dds(cfg["domain"], cfg["iface"])
        self.lock = threading.Lock()
        self.measured = None
        self.lowcmd = self.arm_sdk = None
        self.t_lowcmd = self.t_arm_sdk = self.t_state = None
        self.arm_sdk_weight = 0.0
        self.rewards = None
        self.resets = 0
        self._subs = []
        for topic, typ, cb in (("rt/lowstate", LowState_, self._on_state), ("rt/lowcmd", LowCmd_, self._on_lowcmd),
                               ("rt/arm_sdk", LowCmd_, self._on_arm_sdk),
                               ("rt/rewards_state", String_, self._on_rewards),
                               ("rt/reset_pose/cmd", String_, self._on_reset)):
            s = ChannelSubscriber(topic, typ)
            s.Init(cb, 10)
            self._subs.append(s)

        self.img = None
        self.binocular = set()
        self._last_jpg = {}
        if cameras:
            from teleimager.image_client import ImageClient
            self.img = ImageClient(host=cfg["image_host"], request_port=image_port, request_bgr=False)
            self.cam_config = self.img.get_cam_config()
            self._getters = {}
            for cam, topic in CAMERA_KEYS.items():
                c = self.cam_config.get(f"{topic}_camera", {})
                if c.get("enable_zmq"):
                    self._getters[cam] = getattr(self.img, f"get_{topic}_frame")
                    if c.get("binocular"):
                        self.binocular.add(cam)

    # DDS callbacks (cheap: copy q only)
    def _on_state(self, msg):
        q = np.array([m.q for m in msg.motor_state[:NUM_MOTORS]])
        with self.lock:
            self.measured, self.t_state = q, time.monotonic()

    def _on_lowcmd(self, msg):
        q = np.array([m.q for m in msg.motor_cmd[:NUM_MOTORS]])
        with self.lock:
            self.lowcmd, self.t_lowcmd = q, time.monotonic()

    def _on_arm_sdk(self, msg):
        q = np.array([m.q for m in msg.motor_cmd[:NUM_MOTORS]])
        w = float(msg.motor_cmd[ARM_SDK_WEIGHT].q)
        with self.lock:
            self.arm_sdk, self.arm_sdk_weight, self.t_arm_sdk = q, w, time.monotonic()

    def _on_rewards(self, msg):
        try:
            r = json.loads(msg.data).get("rewards")
            self.rewards = float(r[0]) if r else None
        except (ValueError, TypeError, AttributeError, IndexError):
            pass

    def _on_reset(self, msg):
        self.resets += 1

    # sampling
    def sample(self, now=None):
        """Latest joint data: dict(measured, lowcmd, arm_sdk, source, weight) or None before the first lowstate."""
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.measured is None:
                return None
            return dict(measured=self.measured, lowcmd=self.lowcmd, arm_sdk=self.arm_sdk,
                        source=active_source(self.t_lowcmd, self.t_arm_sdk, now), weight=self.arm_sdk_weight,
                        state_age=now - self.t_state)

    def jpgs(self, max_age=0.5):
        """Latest JPEG per camera. teleimager reports "no frame" whenever a stream pauses > 100 ms (common
        while the sim is busy), so the last frame is held for up to max_age s instead of dropping the tick."""
        out = {}
        now = time.monotonic()
        for cam, get in getattr(self, "_getters", {}).items():
            frame = get()
            if frame is not None and frame.jpg:
                self._last_jpg[cam] = (frame.jpg, now)
            held = self._last_jpg.get(cam)
            if held is not None and now - held[1] <= max_age:
                out[cam] = held[0]
        return out

    def success(self):
        return bool(self.rewards is not None and self.rewards >= 1.0)

    def wait_ready(self, timeout=5.0, cameras=CAMERAS):
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            if self.measured is not None and (self.img is None or set(cameras) <= set(self.jpgs())):
                return True
            time.sleep(0.05)
        return False

    # G1Arms-like, read-only
    def state(self):
        s = self.sample()
        if s is None:
            raise RuntimeError("no rt/lowstate yet")
        return state_from(s["measured"], self.waist)

    def images(self):
        return decode_jpgs(self.jpgs(), self.binocular)

    def observe(self):
        return {"state": self.state(), "images": self.images()}

    def close(self):
        if self.img is not None:
            self.img.close()


class KeyReader:
    """Single key presses from the terminal (cbreak mode); inactive when stdin is not a TTY."""

    def __init__(self):
        self.active = sys.stdin.isatty()
        self._old = None
        if self.active:
            import termios
            import tty
            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

    def poll(self):
        if not self.active:
            return ""
        keys = ""
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if not ch:
                break
            keys += ch.lower()
        return keys

    def restore(self):
        if self._old is not None:
            import termios
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
            self._old = None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True, help="LeRobot dataset dir (appended to if it exists)")
    p.add_argument("--repo-id", default=None, help="default: local/<basename of --root>")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--target", default="sim", choices=sorted(TARGETS))
    p.add_argument("--domain", type=int, default=None, help="DDS domain override (sim --domain 7)")
    p.add_argument("--iface", default=None)
    p.add_argument("--image-host", default=None)
    p.add_argument("--image-port", type=int, default=60000, help="teleimager request port (sim --port-offset 1000 -> 61000)")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-waist", action="store_true", help="14 arm joints (Unitree_G1_ArmsOnly_3Cam)")
    p.add_argument("--waist-action", choices=WAIST_MODES, default="auto", help="see the module docstring")
    p.add_argument("--auto-success", action="store_true", help="stop+save when rt/rewards_state reports success (sim)")
    p.add_argument("--success-hold", type=float, default=1.0,
                   help="s the success flag must stay up before it counts (and before --auto-success stops)")
    p.add_argument("--only-success", action="store_true", help="discard episodes that end without success")
    p.add_argument("--auto-start", action="store_true",
                   help="start an episode on every rt/reset_pose/cmd (sim reset); a running one is ended first")
    p.add_argument("--start-delay", type=float, default=0.0, help="s between the reset and the auto-start")
    p.add_argument("--start-now", action="store_true", help="start recording the first episode immediately")
    p.add_argument("--min-frames", type=int, default=15, help="shorter episodes are discarded")
    p.add_argument("--episode-seconds", type=float, default=0.0,
                   help="stop+save an episode after this long (0 = no limit; e.g. fixed-length VLA rollouts)")
    p.add_argument("--max-episodes", type=int, default=0, help="quit after this many saved episodes (0 = no limit)")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    waist = not a.no_waist
    obs = LiveObserver(a.target, domain=a.domain, iface=a.iface, image_host=a.image_host, image_port=a.image_port,
                       waist=waist)
    if not obs.wait_ready(10.0):
        have = sorted(obs.jpgs()) if obs.img else []
        sys.exit(f"[record] no data: lowstate {'ok' if obs.measured is not None else 'MISSING'}, cameras {have} "
                 f"(domain {obs.cfg['domain']}, image port {a.image_port})")

    from g1lab.lerobot_writer import LeRobotWriter
    repo_id = a.repo_id or f"local/{os.path.basename(os.path.normpath(a.root))}"
    writer = LeRobotWriter(a.root, repo_id, a.task, fps=int(a.fps), waist=waist)
    sink = EpisodeSink(writer, binocular=obs.binocular)
    sess = Session(sink, a.task, waist=waist, auto_success=a.auto_success, only_success=a.only_success,
                   success_hold=a.success_hold, auto_start=a.auto_start, start_delay=a.start_delay,
                   min_frames=a.min_frames, episode_seconds=a.episode_seconds)
    keys = KeyReader()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    print(f"[record] {a.target} domain {obs.cfg['domain']} | {'15 joints (+waist)' if waist else '14 joints'} -> "
          f"{a.root} ({writer.num_episodes} episodes there) | waist action: {a.waist_action}", flush=True)
    print("[record] keys: s = start/stop+save, d = discard, q = quit" if keys.active else
          "[record] stdin is not a terminal: keys disabled (use --auto-start / --start-now, stop with SIGINT/SIGTERM)",
          flush=True)
    if a.start_now:
        sess.start()

    dt = 1.0 / a.fps
    resets_seen = obs.resets
    dropped = 0
    last_status = 0.0
    t_next = time.monotonic()
    src = None
    try:
        while not sess.quit and not stop["flag"]:
            now = time.monotonic()
            for ch in keys.poll():
                sess.key(ch, now)
            if obs.resets != resets_seen:
                resets_seen = obs.resets
                sess.on_reset(now)
            s = obs.sample(now)
            jpgs = obs.jpgs()
            if s is not None and len(jpgs) == len(CAMERAS):
                src = s["source"]
                arms, w = command_for(src, s["measured"], s["lowcmd"], s["arm_sdk"], a.waist_action)
                sess.tick(now, state_from(s["measured"], waist), arms, w, jpgs, obs.success())
            elif sess.recording:
                dropped += 1
            if a.max_episodes and len(sink.saved) >= a.max_episodes and not sess.recording:
                break
            if now - last_status >= 1.0:
                last_status = now
                status = (f"[record] {'REC ep ' + str(sess.episodes) if sess.recording else 'idle'} "
                          f"{sess.frames:4d} fr | src {src or '-'} | waist "
                          f"{(s['measured'][WAIST_YAW_INDEX] if s else float('nan')):+.2f} | success {int(obs.success())}"
                          f" | saved {len(sink.saved)} disc {sink.discarded} | queue {sink.backlog()}"
                          f"{' (encoding)' if sink.busy else ''}{f' | dropped {dropped}' if dropped else ''}")
                print(("\r" + status + "   ") if keys.active else status, end="" if keys.active else "\n", flush=True)
            t_next += dt
            sleep = t_next - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            elif sleep < -0.5:
                t_next = time.monotonic()
    finally:
        keys.restore()
        if sess.recording:
            print("\n[record] interrupted: dropping the unsaved episode", flush=True)
            sess.discard("interrupted")
        print(f"\n[record] finishing: {sink.backlog()} queued events (encoding may take a while)...", flush=True)
        sink.close()
        obs.close()
    print(f"[record] done: saved episodes {sink.saved} -> {a.root}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # DDS / teleimager threads would otherwise keep the process alive


if __name__ == "__main__":
    main()
