"""Deploy a trained LeRobot ACT checkpoint on the G1 arms (+ waist), closed loop at 30 Hz through G1Arms.

    # sim (start it first, e.g. scripts/run_sim.sh --scene lab --randomize --no-viewer --domain 7 --port-offset 1000)
    python 05_training/run_act.py --policy-path outputs/act/checkpoints/last/pretrained_model \
        --episodes 5 --domain 7 --image-port 61000 [--record datasets/act_rollouts]
    # compute actions on the live robot/sim without sending anything
    python 05_training/run_act.py --policy-path ... --dry-run --max-seconds 10
    # real robot (asks for "yes" after a checklist)
    python 05_training/run_act.py --policy-path ... --target real

Loading (lerobot 0.4.1): config.json -> PreTrainedConfig, weights -> ACTPolicy.from_pretrained, and the pre/post
processors saved next to them (policy_preprocessor.json / policy_postprocessor.json + their .safetensors) via
make_pre_post_processors(pretrained_path=...). Those carry the training dataset's normalization stats, so no
dataset is needed at deploy time. The observation is built exactly like lerobot-record does
(predict_action): observation.state float32 (14 or 15 = the checkpoint's input feature), each
observation.images.<cam> RGB uint8 HWC -> float CHW in [0, 1]. A 15-dim action drives the waist too
(G1Arms(waist=True).command -> command_waist); 14 drives the arms only.

ACT predicts a chunk of actions (100 = 3.3 s). The 30 Hz loop never waits for the network: with --replan async
(default) a worker thread computes a new chunk from the newest observation as fast as the PC allows, while the
current chunk keeps playing; the new chunk is entered at the step that passed since its observation. So the
policy re-plans every 0.5-1 s on a laptop CPU and every tick on a GPU. --replan chunk is LeRobot's plain
behaviour (play n_action_steps, then stop and infer). --threads sets the PyTorch CPU threads.

Safety (every tick, in this order; counters are printed per episode):
  * start: the first action is computed, then the robot ramps there from its measured pose (cosine, at most
    --ramp-vel rad/s, at least --ramp-time s) before the closed loop starts;
  * joint limits (URDF arm limits, waist +-1.0 rad), non-finite actions -> hold;
  * lead clamp: |command - measured| <= --max-lead rad (no large tracking errors -> no torque spikes);
  * velocity limit: |command change| <= --max-vel rad/s (per 1/30 s tick);
    defaults 1.0 rad/s and 0.25 rad. The scripted sim demonstrations reach 2.1 rad/s, so a policy trained
    on them runs slower than its plan with the defaults; --max-vel 2.5 --max-lead 0.9 lifts that in sim;
  * --max-seconds per episode; Ctrl-C stops the policy and HOLDS the current measured pose until a second
    Ctrl-C (then exit). After a normal end, --end start moves slowly back to the start pose, --end hold holds.
In sim each episode starts with a sim reset (objects re-randomised with --randomize) and counts successes
(rt/sim_state). --record DIR saves the rollouts (state = measured, action = the command actually sent).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (os.path.join(REPO, "04_recording"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from g1lab.remote_vla import load_joint_limits, wait_sim_reset  # noqa: E402
from g1lab.robot_io import WAIST_YAW_LIMIT  # noqa: E402

CONTROL_HZ = 30.0
DEFAULT_TASK = "put the wrapped box inside the cardboard box"
STATE_KEY, ACTION_KEY, IMAGE_PREFIX = "observation.state", "action", "observation.images."

REAL_CHECKLIST = """\
REAL ROBOT checklist:
  [ ] G1 hanging on the gantry / stand, area around the arms clear, e-stop in reach of a second person
  [ ] robot in DEBUG mode (L2+R2 on the remote): run_act sends rt/lowcmd like xr_teleoperate without --motion
      (with --motion: arm_sdk, the waist is NOT driven)
  [ ] nothing else is commanding the robot (xr_teleoperate, MCP, other run_act) - check with record_live
  [ ] image server running on PC2 (cam_head + both wrist cameras), lighting/scene like the training data
  [ ] checkpoint trained for THIS robot/camera setup; tried in sim and with --dry-run first
  [ ] limits: --max-vel {max_vel} rad/s, --max-lead {max_lead} rad, --max-seconds {max_seconds}"""


# ------------------------------------------------------------------------------------------ pure parts
def policy_io(config):
    """(state_dim, image keys, action_dim) from a lerobot policy config's input/output features."""
    feats = config.input_features
    if STATE_KEY not in feats:
        raise ValueError(f"checkpoint has no {STATE_KEY} input")
    state_dim = int(feats[STATE_KEY].shape[0])
    images = sorted(k[len(IMAGE_PREFIX):] for k in feats if k.startswith(IMAGE_PREFIX))
    action_dim = int(config.output_features[ACTION_KEY].shape[0])
    if state_dim not in (14, 15) or action_dim != state_dim:
        raise ValueError(f"expected a 14- or 15-joint G1 policy, got state {state_dim} / action {action_dim}")
    return state_dim, images, action_dim


def joint_limits(n):
    lo, hi = load_joint_limits()
    if n == 15:
        lo, hi = np.append(lo, -WAIST_YAW_LIMIT), np.append(hi, WAIST_YAW_LIMIT)
    return lo, hi


class SafetyGate:
    """Limits -> lead clamp around the measured pose -> per-tick velocity limit against the previous command."""

    def __init__(self, lower, upper, max_vel=1.0, max_lead=0.25, hz=CONTROL_HZ):
        self.lower, self.upper = np.asarray(lower, float), np.asarray(upper, float)
        self.max_step = float(max_vel) / hz
        self.max_lead = float(max_lead)
        self.counts = dict(limit=0, lead=0, vel=0, rejected=0)

    def __call__(self, target, prev, measured):
        prev = np.asarray(prev, float)
        target = np.asarray(target, float).reshape(-1)
        if target.shape != prev.shape or not np.all(np.isfinite(target)):
            self.counts["rejected"] += 1
            return prev.copy()
        t = np.clip(target, self.lower, self.upper)
        self.counts["limit"] += bool(np.any(t != target))
        m = np.asarray(measured, float)
        tl = np.clip(t, m - self.max_lead, m + self.max_lead)
        self.counts["lead"] += bool(np.any(tl != t))
        d = tl - prev
        dc = np.clip(d, -self.max_step, self.max_step)
        self.counts["vel"] += bool(np.any(dc != d))
        return np.clip(prev + dc, self.lower, self.upper)


def ramp(q0, q1, hz=CONTROL_HZ, min_time=2.0, max_vel=0.5):
    """Cosine waypoints q0 -> q1 (excluding q0), slow enough that the peak speed stays <= max_vel."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    dist = float(np.max(np.abs(q1 - q0))) if q0.size else 0.0
    # cosine profile peak speed = pi/2 * dist / T
    t = max(min_time, np.pi / 2 * dist / max(max_vel, 1e-6))
    n = max(1, int(np.ceil(t * hz)))
    s = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
    return q0 + (q1 - q0) * s[:, None]


class ImageHold:
    """Keeps the last image per camera, so a momentarily missing stream doesn't break the observation."""

    def __init__(self, cams):
        self.cams, self.last = list(cams), {}

    def update(self, images):
        for c in self.cams:
            if images.get(c) is not None:
                self.last[c] = images[c]
        return dict(self.last) if all(c in self.last for c in self.cams) else None


# ------------------------------------------------------------------------------------- lerobot policy
def constant_state_joints(preprocessor, tol=1e-6):
    """Indices of observation.state dims whose training std is ~0 (e.g. the zero-padded waist of real_waist)."""
    for step in getattr(preprocessor, "steps", []):
        std = (getattr(step, "stats", None) or {}).get(STATE_KEY, {}).get("std")
        if std is not None:
            return [int(i) for i in np.flatnonzero(np.asarray(std, dtype=float).reshape(-1) < tol)]
    return []


class LeRobotPolicy:
    """policy(state, images) -> action (np), using the checkpoint's own processors (normalization stats)."""

    def __init__(self, path, device="cpu", n_action_steps=None, task=DEFAULT_TASK, threads=None):
        import torch
        if threads:  # scripts/env.sh sets OMP_NUM_THREADS=1 for the sim; inference wants more
            torch.set_num_threads(int(threads))
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        self.torch = torch
        self.device = torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        cfg.pretrained_path = path
        if n_action_steps:
            cfg.n_action_steps = int(n_action_steps)
        self.config = cfg
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=path, preprocessor_overrides={"device_processor": {"device": device}})
        self.state_dim, self.cams, _ = policy_io(cfg)
        self.image_shapes = {c: tuple(cfg.input_features[IMAGE_PREFIX + c].shape) for c in self.cams}  # (C, H, W)
        self.task = task
        self.constant_joints = constant_state_joints(self.pre)

    @property
    def stats_warning(self):
        if not self.constant_joints:
            return ""
        return (f"[act] WARNING: observation.state joints {self.constant_joints} never moved in the training data "
                f"(std ~0). lerobot divides by std + 1e-8, so any offset there at run time (e.g. a measured waist "
                f"of 0.01 rad) becomes a huge input. Train with data that moves them, or a 14-joint policy.")

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def _obs(self, state, images):
        obs = {STATE_KEY: np.asarray(state, np.float32).reshape(self.state_dim)}
        for c in self.cams:
            img = images[c]
            _, h, w = self.image_shapes[c]
            if img.shape[:2] != (h, w):
                import cv2
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            obs[IMAGE_PREFIX + c] = np.ascontiguousarray(img)
        return obs

    def chunk(self, state, images):
        """The whole predicted chunk (chunk_size, action_dim), un-normalised; no action queue involved."""
        from lerobot.utils.control_utils import prepare_observation_for_inference
        torch = self.torch
        with torch.inference_mode():
            obs = prepare_observation_for_inference(self._obs(state, images), self.device, self.task, "")
            actions = self.policy.predict_action_chunk(self.pre(obs))  # (1, chunk, dim), normalised
            return np.stack([self.post(actions[:, i]).squeeze(0).float().cpu().numpy()
                             for i in range(actions.shape[1])])

    def __call__(self, state, images):
        from lerobot.utils.control_utils import predict_action
        obs = self._obs(state, images)
        act = predict_action(obs, self.policy, self.device, self.pre, self.post, use_amp=False, task=self.task,
                             robot_type="")
        return act.squeeze(0).float().cpu().numpy()


class AsyncChunks:
    """policy(state, images) -> action without ever blocking the control loop (after the first call).

    A worker thread computes the next chunk from the newest observation while the current chunk keeps playing.
    A finished chunk replaces the current one and is entered at the tick that has passed since its observation,
    so the actions stay aligned in time. `plan(state, images)` returns a (chunk, dim) array.
    """

    def __init__(self, plan):
        self.plan = plan
        self.lock = threading.Lock()
        self.generation = 0
        self.reset()

    def reset(self):
        with self.lock:
            self.generation += 1  # a plan still being computed belongs to the old episode: drop it
            self.plans = 0
            self.current, self.i, self.tick, self.ready, self.busy = None, 0, 0, None, False

    def _work(self, state, images, tick, generation):
        chunk = self.plan(state, images)
        with self.lock:
            if generation == self.generation:
                self.ready, self.busy = (chunk, tick), False
                self.plans += 1

    def __call__(self, state, images):
        if self.current is None:  # first action of an episode: nothing to play meanwhile
            self.current, self.i = self.plan(state, images), 0
        with self.lock:
            if self.ready is not None:
                (chunk, tick), self.ready = self.ready, None
                if self.tick - tick < len(chunk):
                    self.current, self.i = chunk, self.tick - tick
            start = not self.busy
            if start:
                self.busy = True
                args = (np.array(state), {k: v.copy() for k, v in images.items()}, self.tick, self.generation)
        if start:
            threading.Thread(target=self._work, args=args, daemon=True).start()
        action = self.current[min(self.i, len(self.current) - 1)]
        self.i += 1
        self.tick += 1
        return action


# ---------------------------------------------------------------------------------------- the loop
class Runner:
    """Runs episodes of `policy` on `robot` (G1Arms-like: state/observe/command/success/reset_sim, .sim)."""

    def __init__(self, robot, policy, n_joints, gate, cams, hz=CONTROL_HZ, send=True, writer=None,
                 drive_waist=True, ramp_time=2.0, ramp_vel=0.5, min_success_steps=15, log=print,
                 sleep=time.sleep, clock=time.perf_counter):
        self.robot, self.policy, self.n, self.gate, self.hz = robot, policy, n_joints, gate, hz
        self.send, self.writer, self.drive_waist = send, writer, drive_waist
        self.ramp_time, self.ramp_vel, self.min_success_steps = ramp_time, ramp_vel, min_success_steps
        self.images = ImageHold(cams)
        self.log, self.sleep, self.clock = log, sleep, clock
        self.stop = False
        self.commands = 0

    # -- helpers
    def _state(self):
        return np.asarray(self.robot.state(), float)[: self.n]

    def _command(self, q):
        if not self.send:
            return
        q = np.asarray(q, float)
        self.robot.command(q if (self.n == 14 or self.drive_waist) else q[:14])
        self.commands += 1

    def _observe(self, timeout=5.0):
        t_end = self.clock() + timeout
        while True:
            obs = self.robot.observe()
            images = self.images.update(obs["images"])
            if images is not None:
                return np.asarray(obs["state"], np.float32)[: self.n], images
            if self.clock() > t_end:
                raise RuntimeError(f"cameras missing: have {sorted(obs['images'])}, need {self.images.cams}")
            self.sleep(0.05)

    def _pace(self, t_next):
        t_next += 1.0 / self.hz
        dt = t_next - self.clock()
        if dt > 0:
            self.sleep(dt)
            return t_next, False
        return (self.clock(), True) if dt < -0.5 else (t_next, True)

    def hold(self):
        """Freeze at the measured pose."""
        q = self._state()
        self._command(q)
        return q

    def move_to(self, q, min_time=None):
        """Slow cosine move from the measured pose (used for the start ramp and --end start)."""
        q0 = self._state()
        lo, hi = self.gate.lower, self.gate.upper
        q1 = np.clip(np.asarray(q, float)[: self.n], lo, hi)
        t_next = self.clock()
        for wp in ramp(q0, q1, self.hz, self.ramp_time if min_time is None else min_time, self.ramp_vel):
            if self.stop:
                return False
            self._command(wp)
            t_next, _ = self._pace(t_next)
        return True

    # -- episodes
    def run_episode(self, max_steps, episode=0):
        st = dict(episode=episode, steps=0, reason="max_steps", infer_ms=[], overruns=0)
        self.gate.counts = {k: 0 for k in self.gate.counts}
        self.policy.reset()
        state, images = self._observe()
        a0 = np.asarray(self.policy(state, images), float)
        self.log(f"[act] ep {episode}: first action computed, ramping to it "
                 f"(max |delta| {np.max(np.abs(a0 - state)):.3f} rad){'' if self.send else ' [dry-run: not sent]'}")
        if self.send and not self.move_to(a0):
            st["reason"] = "interrupted"
            return self._finish(st)
        self.policy.reset()  # re-plan from the pose actually reached
        if self.writer is not None:
            self.writer.start_episode()
        prev = self._state()
        t_next = self.clock()
        step = 0
        while step < max_steps and not self.stop:
            state, images = self._observe()
            t0 = self.clock()
            target = self.policy(state, images)
            st["infer_ms"].append((self.clock() - t0) * 1e3)
            cmd = self.gate(target, prev, state)
            self._command(cmd)
            if self.writer is not None:
                self.writer.add_frame(state, cmd.astype(np.float32), images)
            prev = cmd if self.send else np.asarray(state, float)
            step += 1
            if not self.send and step % int(self.hz) == 0:
                target = np.asarray(target, float).reshape(-1)
                waist = f", waist action {target[14]:+.3f} (measured {state[14]:+.3f})" if self.n == 15 else ""
                self.log(f"[act] dry-run step {step}: max |action - state| {np.max(np.abs(target - state)):.3f} rad"
                         f"{waist}")
            # dry-run never resets the sim, so its success flag may be stale: only judge episodes we drive
            if self.send and getattr(self.robot, "sim", False) and step >= self.min_success_steps \
                    and self.robot.success():
                st["reason"] = "success"
                break
            t_next, over = self._pace(t_next)
            st["overruns"] += over
        if self.stop and st["reason"] == "max_steps":
            st["reason"] = "interrupted"
        st["steps"] = step
        return self._finish(st)

    def _finish(self, st):
        inf = np.asarray(st["infer_ms"] or [0.0])
        st.update(infer_ms_mean=float(inf.mean()), infer_ms_max=float(inf.max()), **{f"gate_{k}": v for k, v in
                                                                                       self.gate.counts.items()})
        del st["infer_ms"]
        self.log(f"[act] ep {st['episode']}: {st['reason']} after {st['steps']} steps | inference mean "
                 f"{st['infer_ms_mean']:.0f} / max {st['infer_ms_max']:.0f} ms | overruns {st['overruns']} | gate "
                 f"limit {st['gate_limit']} lead {st['gate_lead']} vel {st['gate_vel']} rejected {st['gate_rejected']}"
                 + (f" | re-plans {self.policy.plans}" if hasattr(self.policy, "plans") else ""))
        return st


def finish_recording(writer, st, save_on, log=print):
    if writer is None:
        return
    keep = st["reason"] == "success" or (save_on == "all" and st["reason"] == "max_steps")
    if keep and writer.num_frames:
        log(f"[record] saved episode {writer.save_episode()}")
    else:
        writer.discard_episode()
        log(f"[record] discarded episode ({st['reason']})")


# ------------------------------------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", required=True, help=".../checkpoints/<step|last>/pretrained_model")
    p.add_argument("--target", default="sim", choices=["sim", "real"])
    p.add_argument("--domain", type=int, default=None, help="DDS domain override (sim --domain 7)")
    p.add_argument("--image-port", type=int, default=60000, help="teleimager request port (sim --port-offset 1000 -> 61000)")
    p.add_argument("--image-host", default=None)
    p.add_argument("--iface", default=None)
    p.add_argument("--motion", action="store_true", help="drive arms via rt/arm_sdk (robot in motion mode); no waist")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-action-steps", type=int, default=None, help="re-plan every K steps (default: checkpoint's)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-seconds", type=float, default=30.0, help="per episode")
    p.add_argument("--replan", choices=["async", "chunk"], default="async",
                   help="async: re-plan in the background as fast as the PC allows; chunk: play n_action_steps, "
                        "then stop and infer (plain LeRobot)")
    p.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads for inference")
    p.add_argument("--max-vel", type=float, default=1.0, help="rad/s per joint (command change limit)")
    p.add_argument("--max-lead", type=float, default=0.25, help="rad, max |command - measured| per joint")
    p.add_argument("--ramp-time", type=float, default=2.0, help="min s for the start ramp / --end start move")
    p.add_argument("--ramp-vel", type=float, default=0.4, help="peak rad/s of the start ramp")
    p.add_argument("--end", choices=["start", "hold"], default="start",
                   help="after the episodes: move slowly back to the start pose, or hold until Ctrl-C")
    p.add_argument("--dry-run", action="store_true", help="read-only: compute actions, send nothing")
    p.add_argument("--record", metavar="DIR", default=None, help="save rollouts as a LeRobot dataset")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--save-on", choices=["all", "success"], default="all")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    policy = LeRobotPolicy(a.policy_path, a.device, a.n_action_steps, a.task, a.threads)
    n = policy.state_dim
    print(f"[act] {policy.config.type} from {a.policy_path}: state/action {n}, cameras {policy.cams}, "
          f"chunk {policy.config.chunk_size}, "
          + ("re-plans in the background" if a.replan == "async" else f"plays {policy.config.n_action_steps}")
          + f" | limits {a.max_vel} rad/s, {a.max_lead} rad", flush=True)
    if policy.stats_warning:
        print(policy.stats_warning, flush=True)
    if a.motion and n == 15:
        print("[act] --motion: the motion controller owns the waist; the policy's waist action is ignored", flush=True)
    if a.target == "real" and not a.dry_run:
        print(REAL_CHECKLIST.format(max_vel=a.max_vel, max_lead=a.max_lead, max_seconds=a.max_seconds))
        if input("Type 'yes' to run the policy on the REAL robot: ").strip().lower() != "yes":
            sys.exit("aborted")

    if a.dry_run:  # read-only observer: constructing G1Arms would already stream hold commands
        from g1lab.record_live import LiveObserver
        robot = LiveObserver(a.target, domain=a.domain, iface=a.iface, image_host=a.image_host,
                             image_port=a.image_port, waist=(n == 15))
        robot.wait_ready(10.0, policy.cams)
    else:
        from g1lab.robot_io import G1Arms
        robot = G1Arms(a.target, iface=a.iface, image_host=a.image_host, motion=a.motion, domain=a.domain,
                       image_port=a.image_port, waist=(n == 15))
    writer = None
    if a.record:
        from g1lab.lerobot_writer import LeRobotWriter
        writer = LeRobotWriter(a.record, a.repo_id or f"local/{os.path.basename(os.path.normpath(a.record))}",
                               a.task, waist=(n == 15))
    gate = SafetyGate(*joint_limits(n), max_vel=a.max_vel, max_lead=a.max_lead)
    actor = AsyncChunks(policy.chunk) if a.replan == "async" else policy
    runner = Runner(robot, actor, n, gate, policy.cams, send=not a.dry_run, writer=writer,
                    drive_waist=not a.motion, ramp_time=a.ramp_time, ramp_vel=a.ramp_vel)

    def on_sigint(sig, frame):
        if runner.stop:
            raise KeyboardInterrupt
        print("\n[act] Ctrl+C: policy stopped, HOLDING the current pose (Ctrl+C again to exit)", flush=True)
        runner.stop = True

    signal.signal(signal.SIGINT, on_sigint)
    start_q = np.asarray(robot.state(), float)[:n]
    results = []
    try:
        for ep in range(a.episodes):
            if robot.sim and not a.dry_run:
                if ep > 0 and not runner.move_to(start_q):
                    break
                wait_sim_reset(robot)
            st = runner.run_episode(int(a.max_seconds * CONTROL_HZ), ep)
            finish_recording(writer, st, a.save_on)
            results.append(st)
            if runner.stop:
                break
        if not a.dry_run:
            if runner.stop or a.end == "hold":
                q = runner.hold()
                print("[act] holding pose (Ctrl+C to exit)", flush=True)
                runner.stop = False
                while True:  # keep streaming the hold target; exit on the next Ctrl+C
                    robot.command(q if (n == 14 or not a.motion) else q[:14])
                    time.sleep(0.1)
            else:
                print("[act] moving back to the start pose", flush=True)
                runner.move_to(start_q)
    except KeyboardInterrupt:
        print("[act] exit", flush=True)
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        if writer is not None:
            writer.close()
    ok = sum(r["reason"] == "success" for r in results)
    print(f"[act] done: {ok}/{len(results)} successful episodes"
          f"{'' if getattr(robot, 'sim', False) else ' (success is only measured in sim)'}", flush=True)
    sys.stdout.flush()
    os._exit(0)  # DDS / teleimager threads would otherwise keep the process alive


if __name__ == "__main__":
    main()
