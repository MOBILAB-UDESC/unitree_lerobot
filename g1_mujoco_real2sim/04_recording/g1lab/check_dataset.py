"""Sanity-check a recorded LeRobot v3.0 dataset before training or uploading it.

Loads it the way training does (lerobot 0.4.1 LeRobotDataset) and prints what matters for a G1 arm dataset:
episodes / frames / layout, episode lengths, how much each joint moves (a joint that never moves has zero
variance: a policy trained on it alone is fragile, see pad_waist.py), and how many camera frames are
repeats of the previous one (a camera stream slower than the 30 Hz recording shows up here).

    source scripts/env.sh g1lab
    python -m g1lab.check_dataset datasets/sim_scripted_waist
"""

import argparse
import json
from pathlib import Path

import numpy as np


def repeated_frames(path, max_frames=None):
    """Fraction of frames that are (almost) identical to the previous frame of one video file."""
    import av
    diffs, prev = [], None
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if max_frames and i >= max_frames:
                break
            g = frame.to_ndarray(format="gray")[::4, ::4].astype(np.float32)
            if prev is not None:
                diffs.append(np.abs(g - prev).mean())
            prev = g
    d = np.array(diffs)
    return float((d < 0.15 * np.median(d)).mean()) if len(d) else 0.0, len(d) + 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", help="dataset folder (contains meta/, data/, videos/)")
    p.add_argument("--repo-id", default=None, help="only used as a label (default: the folder name)")
    p.add_argument("--videos", type=int, default=3,
                   help="check the first N x 1000 frames of each camera's first video file for repeated frames "
                        "(0 = skip)")
    a = p.parse_args()
    root = Path(a.root)
    info = json.loads((root / "meta" / "info.json").read_text())
    print(f"{root}: {info['total_episodes']} episodes, {info['total_frames']} frames, {info['fps']} fps, "
          f"robot_type {info['robot_type']}, codebase {info['codebase_version']}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(a.repo_id or f"local/{root.name}", root=root)
    item = ds[0]
    cams = [k for k in item if k.startswith("observation.images.")]
    print(f"loads in lerobot: {len(ds)} frames; state {tuple(item['observation.state'].shape)}, "
          f"action {tuple(item['action'].shape)}, cameras {[c.split('.')[-1] for c in cams]} "
          f"{tuple(item[cams[0]].shape) if cams else ''}")

    # Decode the first and last frame of every episode: wrong video offsets in the episode metadata (a merged
    # dataset, see 05_training/merge_datasets.py --repair) only show up when those frames are actually read.
    broken = []
    for e, (lo, hi) in enumerate(zip(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"])):
        try:
            ds[int(lo)], ds[int(hi) - 1]
        except Exception as err:
            broken.append((e, str(err).splitlines()[-1][:80]))
    print(f"video frames of every episode decode: {'yes' if not broken else f'NO, {len(broken)} episodes fail, e.g. {broken[0]}'}")

    table = ds.hf_dataset.with_format("numpy")
    state, action, ep = np.stack(table["observation.state"]), np.stack(table["action"]), np.asarray(table["episode_index"])
    lengths = np.bincount(ep) / info["fps"]
    print(f"episode length: mean {lengths.mean():.1f} s, min {lengths.min():.1f} s, max {lengths.max():.1f} s")
    names = info["features"]["observation.state"]["names"]
    names = names[0] if names and isinstance(names[0], list) else names
    spans = state.max(0) - state.min(0)
    still = [n for n, s in zip(names, spans) if s < 1e-3]
    print("joint range of motion [rad]: " + ", ".join(f"{n[1:] if n.startswith('k') else n} {s:.2f}"
                                                      for n, s in zip(names, spans)))
    print(f"joints that never move: {still or 'none'}")
    print(f"mean |action - state|: {np.abs(action - state).mean():.3f} rad")

    if a.videos:
        for cam in cams:
            files = sorted((root / "videos" / cam).rglob("*.mp4"))
            if files:
                frac, n = repeated_frames(files[0], a.videos * 1000)
                print(f"{cam.split('.')[-1]}: {100 * frac:.0f} % repeated frames in the first {n} frames")


if __name__ == "__main__":
    main()
