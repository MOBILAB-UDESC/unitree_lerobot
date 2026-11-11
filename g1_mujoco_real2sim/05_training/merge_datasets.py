"""Merge LeRobot v3.0 datasets (e.g. real + sim) into one training set with lerobot's aggregate_datasets.

    python 05_training/merge_datasets.py --out datasets/mixed_waist --repo-id local/mixed_waist \
        datasets/real_waist datasets/sim_scripted_waist
    05_training/train_act.sh --root datasets/mixed_waist --repo-id local/mixed_waist --output outputs/act_mixed

Each source is a dataset dir (its repo_id is only a label here: local/<dir name>). lerobot requires identical
fps, robot_type and features, so the sources are checked first with a readable diff. The usual mismatch is
14 vs 15 joints: pad the real 14-joint set first (python -m g1lab.pad_waist SRC DST, waist = 0).
Different task strings are fine (they become separate task indices).

Why a script and not `lerobot-edit-dataset --operation.type merge`: in lerobot 0.4.1 that loads every source
from the same --root, so it only works for datasets in ~/.cache/huggingface/lerobot. This passes one root per
source. The output uses the real dataset's file-size settings (as g1lab.lerobot_writer does), stats are
aggregated from the sources, and the episodes keep the source order (episode index ranges are printed).

Mixing ratio: aggregate concatenates whole datasets. To change the real:sim balance, merge a subset (e.g.
split the sim set first with `lerobot-edit-dataset --operation.type split`) or train with
--dataset.episodes=[...] on the merged set.
"""

import argparse
import json
import sys
from pathlib import Path

CHUNKS_SIZE, DATA_FILES_SIZE_IN_MB, VIDEO_FILES_SIZE_IN_MB = 1000, 100, 200


def load_info(root):
    p = Path(root) / "meta" / "info.json"
    if not p.exists():
        raise SystemExit(f"{root} is not a LeRobot dataset (no meta/info.json)")
    return json.loads(p.read_text())


def compatibility_problems(infos):
    """List of human-readable differences between info.json dicts (relative to the first). Empty = mergeable."""
    problems = []
    ref = infos[0]
    for i, info in enumerate(infos[1:], start=1):
        tag = f"source {i}"
        if info.get("codebase_version") != ref.get("codebase_version"):
            problems.append(f"{tag}: codebase_version {info.get('codebase_version')} != {ref.get('codebase_version')}")
        if info["fps"] != ref["fps"]:
            problems.append(f"{tag}: fps {info['fps']} != {ref['fps']}")
        if info.get("robot_type") != ref.get("robot_type"):
            problems.append(f"{tag}: robot_type {info.get('robot_type')!r} != {ref.get('robot_type')!r}")
        fa, fb = ref["features"], info["features"]
        for key in sorted(set(fa) | set(fb)):
            if key not in fa or key not in fb:
                problems.append(f"{tag}: feature {key!r} only in {'source 0' if key in fa else tag}")
            elif fa[key] != fb[key]:
                a, b = fa[key], fb[key]
                diff = [f for f in ("dtype", "shape", "names", "info") if a.get(f) != b.get(f)]
                msg = f"{tag}: feature {key!r} differs in {diff}"
                if "shape" in diff and key in ("observation.state", "action") and {a["shape"][0], b["shape"][0]} == {14, 15}:
                    msg += " (14 vs 15 joints: pad the 14-joint set with `python -m g1lab.pad_waist SRC DST`)"
                problems.append(msg)
    return problems


def episode_ranges(counts):
    """[(first, last)] global episode indices per source after concatenation."""
    out, start = [], 0
    for n in counts:
        out.append((start, start + n - 1))
        start += n
    return out


def repair_video_timestamps(root):
    """Recompute every episode's from/to timestamp inside its video file; returns how many were wrong.

    lerobot 0.4.1's aggregate_datasets gets them wrong once a merged video file is closed and the next one
    starts while more sources follow: episodes in the later file keep the earlier file's duration as an offset.
    Such a dataset loads, but training stops with `Invalid frame index`. The files themselves are right
    (episodes are concatenated in order), so the timestamps follow from the episode lengths.
    """
    import pandas as pd
    root, fixed = Path(root), 0
    fps = load_info(root)["fps"]
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    ep = pd.concat([pd.read_parquet(f).assign(_src=str(f)) for f in files]).sort_values("episode_index")
    for col in [c for c in ep.columns if c.startswith("videos/") and c.endswith("/from_timestamp")]:
        key = col[: -len("from_timestamp")]
        start = ep.groupby([key + "chunk_index", key + "file_index"])["length"].transform(lambda s: s.cumsum() - s) / fps
        fixed += int(((ep[col] - start).abs() > 0.5 / fps).sum())
        ep[col], ep[key + "to_timestamp"] = start, start + ep["length"] / fps
    if fixed:
        for f in files:
            ep[ep["_src"] == str(f)].drop(columns="_src").to_parquet(f, index=False)
    return fixed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sources", nargs="+", help="dataset dirs, in the order their episodes should appear")
    p.add_argument("--repair", action="store_true",
                   help="only fix the video timestamps of already merged dataset dirs (the sources), in place")
    p.add_argument("--out", help="output dataset dir (must not exist)")
    p.add_argument("--repo-id", default=None, help="default: local/<basename of --out>")
    p.add_argument("--check-only", action="store_true", help="only check compatibility")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    roots = [Path(s).resolve() for s in a.sources]
    if a.repair:
        for r in roots:
            print(f"{r}: fixed {repair_video_timestamps(r)} episode video timestamps")
        return
    if not a.out:
        raise SystemExit("--out is required")
    if len(roots) < 2:
        raise SystemExit("need at least two sources")
    infos = [load_info(r) for r in roots]
    for r, i in zip(roots, infos):
        print(f"  {r}: {i['total_episodes']} episodes, {i['total_frames']} frames, {i.get('robot_type')}, "
              f"state {i['features']['observation.state']['shape']}")
    problems = compatibility_problems(infos)
    if problems:
        raise SystemExit("cannot merge:\n  " + "\n  ".join(problems))
    print("compatible")
    if a.check_only:
        return
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} already exists")
    repo_id = a.repo_id or f"local/{out.name}"

    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    aggregate_datasets([f"local/{r.name}" for r in roots], repo_id, roots=roots, aggr_root=out,
                       data_files_size_in_mb=DATA_FILES_SIZE_IN_MB, video_files_size_in_mb=VIDEO_FILES_SIZE_IN_MB,
                       chunk_size=CHUNKS_SIZE)
    fixed = repair_video_timestamps(out)
    if fixed:
        print(f"fixed {fixed} episode video timestamps (lerobot 0.4.1 aggregate bug)")
    ds = LeRobotDataset(repo_id, root=out)
    print(f"wrote {out} ({repo_id}): {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, "
          f"tasks {list(ds.meta.tasks.index)}")
    for r, (lo, hi) in zip(roots, episode_ranges([i["total_episodes"] for i in infos])):
        print(f"  episodes {lo}-{hi} <- {r}")


if __name__ == "__main__":
    sys.exit(main())
