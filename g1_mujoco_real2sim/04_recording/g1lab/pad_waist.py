"""Convert a 14-joint "ArmsOnly" LeRobot v3.0 dataset to the 15-joint "ArmsWaist" format (waist yaw = 0).

The real recordings never moved the waist, so appending kWaistYaw = 0 to observation.state and action
is exact. This lets real and sim (waist-turning) episodes be merged and trained together. Only the
tables and metadata change (data parquet, per-episode stats, stats.json, info.json); the videos are
copied as they are.

    python -m g1lab.pad_waist datasets/pick_cube_real datasets/real_waist
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from g1lab.lerobot_writer import JOINT_NAMES, JOINT_NAMES_WAIST, ROBOT_TYPE_WAIST

VECTOR_KEYS = ("observation.state", "action")
STAT_PAD = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "q01": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0,
            "q99": 0.0}


def pad_vec(v, value=0.0):
    return np.append(np.asarray(v, dtype=np.float32), np.float32(value))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    a = p.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists():
        raise SystemExit(f"{dst} already exists")
    info = json.loads((src / "meta" / "info.json").read_text())
    for k in VECTOR_KEYS:
        if info["features"][k]["names"] != [JOINT_NAMES]:
            raise SystemExit(f"{src}: {k} is not the 14-joint arm layout")
    shutil.copytree(src, dst)

    for f in sorted((dst / "data").rglob("*.parquet")):
        df = pd.read_parquet(f)
        for k in VECTOR_KEYS:
            df[k] = [pad_vec(v) for v in df[k]]
        df.to_parquet(f, index=False)

    for f in sorted((dst / "meta" / "episodes").rglob("*.parquet")):
        df = pd.read_parquet(f)
        for k in VECTOR_KEYS:
            for stat, value in STAT_PAD.items():
                col = f"stats/{k}/{stat}"
                if col in df:
                    df[col] = [pad_vec(v, value) for v in df[col]]
        df.to_parquet(f, index=False)

    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    for k in VECTOR_KEYS:
        for stat, value in STAT_PAD.items():
            if stat in stats.get(k, {}):
                stats[k][stat] = list(stats[k][stat]) + [value]
    stats_path.write_text(json.dumps(stats, indent=4))

    info["robot_type"] = ROBOT_TYPE_WAIST
    for k in VECTOR_KEYS:
        info["features"][k]["shape"] = [len(JOINT_NAMES_WAIST)]
        info["features"][k]["names"] = [JOINT_NAMES_WAIST]
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    print(f"wrote {dst}: {info['total_episodes']} episodes, {info['total_frames']} frames, 15 joints (waist = 0)")


if __name__ == "__main__":
    main()
