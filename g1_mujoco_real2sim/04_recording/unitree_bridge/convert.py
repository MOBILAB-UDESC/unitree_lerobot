"""Run unitree_lerobot's own JSON -> LeRobot converter with our G1 robot types registered at runtime.

    python -m unitree_bridge.convert --raw-dir xr_teleoperate/teleop/utils/data \
        --repo-id local/pick_cube_teleop --robot-type Unitree_G1_ArmsOnly_3Cam [--output-root datasets]

Needs an env with unitree_lerobot installed (github.com/unitreerobotics/unitree_lerobot; it pins lerobot
0.4.1 like g1lab). Nothing in unitree_lerobot is edited: this wrapper
  1. checks the input layout: --raw-dir is the PARENT of the task folders (raw_dir/<task>/<episode>/data.json),
     and that the first episode has the JSON keys / cameras the robot type needs (mono vs stereo head is
     detected from color_3 unless --binocular is given);
  2. registers Unitree_G1_ArmsOnly_3Cam / Unitree_G1_ArmsWaist_3Cam into ROBOT_CONFIGS (unitree_bridge.configs);
  3. calls unitree_lerobot.utils.convert_unitree_json_to_lerobot.json_to_lerobot(raw_dir, repo_id, robot_type),
     with lerobot's dataset creation wrapped so the result gets the real dataset's file-size settings and is
     finalize()d afterwards (the upstream script never calls finalize(); lerobot 0.4.1 needs it for the
     parquet footers);
  4. reloads the result with LeRobotDataset as a check.

The upstream converter always writes to $HF_LEROBOT_HOME/<repo_id> (default ~/.cache/huggingface/lerobot)
and DELETES that directory first; --output-root redirects HF_LEROBOT_HOME, and an existing output is refused
unless --overwrite. Unlike g1lab.convert_xr_json it uses each episode's text.goal as the task (set the goal in
xr_teleoperate, or use g1lab.convert_xr_json --task to force the real dataset's "pick up cube.").

UNTESTED against a real unitree_lerobot install (it is not installed on the dev PC); written from its
documented API (constants.RobotConfig / ROBOT_CONFIGS, json_to_lerobot(raw_dir, repo_id, robot_type, ...)).
The registration and checks are covered by 04_recording/g1lab/tests/test_unitree_bridge.py with a stand-in module.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from unitree_bridge.configs import ARMS_ONLY, ARMS_WAIST, config_fields, register  # noqa: E402

CONVERTER = "unitree_lerobot.utils.convert_unitree_json_to_lerobot"
# The real pick_cube dataset's layout settings (g1lab.lerobot_writer), so converted sets merge cleanly.
CHUNKS_SIZE, DATA_FILES_SIZE_IN_MB, VIDEO_FILES_SIZE_IN_MB = 1000, 100, 200


def find_episodes(raw_dir):
    """Episode dirs the upstream converter will see (raw_dir/*/*/data.json, sorted like it does)."""
    raw_dir = Path(raw_dir)
    eps = sorted(p.parent for p in raw_dir.glob("*/*/data.json"))
    if not eps and list(raw_dir.glob("*/data.json")):
        raise SystemExit(f"{raw_dir} holds episodes directly; pass its PARENT as --raw-dir "
                         f"(unitree_lerobot expects raw_dir/<task>/<episode>/data.json)")
    if not eps:
        raise SystemExit(f"no raw_dir/<task>/<episode>/data.json under {raw_dir}")
    return eps


def _get(d, dotted):
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def check_episode(data_json, robot_type, binocular=None):
    """Validate one xr_teleoperate data.json for robot_type; returns the binocular flag to use."""
    try:
        ep = json.loads(Path(data_json).read_text())
    except ValueError as e:
        raise SystemExit(f"{data_json}: not valid JSON ({e}) - a crashed recording? repair or delete it "
                         f"(g1lab.convert_xr_json repairs unclosed files)")
    frames = ep.get("data") or []
    if not frames:
        raise SystemExit(f"{data_json}: no frames")
    f0 = frames[0]
    colors = set((f0.get("colors") or {}).keys())
    if binocular is None:
        binocular = "color_3" in colors
    fields = config_fields(binocular)[robot_type]
    missing_cams = [k for k in fields["camera_to_image_key"] if k not in colors]
    if missing_cams:
        raise SystemExit(f"{data_json}: colors has {sorted(colors)}, robot type needs {missing_cams}")
    for side, names in (("states", fields["json_state_data_name"]), ("actions", fields["json_action_data_name"])):
        for name in names:
            v = _get(f0.get(side, {}), name)
            if v is None or (isinstance(v, list) and not v):
                hint = (" - xr_teleoperate does not record the waist; use Unitree_G1_ArmsOnly_3Cam + "
                        "g1lab.pad_waist, or record with g1lab.record_live" if "waist" in name else "")
                raise SystemExit(f"{data_json}: {side}.{name} is missing/empty{hint}")
    n_arm = len(_get(f0["states"], "left_arm.qpos")) + len(_get(f0["states"], "right_arm.qpos"))
    if n_arm != 14:
        raise SystemExit(f"{data_json}: {n_arm} arm joints, expected 14 (G1-29 arms)")
    return binocular


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", required=True, help="PARENT of the task folders (raw_dir/<task>/episode_XXXX)")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--robot-type", default=ARMS_ONLY, choices=[ARMS_ONLY, ARMS_WAIST])
    p.add_argument("--output-root", default=None,
                   help="write to <output-root>/<repo-id> instead of ~/.cache/huggingface/lerobot/<repo-id>")
    p.add_argument("--binocular", choices=["auto", "yes", "no"], default="auto",
                   help="head stream recorded as two eyes (color_0..3)? auto = color_3 present")
    p.add_argument("--mode", choices=["video", "image"], default="video")
    p.add_argument("--overwrite", action="store_true", help="allow the upstream converter to delete an existing output")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    eps = find_episodes(a.raw_dir)
    binocular = check_episode(eps[0] / "data.json", a.robot_type,
                              None if a.binocular == "auto" else a.binocular == "yes")
    print(f"[bridge] {len(eps)} episodes in {len({e.parent for e in eps})} task folder(s), "
          f"{'stereo' if binocular else 'mono'} head, robot_type {a.robot_type}", flush=True)

    if a.output_root:  # before lerobot is imported: HF_LEROBOT_HOME is read at import time
        os.environ["HF_LEROBOT_HOME"] = str(Path(a.output_root).resolve())
    import importlib
    conv = importlib.import_module(CONVERTER)
    if a.output_root:
        conv.HF_LEROBOT_HOME = Path(os.environ["HF_LEROBOT_HOME"])
    out = Path(conv.HF_LEROBOT_HOME) / a.repo_id
    if out.exists() and not a.overwrite:
        raise SystemExit(f"{out} exists and unitree_lerobot would delete it; pass --overwrite or another --repo-id")

    register(binocular)
    created = []
    LeRobotDataset = conv.LeRobotDataset
    orig_create = LeRobotDataset.create

    def create(*args, **kw):
        ds = orig_create(*args, **kw)
        ds.meta.update_chunk_settings(CHUNKS_SIZE, DATA_FILES_SIZE_IN_MB, VIDEO_FILES_SIZE_IN_MB)
        created.append(ds)
        return ds

    LeRobotDataset.create = create
    try:
        conv.json_to_lerobot(Path(a.raw_dir), a.repo_id, a.robot_type, push_to_hub=False, mode=a.mode)
    finally:
        LeRobotDataset.create = orig_create
        for ds in created:
            ds.finalize()

    ds = LeRobotDataset(a.repo_id, root=out)
    print(f"[bridge] wrote {out}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames, "
          f"state {ds.meta.features['observation.state']['shape']}", flush=True)


if __name__ == "__main__":
    main()
