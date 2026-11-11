"""Convert xr_teleoperate ``--record`` episodes into a LeRobot v3.0 dataset.

Input layout (written by ``xr_teleoperate/teleop/utils/episode_writer.py``)::

    <task_dir>/episode_0001/data.json
    <task_dir>/episode_0001/colors/000000_color_0.jpg ...

Output: the same format as the real ``pick_cubeDataset`` (see ``g1lab.lerobot_writer``).

    python -m g1lab.convert_xr_json \\
        --raw-dir xr_teleoperate/teleop/utils/data/sim_pick_cube \\
        --root datasets/sim_pick_cube --repo-id local/sim_pick_cube --task "pick up cube."
"""

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from g1lab.lerobot_writer import CAMERAS, IMAGE_SHAPE, LeRobotWriter

log = logging.getLogger("g1lab.convert_xr_json")

# xr_teleoperate names cameras by order: color_0 = head, then the wrists.
# With a binocular head, color_0/color_1 are the two eyes and the wrists move to color_2/color_3.
MONO_CAM_MAP = {"color_0": "cam_head", "color_1": "cam_left_wrist", "color_2": "cam_right_wrist"}
BINOCULAR_CAM_MAP = {"color_0": "cam_head", "color_2": "cam_left_wrist", "color_3": "cam_right_wrist"}


def load_episode_json(path: Path) -> dict:
    """Load data.json. Episodes cut off mid-write (teleop crashed) lack the closing ``]}``; repair those."""
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = text.rstrip().rstrip(",") + "\n]\n}"
        data = json.loads(repaired)
        log.warning("%s was not closed (teleop crashed?); repaired, %d frames", path, len(data["data"]))
        return data


def pick_cam_map(items: list[dict]) -> dict:
    keys = set()
    for item in items:
        keys.update((item.get("colors") or {}).keys())
    return BINOCULAR_CAM_MAP if "color_3" in keys else MONO_CAM_MAP


def read_rgb(episode_dir: Path, rel_path: str) -> np.ndarray:
    bgr = cv2.imread(str(episode_dir / rel_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(episode_dir / rel_path)
    h, w = IMAGE_SHAPE[:2]
    if bgr.shape[:2] != (h, w):
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def arm_qpos(part: dict) -> np.ndarray:
    return np.asarray(part["left_arm"]["qpos"] + part["right_arm"]["qpos"], dtype=np.float32)


def convert_episode(writer: LeRobotWriter, episode_dir: Path, cam_map: dict | None = None, task: str | None = None) -> int:
    """Add one xr_teleoperate episode to ``writer``. Returns the number of frames written (0 = skipped).

    ``task`` defaults to the episode's ``text.goal``, then to the writer's task.
    """
    data = load_episode_json(episode_dir / "data.json")
    items = data["data"]
    cam_map = cam_map or pick_cam_map(items)

    fps = (data.get("info") or {}).get("image", {}).get("fps")
    if fps and int(round(fps)) != writer.fps:
        log.warning("%s was recorded at %s fps, dataset is %d fps", episode_dir.name, fps, writer.fps)

    writer.start_episode(task=task or (data.get("text") or {}).get("goal") or None)
    last: dict[str, np.ndarray] = {}
    n = 0
    for item in items:
        colors = item.get("colors") or {}
        for key, cam in cam_map.items():
            if colors.get(key):
                last[cam] = read_rgb(episode_dir, colors[key])
        # A camera can drop a frame (teleop logs "image is None"): hold its last image.
        # Frames before every camera has delivered once are skipped.
        if len(last) < len(CAMERAS):
            continue
        writer.add_frame(arm_qpos(item["states"]), arm_qpos(item["actions"]), dict(last))
        n += 1

    if n == 0:
        log.warning("%s: no frame with all cameras %s, skipped", episode_dir.name, sorted(cam_map))
        writer.discard_episode()
        return 0
    if n < len(items):
        log.warning("%s: dropped %d leading frames without all cameras", episode_dir.name, len(items) - n)
    writer.save_episode()
    return n


def find_episodes(raw_dir: Path) -> list[Path]:
    """Episode dirs under ``raw_dir`` (a task dir) or one level deeper (a dir of task dirs)."""
    eps = sorted(p.parent for p in raw_dir.glob("episode_*/data.json"))
    if not eps:
        eps = sorted(p.parent for p in raw_dir.glob("*/episode_*/data.json"))
    return eps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True, help="xr_teleoperate task dir (holds episode_XXXX/)")
    ap.add_argument("--root", type=Path, required=True, help="LeRobot dataset root (appended to if it exists)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--task", default=None, help="task string; default: each episode's text.goal")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--cam-map", default=None,
                    help="override camera mapping, e.g. color_0=cam_head,color_1=cam_left_wrist,color_2=cam_right_wrist")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cam_map = None
    if args.cam_map:
        cam_map = dict(kv.split("=", 1) for kv in args.cam_map.split(","))
        if sorted(cam_map.values()) != sorted(CAMERAS):
            ap.error(f"--cam-map must map onto exactly {CAMERAS}")

    episodes = find_episodes(args.raw_dir)
    if not episodes:
        ap.error(f"no episode_*/data.json under {args.raw_dir}")

    # Without --task each episode uses its own text.goal; the first one is the fallback.
    default_task = args.task or load_episode_json(episodes[0] / "data.json").get("text", {}).get("goal", "")
    writer = LeRobotWriter(args.root, args.repo_id, default_task, fps=args.fps)
    total = 0
    try:
        for ep in episodes:
            n = convert_episode(writer, ep, cam_map, args.task)
            total += n
            log.info("%s -> %d frames", ep.name, n)
    finally:
        writer.close()
    log.info("done: %d episodes, %d frames in %s", writer.num_episodes, total, os.fspath(args.root))


if __name__ == "__main__":
    main()
