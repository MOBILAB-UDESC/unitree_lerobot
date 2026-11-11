"""Write G1 episodes as a LeRobot v3.0 dataset in the exact format of the real ``pick_cubeDataset``.

    w = LeRobotWriter("datasets/sim_pick_cube", "local/sim_pick_cube", "pick up cube.")
    w.start_episode()
    for ...:
        w.add_frame(state14, action14, {"cam_head": rgb, "cam_left_wrist": rgb, "cam_right_wrist": rgb})
    w.save_episode()          # or w.discard_episode()
    w.close()                 # always: calls dataset.finalize(), else the parquet files are unreadable

Requires lerobot 0.4.1 (the version unitree_lerobot pins). See g1lab/README.md.
"""

import logging
import os
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

log = logging.getLogger("g1lab.lerobot_writer")

# SVT-AV1 prints ~40 info lines per encoded video; keep errors only (set SVT_LOG=3 to see them).
os.environ.setdefault("SVT_LOG", "1")

ROBOT_TYPE = "Unitree_G1_ArmsOnly_3Cam"

# Arm joints in G1 motor order (left 15-21, right 22-28), as named in the real dataset.
JOINT_NAMES = [
    f"k{side}{joint}"
    for side in ("Left", "Right")
    for joint in ("ShoulderPitch", "ShoulderRoll", "ShoulderYaw", "Elbow", "WristRoll", "WristPitch", "WristYaw")
]
CAMERAS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]
IMAGE_SHAPE = (480, 640, 3)  # H, W, C

# The real dataset's file layout settings (lerobot 0.4.1 defaults video files to 500 MB).
CHUNKS_SIZE = 1000
DATA_FILES_SIZE_IN_MB = 100
VIDEO_FILES_SIZE_IN_MB = 200

# Optional 15-joint variant: the 14 arm joints + waist yaw (G1 motor 12), for episodes that turn the torso.
ROBOT_TYPE_WAIST = "Unitree_G1_ArmsWaist_3Cam"
JOINT_NAMES_WAIST = JOINT_NAMES + ["kWaistYaw"]


def make_features(joint_names):
    return {
        "observation.state": {"dtype": "float32", "shape": (len(joint_names),), "names": [joint_names]},
        "action": {"dtype": "float32", "shape": (len(joint_names),), "names": [joint_names]},
        **{
            f"observation.images.{cam}": {
                "dtype": "video",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            }
            for cam in CAMERAS
        },
    }


FEATURES = make_features(JOINT_NAMES)
FEATURES_WAIST = make_features(JOINT_NAMES_WAIST)


class LeRobotWriter:
    """Records episodes into a LeRobot v3.0 dataset at ``root``; appends if one already exists there.

    ``state``/``action`` are the 14 arm joint positions (rad) in ``JOINT_NAMES`` order, or with
    ``waist=True`` those plus waist yaw (15, ``JOINT_NAMES_WAIST``, robot_type ``ROBOT_TYPE_WAIST``).
    Images are RGB uint8 arrays of ``IMAGE_SHAPE``, keyed by ``CAMERAS`` names.
    ``save_episode()`` encodes the videos (AV1, a few seconds per episode) and blocks until done.
    """

    def __init__(self, root, repo_id: str, task: str, fps: int = 30, image_writer_threads: int = 4 * len(CAMERAS),
                 waist: bool = False):
        self.joint_names = JOINT_NAMES_WAIST if waist else JOINT_NAMES
        self.features = FEATURES_WAIST if waist else FEATURES
        self.robot_type = ROBOT_TYPE_WAIST if waist else ROBOT_TYPE
        self.root = Path(root)
        self.repo_id = repo_id
        self.task = task
        self.fps = fps
        self._episode_task = None
        self._closed = False

        if (self.root / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset(repo_id, root=self.root)
            self._check_compatible()
            log.info("appending to %s (%d episodes)", self.root, self.dataset.meta.total_episodes)
        else:
            if self.root.is_dir() and not any(self.root.iterdir()):
                self.root.rmdir()  # LeRobotDataset.create() insists on making the dir itself
            elif self.root.exists():
                raise FileExistsError(f"{self.root} exists but is not a LeRobot dataset")
            self.dataset = LeRobotDataset.create(
                repo_id, fps, self.features, root=self.root, robot_type=self.robot_type, use_videos=True
            )
            self.dataset.meta.update_chunk_settings(CHUNKS_SIZE, DATA_FILES_SIZE_IN_MB, VIDEO_FILES_SIZE_IN_MB)
        self.dataset.start_image_writer(num_processes=0, num_threads=image_writer_threads)

    def _check_compatible(self):
        info = self.dataset.meta.info
        problems = []
        if info["fps"] != self.fps:
            problems.append(f"fps {info['fps']} != {self.fps}")
        if info.get("robot_type") != self.robot_type:
            problems.append(f"robot_type {info.get('robot_type')!r} != {self.robot_type!r}")
        for key, ft in self.features.items():
            have = info["features"].get(key)
            if have is None or have["dtype"] != ft["dtype"] or tuple(have["shape"]) != tuple(ft["shape"]) \
                    or have["names"] != ft["names"]:
                problems.append(f"feature {key} differs: {have}")
        if problems:
            raise ValueError(f"cannot append to {self.root}: " + "; ".join(problems))

    # ------------------------------------------------------------------ episodes

    @property
    def num_episodes(self) -> int:
        """Episodes saved in the dataset so far (including ones that were there before)."""
        return self.dataset.meta.total_episodes

    @property
    def num_frames(self) -> int:
        """Frames added to the current, unsaved episode."""
        buf = self.dataset.episode_buffer
        return buf["size"] if buf else 0

    def start_episode(self, task: str | None = None):
        """Begin a new episode. ``task`` overrides the writer's task for this episode only.

        Frames left over from an episode that was never saved are discarded.
        """
        if self.num_frames:
            log.warning("start_episode(): discarding %d unsaved frames", self.num_frames)
            self.discard_episode()
        self._episode_task = task or self.task

    def add_frame(self, state, action, images: dict):
        """Add one timestep. ``images`` maps each name in ``CAMERAS`` to an RGB uint8 (480, 640, 3) array."""
        if self._episode_task is None:
            raise RuntimeError("call start_episode() before add_frame()")
        frame = {
            "observation.state": _vec("state", state, len(self.joint_names)),
            "action": _vec("action", action, len(self.joint_names)),
            "task": self._episode_task,
        }
        if set(images) != set(CAMERAS):
            raise ValueError(f"images must have exactly {CAMERAS}, got {sorted(images)}")
        for cam, img in images.items():
            img = np.asarray(img)
            if img.dtype != np.uint8 or img.shape != IMAGE_SHAPE:
                raise ValueError(f"{cam}: need uint8 {IMAGE_SHAPE} RGB, got {img.dtype} {img.shape}")
            frame[f"observation.images.{cam}"] = img
        self.dataset.add_frame(frame)

    def save_episode(self) -> int | None:
        """Write the current episode (encodes the videos). Returns its episode index, or None if it was empty."""
        if self._episode_task is None:
            raise RuntimeError("call start_episode() before save_episode()")
        self._episode_task = None
        if not self.num_frames:
            log.warning("save_episode(): episode has no frames, nothing saved")
            return None
        index = self.num_episodes
        self.dataset.save_episode()
        log.info("saved episode %d", index)
        return index

    def discard_episode(self):
        """Drop the current episode's frames (and their temporary images)."""
        self._episode_task = None
        if self.dataset.episode_buffer is not None:
            self.dataset.clear_episode_buffer(delete_images=True)

    def close(self):
        """Discard any unsaved episode and finalize the dataset. Must be called; safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self.num_frames:
            log.warning("close(): discarding %d unsaved frames", self.num_frames)
            self.discard_episode()
        self.dataset.stop_image_writer()
        self.dataset.finalize()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _vec(name, x, n) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.shape != (n,):
        raise ValueError(f"{name}: need {n} joints, got shape {np.shape(x)}")
    return v
