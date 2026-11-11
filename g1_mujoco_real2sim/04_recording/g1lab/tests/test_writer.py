"""Run: conda activate g1lab && unset PYTHONPATH LD_LIBRARY_PATH CYCLONEDDS_URI && python -m pytest g1lab/tests -v"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from g1lab.convert_xr_json import main as convert_main
from g1lab.lerobot_writer import CAMERAS, IMAGE_SHAPE, LeRobotWriter

REAL_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "pick_cube_real"  # <repo>/datasets/pick_cube_real
REPO_ID = "local/g1lab_test"
TASK = "pick up cube."
# Fields that legitimately differ between datasets of different length.
COUNT_FIELDS = {"total_episodes", "total_frames", "total_tasks", "splits"}

needs_real = pytest.mark.skipif(not (REAL_ROOT / "meta" / "info.json").exists(), reason="real dataset not present")


def synth_frame(t: int, ep: int):
    """Deterministic smooth images (cheap to encode) and joint values that identify (episode, frame)."""
    h, w, _ = IMAGE_SHAPE
    yy, xx = np.mgrid[0:h, 0:w]
    images = {}
    for c, cam in enumerate(CAMERAS):
        img = np.zeros(IMAGE_SHAPE, np.uint8)
        img[..., 0] = (xx + 8 * t) % 256
        img[..., 1] = (yy + 40 * c) % 256
        img[..., 2] = (60 * ep) % 256
        images[cam] = img
    state = np.linspace(-1, 1, 14, dtype=np.float32) + 0.01 * t + ep
    action = state + 0.05
    return state, action, images


def write_episodes(root, lengths, ep_offset=0):
    with LeRobotWriter(root, REPO_ID, TASK) as w:
        for i, n in enumerate(lengths):
            w.start_episode()
            for t in range(n):
                w.add_frame(*synth_frame(t, ep_offset + i))
            assert w.save_episode() == ep_offset + i
    return w


def assert_info_matches_real(root):
    ours = json.loads((root / "meta" / "info.json").read_text())
    real = json.loads((REAL_ROOT / "meta" / "info.json").read_text())
    for key in (set(ours) | set(real)) - COUNT_FIELDS:
        assert ours.get(key) == real.get(key), f"meta/info.json field {key!r} differs"


@pytest.fixture(scope="module")
def two_episodes(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds") / "g1lab_test"
    write_episodes(root, [40, 25])
    return root


@needs_real
def test_info_matches_real(two_episodes):
    assert_info_matches_real(two_episodes)
    info = json.loads((two_episodes / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"]) == (2, 65, 1)
    assert info["splits"] == {"train": "0:2"}


def test_reload(two_episodes):
    ds = LeRobotDataset(REPO_ID, root=two_episodes)
    assert ds.num_episodes == 2 and ds.num_frames == 65 and len(ds) == 65
    assert list(ds.meta.tasks.index) == [TASK]

    for idx, (ep, t) in [(0, (0, 0)), (39, (0, 39)), (40, (1, 0)), (64, (1, 24))]:
        item = ds[idx]
        state, action, images = synth_frame(t, ep)
        assert item["episode_index"].item() == ep and item["frame_index"].item() == t
        assert item["task"] == TASK
        assert item["observation.state"].dtype == torch.float32
        torch.testing.assert_close(item["observation.state"], torch.from_numpy(state))
        torch.testing.assert_close(item["action"], torch.from_numpy(action))
        assert item["timestamp"].item() == pytest.approx(t / 30, abs=1e-6)
        for cam in CAMERAS:
            img = item[f"observation.images.{cam}"]
            assert img.shape == (3, 480, 640)
            decoded = (img.permute(1, 2, 0).numpy() * 255).astype(np.float32)
            # Lossy AV1: compare the mean colour, which identifies the frame and the episode.
            np.testing.assert_allclose(decoded.mean((0, 1)), images[cam].mean((0, 1)), atol=6)


def test_parquet_schema_matches_real(two_episodes):
    import pyarrow.parquet as pq

    if not REAL_ROOT.exists():
        pytest.skip("real dataset not present")
    for rel in ["data/chunk-000/file-000.parquet", "meta/episodes/chunk-000/file-000.parquet"]:
        ours = pq.read_schema(two_episodes / rel)
        real = pq.read_schema(REAL_ROOT / rel)
        assert ours.remove_metadata().equals(real.remove_metadata()), rel
    assert (two_episodes / "meta" / "stats.json").exists()


def test_append_and_discard(tmp_path):
    root = tmp_path / "ds"
    write_episodes(root, [12])
    with LeRobotWriter(root, REPO_ID, TASK) as w:
        assert w.num_episodes == 1
        w.start_episode()
        for t in range(5):
            w.add_frame(*synth_frame(t, 9))
        w.discard_episode()
        w.start_episode(task="other task")
        for t in range(10):
            w.add_frame(*synth_frame(t, 1))
        assert w.save_episode() == 1
        w.start_episode()
        w.add_frame(*synth_frame(0, 9))  # unsaved at close(): dropped
    ds = LeRobotDataset(REPO_ID, root=root)
    assert ds.num_episodes == 2 and len(ds) == 22
    assert ds[12]["task"] == "other task" and ds[12]["episode_index"].item() == 1
    torch.testing.assert_close(ds[21]["observation.state"], torch.from_numpy(synth_frame(9, 1)[0]))
    if REAL_ROOT.exists():
        assert_info_matches_real(root)
    assert not any((root / "images").rglob("*.png")), "temporary frames left behind"


def test_rejects_bad_input(tmp_path):
    with LeRobotWriter(tmp_path / "ds", REPO_ID, TASK) as w:
        state, action, images = synth_frame(0, 0)
        with pytest.raises(RuntimeError):
            w.add_frame(state, action, images)
        w.start_episode()
        with pytest.raises(ValueError):
            w.add_frame(state[:7], action, images)
        with pytest.raises(ValueError):
            w.add_frame(state, action, {k: v for k, v in images.items() if k != "cam_head"})
        with pytest.raises(ValueError):
            w.add_frame(state, action, {**images, "cam_head": images["cam_head"][::2, ::2]})
        assert w.save_episode() is None


def test_convert_xr_json(tmp_path):
    """Fake an xr_teleoperate --record task dir (mono head, one unclosed data.json) and convert it."""
    task_dir = tmp_path / "raw" / "sim_pick_cube"
    lengths = [8, 6]
    for e, n in enumerate(lengths):
        ep_dir = task_dir / f"episode_{e + 1:04d}"
        (ep_dir / "colors").mkdir(parents=True)
        items = []
        for t in range(n):
            state, action, images = synth_frame(t, e)
            colors = {}
            for c, cam in enumerate(CAMERAS):
                name = f"colors/{t:06d}_color_{c}.jpg"
                cv2.imwrite(str(ep_dir / name), cv2.cvtColor(images[cam], cv2.COLOR_RGB2BGR))
                colors[f"color_{c}"] = name
            part = lambda q: {"left_arm": {"qpos": q[:7].tolist()}, "right_arm": {"qpos": q[7:].tolist()},
                              "left_ee": {"qpos": []}, "right_ee": {"qpos": []}, "body": {"qpos": []}}
            items.append({"idx": t, "colors": colors, "depths": {}, "states": part(state),
                          "actions": part(action), "tactiles": None, "audios": None, "sim_state": None})
        head = {"info": {"image": {"width": 640, "height": 480, "fps": 30}}, "text": {"goal": "goal from json"}}
        text = json.dumps({**head, "data": items}, indent=4)
        if e == 1:
            text = text.rstrip()[: -len("\n    ]\n}")]  # as left by a crashed teleop: no closing "]}"
            assert text.rstrip().endswith("}")
        (ep_dir / "data.json").write_text(text)

    root = tmp_path / "out"
    convert_main(["--raw-dir", str(task_dir), "--root", str(root), "--repo-id", REPO_ID, "--task", TASK])
    ds = LeRobotDataset(REPO_ID, root=root)
    assert ds.num_episodes == 2 and len(ds) == sum(lengths)
    item = ds[lengths[0] + 3]
    assert item["task"] == TASK and item["episode_index"].item() == 1
    torch.testing.assert_close(item["observation.state"], torch.from_numpy(synth_frame(3, 1)[0]))
    torch.testing.assert_close(item["action"], torch.from_numpy(synth_frame(3, 1)[1]))
    if REAL_ROOT.exists():
        assert_info_matches_real(root)


@needs_real
def test_real_dataset_loads():
    ds = LeRobotDataset("local/pick_cube", root=REAL_ROOT)
    assert ds.num_episodes == 52 and len(ds) == 17345
    for idx in (0, len(ds) - 1):
        item = ds[idx]
        assert item["observation.state"].shape == (14,)
        for cam in CAMERAS:
            assert item[f"observation.images.{cam}"].shape == (3, 480, 640)
