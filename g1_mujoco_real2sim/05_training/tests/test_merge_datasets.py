"""merge_datasets: compatibility diff (pure) + one tiny real merge (writes two 2-episode datasets, ~15 s).

    source scripts/env.sh g1lab && python -m pytest 05_training/tests -v -p no:cacheprovider
"""

import copy
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import merge_datasets as md  # noqa: E402


def info(n=15, robot="Unitree_G1_ArmsWaist_3Cam", fps=30):
    vec = {"dtype": "float32", "shape": [n], "names": [[f"j{i}" for i in range(n)]]}
    return {"codebase_version": "v3.0", "fps": fps, "robot_type": robot, "total_episodes": 2,
            "features": {"observation.state": dict(vec), "action": copy.deepcopy(vec),
                         "observation.images.cam_head": {"dtype": "video", "shape": [480, 640, 3],
                                                         "names": ["height", "width", "channel"],
                                                         "info": {"video.codec": "av1"}}}}


def test_compatible_and_diffs():
    assert md.compatibility_problems([info(), info()]) == []
    p = md.compatibility_problems([info(14, "Unitree_G1_ArmsOnly_3Cam"), info()])
    assert any("robot_type" in s for s in p) and any("pad_waist" in s for s in p)
    other = info()
    other["features"]["observation.images.cam_head"]["info"]["video.codec"] = "h264"
    del other["features"]["action"]
    p = md.compatibility_problems([info(), other, info(fps=15)])
    assert any("'info'" in s for s in p) and any("only in source 0" in s for s in p)
    assert any("source 2: fps" in s for s in p)


def test_episode_ranges():
    assert md.episode_ranges([52, 5, 3]) == [(0, 51), (52, 56), (57, 59)]


def test_real_merge(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from g1lab.lerobot_writer import CAMERAS, IMAGE_SHAPE, LeRobotWriter

    def make(name, task, eps, offset):
        with LeRobotWriter(tmp_path / name, f"local/{name}", task, waist=True) as w:
            for e in range(eps):
                w.start_episode()
                for t in range(4):
                    v = np.full(15, offset + e + 0.01 * t, np.float32)
                    w.add_frame(v, v, {c: np.full(IMAGE_SHAPE, 40 * e, np.uint8) for c in CAMERAS})
                w.save_episode()

    make("real", "pick up cube.", 2, 0.0)
    make("sim", "put the wrapped box inside the cardboard box", 1, 10.0)
    md.main([str(tmp_path / "real"), str(tmp_path / "sim"), "--out", str(tmp_path / "mixed")])
    ds = LeRobotDataset("local/mixed", root=tmp_path / "mixed")
    assert ds.meta.total_episodes == 3 and ds.meta.total_frames == 12 and len(ds.meta.tasks) == 2
    assert ds.meta.robot_type == "Unitree_G1_ArmsWaist_3Cam"
    last = ds[11]
    assert float(last["observation.state"][0]) == pytest.approx(10.03, abs=1e-4)
    with pytest.raises(SystemExit, match="exists"):
        md.main([str(tmp_path / "real"), str(tmp_path / "sim"), "--out", str(tmp_path / "mixed")])
