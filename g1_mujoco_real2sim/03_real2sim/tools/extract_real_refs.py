"""Dump reference data from the real LeRobot v3.0 dataset for real2sim calibration.

For each requested episode: states/actions (npz) and the three camera frames at a few frame indices
(PNG). Needs pandas + pyarrow + ffmpeg (e.g. the `lerobot` or `g1lab` conda env).

    python 03_real2sim/tools/extract_real_refs.py --dataset datasets/pick_cube_real --episodes 0 7 23 \
        --frames 0 60 150 240 300 --out 03_real2sim/calib/real_refs
"""

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd

CAMS = ["cam_head", "cam_left_wrist", "cam_right_wrist"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--frames", type=int, nargs="+", default=[0, 60, 150, 240, 300])
    p.add_argument("--all-first-frames", action="store_true", help="also dump frame 0 head image of every episode")
    p.add_argument("--video-cams", nargs="*", default=[],
                   help="also dump every frame of these cameras for the requested episodes (gray 320x240 npz)")
    p.add_argument("--color-step", type=int, default=0,
                   help="also dump every Nth frame of all 3 cameras in colour (320x240 BGR npz) for the episodes")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    info = json.load(open(os.path.join(a.dataset, "meta/info.json")))
    fps = info["fps"]
    eps = pd.read_parquet(os.path.join(a.dataset, "meta/episodes/chunk-000/file-000.parquet"))
    data = pd.concat(pd.read_parquet(os.path.join(a.dataset, "data", c, f))
                     for c in sorted(os.listdir(os.path.join(a.dataset, "data")))
                     for f in sorted(os.listdir(os.path.join(a.dataset, "data", c))))
    os.makedirs(a.out, exist_ok=True)

    def grab(ep_row, cam, frame_idx, path):
        chunk = int(ep_row[f"videos/observation.images.{cam}/chunk_index"])
        file = int(ep_row[f"videos/observation.images.{cam}/file_index"])
        t0 = float(ep_row[f"videos/observation.images.{cam}/from_timestamp"])
        video = os.path.join(a.dataset, info["video_path"].format(
            video_key=f"observation.images.{cam}", chunk_index=chunk, file_index=file))
        t = t0 + frame_idx / fps
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.4f}", "-i", video, "-frames:v", "1", path],
                       check=True)

    episodes = sorted(set(a.episodes) | (set(eps.episode_index) if a.all_first_frames else set()))
    for ep in episodes:
        row = eps[eps.episode_index == ep].iloc[0]
        d = data[data.episode_index == ep].sort_values("frame_index")
        np.savez(os.path.join(a.out, f"ep{ep:03d}.npz"),
                 state=np.stack(d["observation.state"].values).astype(np.float32),
                 action=np.stack(d["action"].values).astype(np.float32))
        for cam in a.video_cams if ep in a.episodes else []:
            chunk = int(row[f"videos/observation.images.{cam}/chunk_index"])
            file = int(row[f"videos/observation.images.{cam}/file_index"])
            t0 = float(row[f"videos/observation.images.{cam}/from_timestamp"])
            video = os.path.join(a.dataset, info["video_path"].format(
                video_key=f"observation.images.{cam}", chunk_index=chunk, file_index=file))
            raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-i", video, "-frames:v", str(len(d)),
                                  "-vf", "scale=320:240,format=gray", "-f", "rawvideo", "-"],
                                 check=True, capture_output=True).stdout
            frames_arr = np.frombuffer(raw, np.uint8).reshape(-1, 240, 320)
            np.savez_compressed(os.path.join(a.out, f"ep{ep:03d}_{cam}_video.npz"), frames=frames_arr)
            print(f"  {cam}: {len(frames_arr)} frames")
        if a.color_step and ep in a.episodes:
            out = {}
            for cam in CAMS:
                chunk = int(row[f"videos/observation.images.{cam}/chunk_index"])
                file = int(row[f"videos/observation.images.{cam}/file_index"])
                t0 = float(row[f"videos/observation.images.{cam}/from_timestamp"])
                video = os.path.join(a.dataset, info["video_path"].format(
                    video_key=f"observation.images.{cam}", chunk_index=chunk, file_index=file))
                raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-i", video, "-frames:v", str(len(d)),
                                      "-vf", "scale=320:240", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
                                     check=True, capture_output=True).stdout
                out[cam] = np.frombuffer(raw, np.uint8).reshape(-1, 240, 320, 3)[::a.color_step]
            n = min(len(v) for v in out.values())
            np.savez_compressed(os.path.join(a.out, f"ep{ep:03d}_color.npz"),
                                index=np.arange(n) * a.color_step, **{k: v[:n] for k, v in out.items()})
        frames = a.frames if ep in a.episodes else [0]
        cams = CAMS if ep in a.episodes else ["cam_head"]
        for f in frames:
            if f >= len(d):
                continue
            for cam in cams:
                grab(row, cam, f, os.path.join(a.out, f"ep{ep:03d}_f{f:04d}_{cam}.png"))
        print(f"episode {ep}: {len(d)} frames, dumped {frames}")


if __name__ == "__main__":
    main()
