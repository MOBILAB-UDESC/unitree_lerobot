"""Back up LeRobot checkpoints to one Hugging Face model repo while training runs.

Waits for each requested checkpoint of a run to be written, then uploads it to the same repo: the policy
(pretrained_model/) goes to the repo root, so stock LeRobot loads it with the repo name, and training_state/
next to it, so the run can be resumed from the Hub. The root is always the newest checkpoint; every checkpoint
is also kept in `checkpoints/040000/pretrained_model/`, ... and gets a tag `step-040000`, ...
Started by `train_act_docker.sh backup`.

    python push_checkpoints.py /data/outputs/act-box-sim mobilab-udesc/act-g1-box-into-box-sim \
        --steps 40000 60000 80000 100000

The Hugging Face token is read from HF_TOKEN or from $HF_HOME/token; the script waits until one is there.
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi, get_token


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="lerobot-train output folder (contains checkpoints/)")
    p.add_argument("repo", help="Hub model repo, e.g. mobilab-udesc/act-g1-box-into-box-sim")
    p.add_argument("--steps", type=int, nargs="+", required=True, help="checkpoints to upload")
    p.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    p.add_argument("--poll", type=float, default=60, help="seconds between checks")
    a = p.parse_args()

    while not get_token():
        print("no Hugging Face token yet (HF_TOKEN or $HF_HOME/token), waiting", flush=True)
        time.sleep(a.poll)
    api = HfApi()
    api.create_repo(a.repo, repo_type="model", private=not a.public, exist_ok=True)
    print(f"logged in as {api.whoami()['name']}, uploading to {a.repo}", flush=True)

    for step in sorted(a.steps):
        tag, folder = f"step-{step:06d}", f"checkpoints/{step:06d}/pretrained_model"
        tagged = tag in [t.name for t in api.list_repo_refs(a.repo).tags]
        if tagged and f"{folder}/model.safetensors" in api.list_repo_files(a.repo):
            print(f"{tag}: already on the Hub", flush=True)
            continue
        ckpt = Path(a.run_dir) / "checkpoints" / f"{step:06d}"
        print(f"{tag}: waiting for {ckpt}", flush=True)
        # training_step.json is written last; the extra wait covers a slow network disk.
        while not (ckpt / "training_state" / "training_step.json").exists():
            time.sleep(a.poll)
            tagged = False
        if not tagged:
            time.sleep(a.poll)
        for attempt in range(1, 11):
            try:
                if not tagged:
                    api.upload_folder(repo_id=a.repo, folder_path=ckpt / "pretrained_model",
                                      commit_message=f"policy at step {step}")
                    api.upload_folder(repo_id=a.repo, folder_path=ckpt / "training_state", path_in_repo="training_state",
                                      commit_message=f"training state at step {step}")
                    api.create_tag(a.repo, tag=tag, exist_ok=True)
                # The root only holds the newest checkpoint: keep every one in its own folder as well.
                api.upload_folder(repo_id=a.repo, folder_path=ckpt / "pretrained_model", path_in_repo=folder,
                                  commit_message=f"keep the policy at step {step} in {folder}")
                print(f"{tag}: uploaded", flush=True)
                break
            except Exception as err:
                print(f"{tag}: upload attempt {attempt} failed: {err}", flush=True)
                time.sleep(a.poll * attempt)
    print("done", flush=True)


if __name__ == "__main__":
    main()
