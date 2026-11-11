#!/usr/bin/env bash
# Train ACT on a GPU machine that has Docker with NVIDIA GPU support (a server or a cluster node).
# Run this on that machine, from a clone of this repo.
#
#   05_training/docker/train_act_docker.sh setup    # once per node: Docker image + LeRobot 0.4.1 venv on /data
#   05_training/docker/train_act_docker.sh smoke    # 200 steps: downloads the dataset, proves the GPU pipeline
#   05_training/docker/train_act_docker.sh train    # the real run, detached (survives logout)
#   05_training/docker/train_act_docker.sh status   # last log lines + checkpoints
#   05_training/docker/train_act_docker.sh backup   # optional: upload checkpoints to the Hub as they appear
#   05_training/docker/train_act_docker.sh stop     # stop and remove the training container
#
# backup needs PUSH_REPO and a write token, stored once on that machine without showing it:
#   read -rsp "HF token: " T && (umask 077; printf %s "$T" > $SCRATCH/.cache/huggingface/token)
# It uploads steps BACKUP_STEPS (default "40000 60000 80000 100000") to PUSH_REPO: main = newest, one tag
# step-040000, ... per checkpoint, training_state/ included so the run can be resumed from the Hub.
#
# Settings (environment variables):
#   DATASET     Hub dataset                  (default mobilab-udesc/g1_box_into_box_sim_scripted)
#   NAME        run / container name         (default act-box-sim)
#   STEPS, BATCH, SAVE_FREQ                  (default 100000, 16, 20000)
#   SCRATCH     folder for everything the run writes (default ~/lerobot_data); on a cluster use a disk
#               that the login node also sees, e.g. SCRATCH=/scratch/$USER
#   RESUME=1    continue NAME from its last checkpoint
#   EXTRA       more lerobot-train flags, e.g. EXTRA="--policy.chunk_size=50 --policy.n_action_steps=50"
#   PUSH_REPO   e.g. mobilab-udesc/act-g1-box-into-box-sim: also upload the final model to the Hub when
#               training ends (needs HF_TOKEN with write access). Default: nothing is uploaded.
#   HF_TOKEN    only for private datasets or PUSH_REPO. Set it without echo: read -s HF_TOKEN && export HF_TOKEN
#
# Why a venv instead of the image's own LeRobot: huggingface/lerobot-gpu:latest follows LeRobot main, but
# 05_training/run_act.py and Unitree's tools use LeRobot 0.4.1, and a newer checkpoint may not load there.
# The venv has LeRobot 0.4.1 with a CUDA 12.8 PyTorch (the RTX PRO 6000 Blackwell needs it); the image only
# supplies the system libraries. Everything lives in $SCRATCH (= /data in the container).
# Checkpoints stay in $SCRATCH/outputs/<NAME>/checkpoints.
set -eo pipefail

DATASET="${DATASET:-mobilab-udesc/g1_box_into_box_sim_scripted}"
NAME="${NAME:-act-box-sim}"
STEPS="${STEPS:-100000}" BATCH="${BATCH:-16}" SAVE_FREQ="${SAVE_FREQ:-20000}"
IMAGE="${IMAGE:-huggingface/lerobot-gpu:latest}"
SCRATCH="${SCRATCH:-$HOME/lerobot_data}"
VENV=/data/venvs/lerobot041
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

mkdir -p "$SCRATCH/outputs"
RUN=(docker run --gpus all --shm-size 16gb
  --user "$(id -u):$(id -g)"
  -v "$SCRATCH":/data -w /data
  -e HOME=/data
  -e HF_HOME=/data/.cache/huggingface
  -e HF_LEROBOT_HOME=/data/.cache/huggingface/lerobot
  -e TORCH_HOME=/data/.cache/torch
  -e TRITON_CACHE_DIR=/data/.cache/triton
  -e UV_CACHE_DIR=/data/.cache/uv -e UV_HTTP_TIMEOUT=900 -e UV_LINK_MODE=copy
  -e HF_TOKEN)

train_cmd() {  # output_dir steps batch save_freq
  local push="--policy.push_to_hub=false"
  [ -n "$PUSH_REPO" ] && [ "$2" -gt 1000 ] && push="--policy.push_to_hub=true --policy.repo_id=$PUSH_REPO"
  echo "$VENV/bin/lerobot-train --dataset.repo_id=$DATASET --policy.type=act --policy.device=cuda" \
    "$push --output_dir=$1 --job_name=$NAME --steps=$2 --batch_size=$3 --save_freq=$4" \
    "--wandb.enable=false $EXTRA"
}

case "${1:-}" in
  setup)
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    docker pull "$IMAGE"
    docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE" | tee "$SCRATCH/outputs/image_digest.txt"
    "${RUN[@]}" --rm "$IMAGE" bash -c "
      set -e
      if [ ! -x $VENV/bin/python ]; then
        if command -v uv >/dev/null; then uv venv --python 3.10 $VENV; else python3 -m venv $VENV; fi
      fi
      if command -v uv >/dev/null; then PIP='uv pip install --python $VENV/bin/python'; else PIP='$VENV/bin/pip install'; fi
      \$PIP 'torch>=2.7,<2.8' 'torchvision>=0.22,<0.23' --index-url $TORCH_INDEX
      \$PIP 'lerobot==0.4.1'   # from PyPI; keeps the CUDA 12.8 torch installed above
      $VENV/bin/python -c \"import torch, lerobot; print('lerobot', lerobot.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')\"
    "
    ;;
  smoke)
    rm -rf "$SCRATCH/outputs/$NAME-smoke"
    "${RUN[@]}" --rm "$IMAGE" bash -c "$(train_cmd "/data/outputs/$NAME-smoke" 200 8 200)"
    ls "$SCRATCH/outputs/$NAME-smoke/checkpoints/last/pretrained_model"
    echo "SMOKE OK"
    ;;
  train)
    OUT="/data/outputs/$NAME"
    if [ "$RESUME" = 1 ]; then
      CMD="$VENV/bin/lerobot-train --resume=true --config_path=$OUT/checkpoints/last/pretrained_model/train_config.json"
    elif [ -e "$SCRATCH/outputs/$NAME" ]; then
      echo "$SCRATCH/outputs/$NAME exists: pick another NAME, or RESUME=1 to continue it" >&2; exit 1
    else
      CMD="$(train_cmd "$OUT" "$STEPS" "$BATCH" "$SAVE_FREQ")"
    fi
    docker rm "$NAME" >/dev/null 2>&1 || true
    "${RUN[@]}" -d --name "$NAME" "$IMAGE" bash -c "exec $CMD"
    echo "started container $NAME. Follow it:  docker logs -f $NAME   (Ctrl-C stops following, not training)"
    echo "checkpoints: $SCRATCH/outputs/$NAME/checkpoints/"
    ;;
  status)
    docker ps -a --filter "name=^$NAME\$" --format '{{.Names}}: {{.Status}}'
    docker logs --tail 400 "$NAME" 2>&1 | grep -E "step:|Error|error|Traceback" | tail -8 || true
    ls "$SCRATCH/outputs/$NAME/checkpoints" 2>/dev/null || echo "no checkpoints yet"
    ;;
  backup)
    [ -n "$PUSH_REPO" ] || { echo "set PUSH_REPO, e.g. PUSH_REPO=mobilab-udesc/act-g1-box-into-box-sim" >&2; exit 1; }
    cp "$(dirname "$0")/push_checkpoints.py" "$SCRATCH/outputs/"
    docker rm -f "$NAME-backup" >/dev/null 2>&1 || true
    "${RUN[@]}" -d --name "$NAME-backup" "$IMAGE" $VENV/bin/python /data/outputs/push_checkpoints.py \
      "/data/outputs/$NAME" "$PUSH_REPO" --steps ${BACKUP_STEPS:-40000 60000 80000 100000}
    echo "started container $NAME-backup. Follow it:  docker logs -f $NAME-backup"
    ;;
  stop)
    docker stop "$NAME" && docker rm "$NAME"
    docker rm -f "$NAME-backup" >/dev/null 2>&1 || true
    ;;
  *)
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
