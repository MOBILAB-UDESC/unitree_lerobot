#!/usr/bin/env bash
# Train an ACT policy on a G1 LeRobot dataset with lerobot-train (lerobot 0.4.1, as pinned by unitree_lerobot).
#
#   05_training/train_act.sh --root datasets/mixed_waist --repo-id local/mixed_waist --output outputs/act_mixed
#   05_training/train_act.sh --root /tmp/sim --repo-id local/sim --output /tmp/act_smoke \
#       --steps 100 --batch-size 2 --device cpu          # CPU smoke test (this PC has no GPU)
#   ... -- --policy.chunk_size=50 --policy.n_action_steps=50   # anything after -- goes to lerobot-train
#
# Real training runs on a GPU machine (see 05_training/docker/train_act_docker.sh), with the same
# lerobot-train command and --policy.device=cuda; copy the dataset dir there (or pull it from the Hub, see 04_recording/hf/) and pass the
# same flags. The checkpoint to deploy with 05_training/run_act.py is
#   <output>/checkpoints/last/pretrained_model   (config.json, model.safetensors, policy_*processor*.json + stats)
#
# The policy's input/output size comes from the dataset: a 15-joint (ArmsWaist) dataset gives a policy that
# also drives the waist, a 14-joint one only the arms. Mix real (pad_waist'ed) + sim with merge_datasets.py.
# Never pushes anything: --policy.push_to_hub=false and no --dataset.push_to_hub.
set -eo pipefail

usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

ROOT_DIR="" REPO_ID="" OUTPUT="" STEPS=100000 BATCH=8 DEVICE=cuda WORKERS="" SAVE_FREQ="" LOG_FREQ="" JOB=""
EXTRA=() RESUME=false
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT_DIR="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size) BATCH="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --num-workers) WORKERS="$2"; shift 2 ;;
    --save-freq) SAVE_FREQ="$2"; shift 2 ;;
    --log-freq) LOG_FREQ="$2"; shift 2 ;;
    --job-name) JOB="$2"; shift 2 ;;
    --resume) RESUME=true; shift ;;
    -h|--help) usage ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done
[ -n "$REPO_ID" ] && [ -n "$OUTPUT" ] || { echo "--repo-id and --output are required" >&2; usage 1; }
if [ -n "$ROOT_DIR" ] && [ ! -f "$ROOT_DIR/meta/info.json" ]; then
  echo "$ROOT_DIR is not a LeRobot dataset (no meta/info.json)" >&2; exit 1
fi
if [ -e "$OUTPUT" ] && [ "$RESUME" != true ]; then
  echo "$OUTPUT exists; pick a new --output or pass --resume" >&2; exit 1
fi

# Use the g1lab env unless lerobot-train is already on PATH (e.g. inside a LeRobot Docker image).
if ! command -v lerobot-train >/dev/null 2>&1; then
  source "$(dirname "${BASH_SOURCE[0]}")/../scripts/env.sh" "${G1LAB_ENV:-g1lab}"
fi

# CPU: few dataloader workers (video decoding is the bottleneck and RAM is small); GPU: lerobot defaults.
if [ "$DEVICE" = cpu ]; then
  WORKERS="${WORKERS:-2}"; SAVE_FREQ="${SAVE_FREQ:-$STEPS}"; LOG_FREQ="${LOG_FREQ:-10}"
fi
ARGS=(
  --dataset.repo_id="$REPO_ID"
  --policy.type=act
  --policy.device="$DEVICE"
  --policy.push_to_hub=false
  --output_dir="$OUTPUT"
  --job_name="${JOB:-act_$(basename "$REPO_ID")}"
  --steps="$STEPS"
  --batch_size="$BATCH"
  --wandb.enable=false
)
[ -n "$ROOT_DIR" ] && ARGS+=(--dataset.root="$ROOT_DIR")
[ -n "$WORKERS" ] && ARGS+=(--num_workers="$WORKERS")
[ -n "$SAVE_FREQ" ] && ARGS+=(--save_freq="$SAVE_FREQ")
[ -n "$LOG_FREQ" ] && ARGS+=(--log_freq="$LOG_FREQ")
[ "$RESUME" = true ] && ARGS+=(--resume=true --config_path="$OUTPUT/checkpoints/last/pretrained_model/train_config.json")

echo "+ lerobot-train ${ARGS[*]} ${EXTRA[*]}"
exec lerobot-train "${ARGS[@]}" "${EXTRA[@]}"
