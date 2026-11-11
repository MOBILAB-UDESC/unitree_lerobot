#!/usr/bin/env bash
# PRINT (never run) the Hugging Face Hub commands that publish the lab's G1 LeRobot datasets.
#
#   04_recording/hf/upload_datasets.sh "<LAB_HF_ORG>"          # e.g. mobilab-udesc
#   REAL_DIR=... REAL_WAIST_DIR=... SIM_DIR=... 04_recording/hf/upload_datasets.sh mobilab-udesc
#
# For each dataset that exists locally it renders the card (hf/cards/<name>/README.md, with <LAB_HF_ORG> and
# the episode/frame counts filled in) into a staging dir and prints:
#   hf repo create      (private dataset repo)
#   hf upload           (the dataset dir; images/ is lerobot's temporary PNG dir and is skipped)
#   hf upload           (the card as README.md)
#   hf repo tag create  v3.0 - LeRobotDataset(repo_id) refuses a Hub dataset without its codebase-version tag
# Review them, then copy-paste. You must be logged in (`hf auth login`, or HF_TOKEN) with write access to the
# org. Repos are created --private; make them public on the Hub once the license/card is confirmed.
#
# Datasets (defaults are this repo's gitignored datasets/ dir):
#   REAL_DIR        real 52-episode pick_cube set, 14 joints     -> <org>/g1_pick_cube_real
#   REAL_WAIST_DIR  the same padded to 15 joints (g1lab.pad_waist) -> <org>/g1_pick_cube_real_waist
#   SIM_DIR         scripted sim box-into-box set, 15 joints      -> <org>/g1_box_into_box_sim_scripted
set -eo pipefail

ORG="${1:-<LAB_HF_ORG>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
REAL_DIR="${REAL_DIR:-$REPO/datasets/pick_cube_real}"
REAL_WAIST_DIR="${REAL_WAIST_DIR:-$REPO/datasets/real_waist}"
SIM_DIR="${SIM_DIR:-$REPO/datasets/sim_scripted_waist}"
STAGE="${STAGE:-${TMPDIR:-/tmp}/g1_hf_cards}"
VERSION=v3.0

if [ "$ORG" = "<LAB_HF_ORG>" ]; then
  echo "# NOTE: no org given - commands use the placeholder <LAB_HF_ORG> (usage: $0 <org>)"
fi
if [ -n "$HF_TOKEN" ] || [ -s "${HF_HOME:-$HOME/.cache/huggingface}/token" ]; then
  echo "# HF login: a token is present (check the account with: hf auth whoami)"
else
  echo "# HF login: NO token found - run 'hf auth login' (write token for $ORG) before the commands below"
fi
command -v hf >/dev/null 2>&1 || echo "# 'hf' not on PATH: source scripts/env.sh g1lab (huggingface_hub CLI)"

render_card() {  # name src_dir -> path of the rendered README.md
  local name="$1" dir="$2" out="$STAGE/$1/README.md"
  mkdir -p "$STAGE/$name"
  python3 - "$HERE/cards/$name/README.md" "$dir/meta/info.json" "$ORG" "$out" "$dir" <<'EOF'
import json, sys
from pathlib import Path
card, info_path, org, out, root = sys.argv[1:]
info = json.loads(Path(info_path).read_text())
task = ""
try:  # tasks.parquet index = task string (lerobot v3.0)
    import pandas as pd
    task = "; ".join(pd.read_parquet(Path(root) / "meta" / "tasks.parquet").index.astype(str))
except Exception:
    task = "see meta/tasks.parquet"
text = Path(card).read_text()
for k, v in {"<LAB_HF_ORG>": org, "{{TOTAL_EPISODES}}": str(info["total_episodes"]),
             "{{TOTAL_FRAMES}}": str(info["total_frames"]), "{{TASK}}": task}.items():
    text = text.replace(k, v)
Path(out).write_text(text)
EOF
  echo "$out"
}

print_dataset() {  # name src_dir expected_robot_type
  local name="$1" dir="$2" robot="$3" repo_id="$ORG/$1"
  echo
  echo "# ---- $repo_id  <-  $dir"
  if [ ! -f "$dir/meta/info.json" ]; then
    echo "# SKIPPED: $dir is not a LeRobot dataset (set ${4} to its path)"
    return
  fi
  local summary
  summary=$(python3 -c "import json,sys; i=json.load(open(sys.argv[1])); print(i['codebase_version'], i['robot_type'], i['total_episodes'], i['total_frames'])" "$dir/meta/info.json")
  read -r cv rt eps frames <<<"$summary"
  echo "# $eps episodes, $frames frames, $rt, $cv, $(du -sh "$dir" | cut -f1)"
  [ "$cv" = "$VERSION" ] || echo "# WARNING: codebase_version $cv != $VERSION"
  [ "$rt" = "$robot" ] || echo "# WARNING: robot_type $rt, expected $robot"
  local card
  card=$(render_card "$name" "$dir")
  printf 'hf repo create %q --repo-type dataset --private\n' "$repo_id"
  printf 'hf upload %q %q . --repo-type dataset --exclude "images/*" --commit-message %q\n' \
    "$repo_id" "$dir" "Add $name (LeRobot $VERSION)"
  printf 'hf upload %q %q README.md --repo-type dataset --commit-message "Add dataset card"\n' "$repo_id" "$card"
  printf 'hf repo tag create %q %q --repo-type dataset\n' "$repo_id" "$VERSION"
}

echo "# Review, then run these yourself. Nothing has been uploaded."
print_dataset g1_pick_cube_real "$REAL_DIR" Unitree_G1_ArmsOnly_3Cam REAL_DIR
print_dataset g1_pick_cube_real_waist "$REAL_WAIST_DIR" Unitree_G1_ArmsWaist_3Cam REAL_WAIST_DIR
print_dataset g1_box_into_box_sim_scripted "$SIM_DIR" Unitree_G1_ArmsWaist_3Cam SIM_DIR
echo
echo "# Check a published set (downloads it): python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; print(D('$ORG/g1_pick_cube_real_waist'))\""
