#!/usr/bin/env bash
# Run a remote VLA policy on the G1 arms (sim by default). All args go to g1lab.remote_vla, e.g.:
#   scripts/run_vla.sh --backend dream --prompt "put the wrapped box inside the cardboard box"
#   scripts/run_vla.sh --backend openpi --url ws://gpu-box:8000 --episodes 5
#   scripts/run_vla.sh --backend gr00t --url tcp://gpu-box:5555 --domain 3 --image-port 61000
# Env: DREAM_API_KEY (dream), GR00T_API_TOKEN / OPENPI_API_KEY (optional),
#      G1LAB_ENV conda env (default g1lab: has lerobot, needed for --record).
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "${G1LAB_ENV:-g1lab}"
cd "$REPO"
exec python -u -m g1lab.remote_vla "$@"
