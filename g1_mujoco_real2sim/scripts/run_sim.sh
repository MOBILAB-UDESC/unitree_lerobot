#!/usr/bin/env bash
# Start the MuJoCo G1 that xr_teleoperate --sim talks to. Extra args go to g1_xr_sim.py:
#   scripts/run_sim.sh                          stage 1: demo table, stereo head camera
#   scripts/run_sim.sh --addon wrist_cameras    stage 2: + gloved hands and D405 wrist cameras
#   scripts/run_sim.sh --scene lab              stage 3: the real2sim lab task (includes the wrist cameras)
#   scripts/run_sim.sh --scene lab --randomize  ... with randomised objects / lights for data collection
set -e
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" ""  # "" = env from G1_ENV (default g1sim), not our args
exec python -u "$REPO/01_teleop/g1sim/g1_xr_sim.py" "$@"
