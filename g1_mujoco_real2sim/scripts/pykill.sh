#!/usr/bin/env bash
# Kill python processes whose command line contains $1 (never matches shells running this pattern).
# Usage: scripts/pykill.sh "g1_xr_sim.py" [--list]
pids=$(ps -eo pid=,comm=,args= | awk -v pat="$1" '$2 ~ /^python/ && index($0, pat) {print $1}')
[ -z "$pids" ] && { echo "no python process matching: $1"; exit 0; }
if [ "$2" = "--list" ]; then ps -o pid=,args= -p $pids; else kill $pids && echo "killed: $pids"; fi
