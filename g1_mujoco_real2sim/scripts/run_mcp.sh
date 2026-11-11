#!/usr/bin/env bash
# Launch the G1 MCP server (04_recording/g1lab/mcp_server.py) over stdio, so an AI assistant (e.g. Claude
# Code) can look through the robot's cameras and move it. MCP clients start this from any folder, and
# nothing but MCP traffic may reach stdout (logs go to stderr).
#   scripts/run_mcp.sh [--target sim|real] [--domain N] [--image-port P] [--dataset-root DIR] ...
# Env: G1_MCP_ENV  conda env (default g1lab: has lerobot, needed to record episodes)
#      G1_MCP_LOG  append the server's stderr log to this file instead
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" "${G1_MCP_ENV:-g1lab}" >&2
if [ -n "$G1_MCP_LOG" ]; then exec 2>>"$G1_MCP_LOG"; fi
cd "$REPO"
PYTHONUNBUFFERED=1 exec python -u -m g1lab.mcp_server "$@"
