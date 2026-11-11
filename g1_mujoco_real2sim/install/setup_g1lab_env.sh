#!/usr/bin/env bash
# Build the `g1lab` conda env: LeRobot 0.4.1 recording/training backbone for the G1 sim.
# Usage: install/setup_g1lab_env.sh   (re-running is safe; it reuses an existing env)
set -eo pipefail

# ~/.bashrc sources ROS 2 Jazzy; its paths break conda Python and CycloneDDS.
unset PYTHONPATH LD_LIBRARY_PATH CYCLONEDDS_URI

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${G1LAB_ENV:-g1lab}"   # override to build a throwaway copy

# Same fallback as scripts/env.sh: install.sh may have just installed Miniforge, and this terminal's
# PATH does not have it yet.
if ! command -v conda >/dev/null 2>&1 && [ -x "$HOME/miniforge3/bin/conda" ]; then
  export PATH="$HOME/miniforge3/bin:$PATH"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV"; then
    conda create -y -n "$ENV" -c conda-forge python=3.10
fi
conda activate "$ENV"
unset PYTHONPATH LD_LIBRARY_PATH CYCLONEDDS_URI

# IK stack (same versions xr_teleoperate uses).
conda install -y -c conda-forge pinocchio=3.1.0 casadi

# CPU torch first so lerobot does not pull the CUDA wheels.
pip install "torch>=2.2.1,<2.8.0" "torchvision>=0.21.0,<0.23.0" \
    --index-url https://download.pytorch.org/whl/cpu

# lerobot 0.4.1 is what unitree_lerobot pins (commit a5b29d43).
pip install "lerobot==0.4.1" \
    "mujoco==3.17.0" pyzmq pyyaml logging_mp opencv-python mcp pytest meshcat matplotlib scipy

# conda-forge pinocchio 3.1.0 is only built against numpy 1.x ("_ARRAY_API not found" + segfault on
# numpy 2). lerobot itself works with numpy 1.26; only rerun-sdk>=0.24 (lerobot's viz extra) declares
# numpy>=2, and it still imports fine. A single resolve is impossible, so downgrade afterwards
# (pip prints a rerun-sdk conflict warning; that is expected). opencv 4.12+ needs numpy 2.
pip install "numpy<2" "opencv-python<4.12" "opencv-python-headless<4.12"

TP="${G1_THIRD_PARTY:-$ROOT/third_party}"  # cloned by install/install.sh
pip install -e "$TP/unitree_sdk2_python"
pip install -e "$TP/xr_teleoperate/teleop/teleimager" --no-deps

# vuer (if anything pulls it) breaks with newer params-proto.
if pip show vuer >/dev/null 2>&1; then
    pip install "params-proto==2.13.2"
fi

python - <<'EOF'
import lerobot, mujoco, pinocchio, casadi, zmq, cv2, mcp, unitree_sdk2py, teleimager
from importlib.metadata import version
print("lerobot", version("lerobot"), "| mujoco", mujoco.__version__, "| pinocchio", pinocchio.__version__)
import av
codecs = av.codecs_available
assert "libsvtav1" in codecs, "PyAV build lacks libsvtav1 (needed for av1 video like the real dataset)"
print("PyAV", av.__version__, "with libsvtav1: OK")
EOF

echo "g1lab env ready. Run tests with:"
echo "  source scripts/env.sh $ENV && python -m pytest 04_recording -q"
