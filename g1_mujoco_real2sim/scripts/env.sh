# Sourced by every script: conda env + this repo's stage folders on PYTHONPATH.
#   source scripts/env.sh [ENV]   (from a script that has its own args, pass "" to use G1_ENV)
#   G1_ENV        conda env for sim + teleop (default g1sim)
#   G1_THIRD_PARTY  where install.sh cloned xr_teleoperate / unitree_mujoco / ... (default third_party/)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Keep ROS 2 (often sourced in ~/.bashrc) out: its PYTHONPATH shadows conda's pinocchio and its
# CycloneDDS libs/config override the loopback DDS setup used here.
unset PYTHONPATH LD_LIBRARY_PATH CYCLONEDDS_URI CMAKE_PREFIX_PATH AMENT_PREFIX_PATH
if ! command -v conda >/dev/null 2>&1 && [ -x "$HOME/miniforge3/bin/conda" ]; then
  export PATH="$HOME/miniforge3/bin:$PATH"
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${1:-${G1_ENV:-g1sim}}"
# One math thread per process: the sim, IK and recorders are many small real-time processes; with the
# default (one OpenMP/BLAS thread per core, spinning while idle) they starve each other and the sim drops
# to a few fps. Training scripts raise this again.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export G1_THIRD_PARTY="${G1_THIRD_PARTY:-$REPO/third_party}"
export PYTHONPATH="$REPO/01_teleop/g1sim:$REPO/02_wrist_cameras:$REPO/03_real2sim:$REPO/03_real2sim/tools:$REPO/04_recording:$REPO/05_training"
