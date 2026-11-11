#!/usr/bin/env bash
# One-shot installer for stages 1-3 (MuJoCo G1 + Quest 3 + xr_teleoperate) on a fresh Ubuntu PC.
# Safe to re-run. Installs Miniforge (conda) if missing, creates the conda env 'g1sim', clones the
# upstream repos into third_party/ at the exact commits this repo was tested with, and makes the HTTPS
# certificate the Quest needs. Stages 4-5 (LeRobot) use a second env: install/setup_g1lab_env.sh.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${G1_ENV:-g1sim}"
mkdir -p "$ROOT/third_party"
cd "$ROOT/third_party"

# Versions tested together (2026-11-06).
XR_TELEOPERATE_COMMIT=817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b
UNITREE_MUJOCO_COMMIT=1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d
UNITREE_SDK2_PYTHON_COMMIT=814556d15970dd2ecf1c9984e845ca02ab07e206
UNITREE_LEROBOT_COMMIT="${UNITREE_LEROBOT_COMMIT:-main}"  # only its data tools are used (stage 4)

step() { echo; echo "==== $* ===="; }

# The kit's env must not see ROS 2 (if this PC sources it in ~/.bashrc).
unset PYTHONPATH LD_LIBRARY_PATH CYCLONEDDS_URI CMAKE_PREFIX_PATH AMENT_PREFIX_PATH

step "1/6 conda"
if ! command -v conda >/dev/null 2>&1; then
  if [ -x "$HOME/miniforge3/bin/conda" ]; then
    export PATH="$HOME/miniforge3/bin:$PATH"
  else
    echo "conda not found - installing Miniforge to ~/miniforge3"
    curl -fsSL -o /tmp/Miniforge3.sh \
      "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash /tmp/Miniforge3.sh -b -p "$HOME/miniforge3"
    rm -f /tmp/Miniforge3.sh
    export PATH="$HOME/miniforge3/bin:$PATH"
    "$HOME/miniforge3/bin/conda" init bash >/dev/null
  fi
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

step "2/6 conda env '$ENV_NAME' (python 3.10 + pinocchio 3.1.0)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10 pinocchio=3.1.0 numpy=1.26.4 -c conda-forge
fi
conda activate "$ENV_NAME"

step "3/6 upstream repos (pinned)"
clone_at() {  # dir url commit
  [ -d "$1/.git" ] || git clone "$2" "$1"
  git -C "$1" fetch --quiet origin
  git -C "$1" checkout --quiet "$3"
}
clone_at xr_teleoperate https://github.com/unitreerobotics/xr_teleoperate.git "$XR_TELEOPERATE_COMMIT"
git -C xr_teleoperate submodule update --init --depth 1
clone_at unitree_mujoco https://github.com/unitreerobotics/unitree_mujoco.git "$UNITREE_MUJOCO_COMMIT"
clone_at unitree_sdk2_python https://github.com/unitreerobotics/unitree_sdk2_python.git "$UNITREE_SDK2_PYTHON_COMMIT"
clone_at unitree_lerobot https://github.com/unitreerobotics/unitree_lerobot.git "$UNITREE_LEROBOT_COMMIT"

step "4/6 python packages"
# vuer 0.0.60 breaks with params-proto 3.x ("cannot import name 'Flag'"), so pin 2.x.
pip install "mujoco==3.17.0" "params-proto==2.13.2" pyzmq pyyaml logging_mp pytest pillow
(cd xr_teleoperate/teleop/teleimager && pip install -e . --no-deps)
(cd xr_teleoperate/teleop/televuer && pip install -e .)
pip install -r xr_teleoperate/requirements.txt
(cd unitree_sdk2_python && pip install -e .)
pip install "params-proto==2.13.2"  # televuer's vuer dependency may have pulled 3.x back in

step "5/6 HTTPS certificate for the Quest (~/.config/xr_teleoperate)"
IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="src") print $(i+1)}')"
CERT_DIR="$HOME/.config/xr_teleoperate"
if [ -f "$CERT_DIR/cert.pem" ]; then
  echo "already present - keeping it"
else
  mkdir -p "$CERT_DIR"
  openssl req -x509 -nodes -days 3650 -newkey rsa:2048 -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
    -subj "/CN=g1-sim-xr" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${IP:-127.0.0.1}"
fi

step "6/6 check"
python -c "import mujoco, pinocchio, cv2, zmq, vuer, televuer, teleimager, unitree_sdk2py; print('imports OK')"
cd "$ROOT"
SCENE=demo; [ -d 03_real2sim ] && SCENE=lab
python -c "import sys; sys.path[:0] = ['01_teleop/g1sim', '02_wrist_cameras', '03_real2sim']; import g1_xr_sim as s; \
a = s.parse_args(['--scene', '$SCENE']); m, c = s.build_model(a); print('$SCENE scene OK:', sorted(c))"

echo
echo "Install finished. Next (see README.md):"
echo "  sudo ufw allow 8012 && sudo ufw allow 8080      # only if the firewall is on"
echo "  scripts/run_sim.sh                               # terminal 1"
echo "  scripts/run_teleop.sh --controller               # terminal 2, then open http://${IP}:8080 on the Quest"
echo "  install/setup_g1lab_env.sh                       # later, for recording / training (stages 4-5)"
