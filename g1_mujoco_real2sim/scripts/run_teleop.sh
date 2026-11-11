#!/usr/bin/env bash
# Start Unitree xr_teleoperate against the MuJoCo sim (DDS domain 1 on loopback, images from localhost),
# with the wrist camera views added to the headset display (02_wrist_cameras/teleop_views.py).
#   scripts/run_teleop.sh               hand tracking, arms only (like debug mode on the real robot)
#   scripts/run_teleop.sh --controller  Quest controllers: arms follow the controllers, left stick walks,
#                                       right stick twists the hip (xr_teleoperate --input-mode controller --motion)
#   scripts/run_teleop.sh --plain       unmodified teleop_hand_and_arm.py (no wrist views)
#   --wrist-layout pip|side|off         wrist views at start (default pip; toggle in VR: left X / the
#                                       "WRIST VIEW" button, or the button at the bottom-left of the page)
# Any other args go to teleop_hand_and_arm.py, e.g.:  scripts/run_teleop.sh --controller --record
# Env: G1_ENV conda env (default g1sim)
set -e
# "" so env.sh doesn't take our first argument for the env name.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" ""
IP="$(ip -4 route get 1.1.1.1 2>/dev/null | awk '{for(i=1;i<NF;i++) if($i=="src") print $(i+1)}')"
IP="${IP:-127.0.0.1}"
python "$REPO/scripts/quest_link.py" "$IP" 8080 &
LINK_PID=$!
trap 'kill $LINK_PID 2>/dev/null' EXIT
echo "On the Quest 3 browser open:  http://${IP}:8080"
echo "  (redirects to https://${IP}:8012/?ws=wss://${IP}:8012)"
PLAIN=0
MODE_ARGS=()
ARGS=()
for a in "$@"; do
  case "$a" in
    --controller) MODE_ARGS=(--input-mode controller --motion) ;;
    --plain) PLAIN=1 ;;
    *) ARGS+=("$a") ;;
  esac
done
FIXED=(--arm G1_29 --sim --network-interface lo --img-server-ip 127.0.0.1)
if [ "$PLAIN" = 1 ]; then
  cd "$G1_THIRD_PARTY/xr_teleoperate/teleop"
  python teleop_hand_and_arm.py "${MODE_ARGS[@]}" "${FIXED[@]}" "${ARGS[@]}"
else
  python "$REPO/02_wrist_cameras/teleop_views.py" --wrist-layout pip "${MODE_ARGS[@]}" "${FIXED[@]}" "${ARGS[@]}"
fi
