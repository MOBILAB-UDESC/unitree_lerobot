"""Headless check of the sim's DDS bridge (no Quest, no xr_teleoperate needed).

Run g1_xr_sim.py first, then this. It measures the rt/lowstate rate, then publishes rt/lowcmd the
same way xr_teleoperate does (all joints held, left elbow following a sine) and checks it tracks.
"""

import argparse
import math
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

LEFT_ELBOW = 18

p = argparse.ArgumentParser()
p.add_argument("--domain", type=int, default=1)
p.add_argument("--iface", default="lo")
args = p.parse_args()

ChannelFactoryInitialize(args.domain, args.iface)
latest = {"msg": None, "count": 0}


def on_state(msg: LowState_):
    latest["msg"] = msg
    latest["count"] += 1


sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(on_state, 10)

t_end = time.time() + 5
while latest["msg"] is None and time.time() < t_end:
    time.sleep(0.01)
assert latest["msg"] is not None, "no rt/lowstate received - is g1_xr_sim.py running on the same domain/interface?"

n0 = latest["count"]
time.sleep(1.0)
rate = latest["count"] - n0
print(f"rt/lowstate rate: {rate} Hz (mode_machine={latest['msg'].mode_machine})")

pub = ChannelPublisher("rt/lowcmd", LowCmd_)
pub.Init()
crc = CRC()
cmd = unitree_hg_msg_dds__LowCmd_()
cmd.mode_machine = latest["msg"].mode_machine
q0 = [latest["msg"].motor_state[i].q for i in range(29)]
for i in range(29):
    mc = cmd.motor_cmd[i]
    mc.mode, mc.q, mc.dq, mc.tau = 1, q0[i], 0.0, 0.0
    mc.kp, mc.kd = (300.0, 3.0) if i < 15 else (80.0, 3.0)

max_err = 0.0
t0 = time.time()
while time.time() - t0 < 4.0:
    t = time.time() - t0
    target = 0.6 + 0.4 * math.sin(2 * math.pi * 0.5 * t)
    cmd.motor_cmd[LEFT_ELBOW].q = target
    cmd.crc = crc.Crc(cmd)
    pub.Write(cmd)
    if t > 1.0:
        max_err = max(max_err, abs(latest["msg"].motor_state[LEFT_ELBOW].q - target))
    time.sleep(0.004)

print(f"left elbow max tracking error after settling: {max_err:.3f} rad")
ok = rate > 400 and max_err < 0.15
print("PASS" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
