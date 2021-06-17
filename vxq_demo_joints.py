import numpy as np
import time
from vxq_hid import VXQHID

dots = np.array([
    [2048, 2048, 2048],
    [2048-768, 2048+512, 2048-1024],
    [2048+768, 2048+512, 2048-1024],
    [2048,2048,2048],
    [2048-768, 2048+512, 2048-1024],
    [2048+768, 2048+512, 2048-1024],
    [2048, 2048, 2048],
], dtype='float32')

v = VXQHID(        # rod lengths, measured from actual robot
        l1 = 300.,
        l1_e = 30.,
        l2 = 560.,
        l2_e = 30.,
        l3 = 685., # to j5 center
)
print('serial number is', v.sn)
v.wait_for_readouts()

here = v.get_joints()
print('bot currently at', here)

k = vxq_kinematics = v.k

# move to a given encoder count in joint space.
def joint_goto(p, speed):
    dt = 0.08 # send command every 80 ms
    here_joints = v.get_joints()[0:len(p)]
    wps = k.planpath_raw(here_joints, p, speed=speed, dt=dt)
    for wp in wps:
        time.sleep(dt)
        v.go(wp)

def joint_demo():
    for dot in dots:
        joint_goto(dot, speed=250)

joint_demo()
