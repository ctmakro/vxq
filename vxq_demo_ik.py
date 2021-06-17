import numpy as np
import time
from vxq_hid import VXQHID

dots = np.array([
    [0,200,100],
    [0,1100,100],
    [0,200,1100],
    [0,200,100],

    [0,300,-10],
    [800, 300, -10],
    [800,600, -10],
    [-800, 600, -10],
    [-800, 300, -10],

    [0,200,100]
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

# move to a given coordinate in ik mode.
def ik_goto(p, speed):
    dt = 0.08 # send command every 80 ms
    here_coords = k.fk(k.count2rad(v.get_joints()[0:3]))
    wps = k.planpath_ik(here_coords, p, speed=speed, dt=dt)
    for wp in wps:
        time.sleep(dt)
        v.go(k.rad2count(wp))

def ik_demo():
    for dot in dots:
        ik_goto(dot, speed=250)

ik_demo()
