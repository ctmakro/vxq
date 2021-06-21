import numpy as np
import time
from vxq_hid import VXQHID,arrayfmt

pi = np.pi

# target angles in radians.
neut = np.array([pi*.5, pi*.5, pi*1.], dtype='float32')
dots = np.array([
    [0,0,0],
    [-.375*pi, .25*pi, -.5*pi],
    [+.375*pi, +.25*pi, -.5*pi],
    [0,0,0],
    [-.375*pi, +.25*pi, -.5*pi],
    [+.375*pi, +.25*pi, -.5*pi],
    [0,0,0],
], dtype='float32') + neut

v = VXQHID() # don't care about rod lengths (not using IK)
print('serial number is', v.sn)

k = vxq_kinematics = v.k

here = v.get_joints()
here_c = k.fk(k.count2rad(here[0:3]))
print('bot currently at', here, arrayfmt(here_c))

# convert all radians to encoder counts
for i in range(len(dots)):
    dots[i] = np.array(k.rad2count(dots[i]))

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
time.sleep(2)
