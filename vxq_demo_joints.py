import numpy as np
import time
from vxq_hid import VXQHID

pi = np.pi

# target angles(j1-j3) in radians.
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

v = VXQHID() # don't care about arm lengths (not using IK)
print('serial number is', v.sn)
print('bot currently at', v.whereami())

k = vxq_kinematics = v.k

def joint_demo():
    for dot in dots:
        v.g1_radians(dot, speed=250)

joint_demo()
joint_demo()
time.sleep(2)
