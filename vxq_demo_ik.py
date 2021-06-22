import numpy as np
import time
from vxq_hid import VXQHID

pi = np.pi

# points, in cartesian coordinates
dots = np.array([
    [0,200,100],
    [0,1100,100],
    [0,200,1100],
    [0,200,100],

    [0,300, -10],
    [800, 300, -10],
    [800,600, -10],
    [-800, 600, -10],
    [-800, 300, -10],

    [0,200,100]
], dtype='float32')

v = VXQHID(configuration='original')
# v = VXQHID(configuration='shorter')
print('serial number is', v.sn)
print('bot currently at', v.whereami())

k = v.k # kinematics object

for dot in dots:
    print('attempting to ik', dot)
    k.ik(dot) # test if all points can be IK'd (are within reachable range

# iterate over the points.
def ik_demo(r=1):
    # move to starting point in joint space
    v.g1_cartesian_joint(dots[0], speed=300)
    for j in range(r):
        for i in range(1, len(dots)):
            v.g1_cartesian_ik_withstart(dots[i-1], dots[i], speed=200)

ik_demo()
ik_demo()
time.sleep(2)
