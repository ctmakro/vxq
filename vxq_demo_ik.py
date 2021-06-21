import numpy as np
import time
from vxq_hid import VXQHID,arrayfmt

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

# v = VXQHID(        # rod lengths, measured from actual robot
#         l1 = 300.,
#         l1_e = 30.,
#         l2 = 560.,
#         l2_e = 30.,
#         l3 = 685., # to j5 center
# )d
v = VXQHID(        # rod lengths, shorter(newer) version
        l1 = 300.,
        l1_e = 30.,
        l2 = 510.,
        l2_e = 30.,
        l3 = 640., # to j5 center
)
print('serial number is', v.sn)

k = v.k # kinematics object

for dot in dots:
    print('attempting to ik', dot)
    k.ik(dot) # test if all points can be IK'd (are within reachable range


here = v.get_joints() # display current location
here_c = k.fk(k.count2rad(here[0:3]))
print('bot currently at', here, arrayfmt(here_c))


# move to a given encoder count in joint space.
def joint_goto(p, speed=100):
    dt = 0.08 # send command every 80 ms
    here_joints = v.get_joints()[0:3]
    wps = k.planpath_raw(here_joints, p, speed=speed, dt=dt)
    for wp in wps:
        time.sleep(dt)
        v.go(wp)

# move to a given coordinate in ik mode.
def ik_move(p0, p1, speed):
    dt = 0.08 # send command every 80 ms
    wps = k.planpath_ik(p0, p1, speed=speed, dt=dt)
    for wp in wps:
        time.sleep(dt)
        v.go(k.rad2count(wp))

# iterate over the points.
def ik_demo():
    # move to starting point in joint space
    starting_point_j = k.rad2count(k.ik(dots[0]))
    print('spj', starting_point_j)
    joint_goto(starting_point_j, speed=400)

    for i in range(1, len(dots)):
        ik_move(dots[i-1], dots[i], speed=200)

ik_demo()
time.sleep(2)
