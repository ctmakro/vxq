from vxq_hid import *
from arucofun import *

import sys
vxq_model = sys.argv[1] if len(sys.argv)>=2 else 'original'

v = VXQHID(configuration=vxq_model)

time.sleep(.5)

cl = camloop(tabletop_square_matcher_gen(), threaded=True)

tst = None
tags = {}

from experiment_table_config import mcrcf


def tealoop():
    while 1:
        move_back()
        try_move_to_5()

def move_back():
    side_up = (mcrcf[1]+mcrcf[2])/2+np.array([0,-300,200])
    base_up = [0, 1, 600]

    # move to starting point
    time.sleep(0.5)
    v.g1_cartesian_joint(side_up, j1_first=False, speed=800)
    time.sleep(0.5)

def try_move_to_5():

    # if camera to square coords solution exists
    if tst is not None and tst.has_solution():
        at = AffineTransform()

        # try to get solution from camera to table coords
        usc_l = np.array([unit_square_coords[i] for i in range(4)])
        mcrcf_l = np.array([mcrcf[i][0:2] for i in range(4)])
        at.estimate_from(usc_l, mcrcf_l)

        # print('usc',usc_l)
        # print('mcrcf',mcrcf_l)
        # print(at.transform(np.array([0.5, 0.5])))

        # if solution from camera to table coords exists:
        if at.has_solution():
            # 5 not occluded, 4 occluded, meaning cup over 4
            if 5 in tags and 4 not in tags:

                tag5 = tags[5]
                # to the right of tag 5
                t5c = Detection(tag5.marker_id, tag5.corners) # make copy
                t5r = t5c.cxy + t5c.rxy * 3.2

                screen = t5r

                sc = tst.transform(screen) # screen to square
                rc = at.transform(sc) # square to robot cartesian

                default_height = mcrcf[0][2]

                # let's move! but from another thread...
                def go_pour_tea():
                    try:
                        cancelled = lambda:v.cancel_movement
                        v.g1_cartesian_joint(
                        # v.g1_cartesian_ik(
                            list(rc)+[default_height],
                            j1_first = True,
                            speed=600,
                        )
                        time.sleep(1)

                        if cancelled(): return
                        v.go_j(4, 2000) # 出水
                        if cancelled(): return

                        time.sleep(1+3)

                        if cancelled(): return
                        v.go_j(4, 3000) # 停
                        if cancelled(): return

                        time.sleep(2)

                    except Exception as e:
                        # point might get out of reach or something
                        print(e)

                th = threading.Thread(target=go_pour_tea, daemon=True)
                th.start()

                # check if target moved
                while th.is_alive():
                    time.sleep(0.05)

                    # if 4 become visible, or
                    # tag 5 moved for more than size / 2
                    if 4 in tags or (5 in tags and
                        np.sum(np.square(tags[5].cxy - t5c.cxy))
                        > (t5c.size/2)**2):

                        # cancel_movement
                        v.cancel_movement = True
                        th.join()
                        v.cancel_movement = False
                        v.go_j(4, 3000) # 停

                        break

threading.Thread(target=tealoop, daemon=True).start()

while 1:
    cl.update()
    time.sleep(.03) # 20hz max
    res = cl.result
    if res is not None:
        tags, tst = cl.result
