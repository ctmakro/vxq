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

def move_to_5_and_back():
    side_up = (mcrcf[1]+mcrcf[2])/2+np.array([0,-300,200])
    base_up = [0, 1, 600]
    # v.g1_cartesian_joint(center_up)
    # v.g1_cartesian_joint(base_up)
    while 1:
        # for i in range(4):
        #     v.g1_cartesian_joint(mcrcf[i])
        #     time.sleep(0.5)
        v.g1_cartesian_joint(side_up, j1_first=False, speed=250)
        time.sleep(0.5)

        if tst is not None and tst.has_solution():
            at = AffineTransform()

            usc_l = np.array([unit_square_coords[i] for i in range(4)])
            mcrcf_l = np.array([mcrcf[i][0:2] for i in range(4)])
            at.estimate_from(usc_l, mcrcf_l)

            # print('usc',usc_l)
            # print('mcrcf',mcrcf_l)
            # print(at.transform(np.array([0.5, 0.5])))

            if at.has_solution():
                if 5 in tags:
                    # to the right of tag
                    t5 = tags[5]
                    t5r = t5.cxy + t5.rxy * 3.2

                    screen = t5r

                    sc = tst.transform(screen) # screen to square
                    rc = at.transform(sc) # square to robot cartesian

                    default_height = mcrcf[0][2]
                    try:
                        v.g1_cartesian_joint(
                        # v.g1_cartesian_ik(
                            list(rc)+[default_height],
                            j1_first = True,
                            speed=200,
                        )

                        # time.sleep(0.5)

                        v.go_j(4, 2000) # 出水
                        time.sleep(1+3)

                        v.go_j(4, 3000) # 停
                        time.sleep(2)

                    except Exception as e:
                        # point might get out of reach or something
                        print(e)

                    time.sleep(0.5)


threading.Thread(target=move_to_5_and_back, daemon=True).start()

while 1:
    cl.update()
    time.sleep(.05) # 20hz max
    res = cl.result
    if res is not None:
        tags, tst = cl.result
