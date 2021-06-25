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
    center_up = (mcrcf[0]+mcrcf[2])/2+np.array([0,0,500])
    base_up = [0, 1, 600]
    # v.g1_cartesian_joint(center_up)
    # v.g1_cartesian_joint(base_up)
    while 1:
        # for i in range(4):
        #     v.g1_cartesian_joint(mcrcf[i])
        #     time.sleep(0.5)
        v.g1_cartesian_joint(base_up, speed=300)
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
                    screen = tags[5].cxy
                    sc = tst.transform(screen) # screen to square
                    rc = at.transform(sc) # square to robot cartesian

                    default_height = mcrcf[0][2]
                    v.g1_cartesian_joint(
                        list(rc)+[default_height], speed=200)
                    time.sleep(0.5)


threading.Thread(target=move_to_5_and_back, daemon=True).start()

while 1:
    cl.update()
    time.sleep(.05) # 20hz max
    res = cl.result
    if res is not None:
        tags, tst = cl.result
