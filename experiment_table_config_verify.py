from vxq_hid import VXQHID as VH
import time
from experiment_table_config import mcrcf

import sys
vxq_model = sys.argv[1] if len(sys.argv)>=2 else 'original'

print('model specified:', vxq_model)
v = VH(configuration=vxq_model)

for k in sorted(mcrcf.keys()):
    print(f'moving to marker {k}...')
    v.g1_cartesian_joint(mcrcf[k])
    print(f'right now at marker {k}.')
    print('Please move the corresponding marker to right below the end of the robot arm. Press Enter to continue to next point')
    input()

print('All points visited, exiting...')
