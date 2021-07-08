from vxq_hid import VXQHID as VH
import time
from experiment_table_config import mcrcf
import numpy as np
import sys
vxq_model = sys.argv[1] if len(sys.argv)>=2 else 'original'

print('所选择的型号 model specified:', vxq_model)
v = VH(configuration=vxq_model)

home_pos = mcrcf['home_pos']
move_height = mcrcf['move_height']

for k in [0,1,2,3]:
    print(f'移动到标记{k} moving to marker {k}...')

    mk = mcrcf[k]
    mk_higher = mk.copy()
    mk_higher[2] = move_height

    v.g1_cartesian_joint(mk_higher)
    v.g1_cartesian_joint(mk)

    print(f'已经移动到标记{k} right now at marker {k}.')
    print('请将对应标记移动到机械臂末端正下方 Please move the corresponding marker to right below the end of the robot arm. Press Enter to continue to next point')
    input()

    v.g1_cartesian_joint(mk_higher)


print('所有点已遍历 All points visited')

print('正在前往加茶位置')
v.g1_cartesian_joint(home_pos)

print('到达加茶位置')
input()

print('上升到移动高度...')
v.g1_cartesian_joint(home_pos[0:2]+[mcrcf['move_height']])

print('到达移动高度')

print('按任意键退出')
input()
