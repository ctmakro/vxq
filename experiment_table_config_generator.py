
import sys
fn = sys.argv[1] if len(sys.argv)>=2 else 'table_config.json'

print('目标配置文件名：',fn)

print('（倒茶机）实验台配置（正方形坐标系）生成器')
print('请参考 table_config.jpg')
print('请输入正方形的边长A（毫米）')
side = float(input())

print('请输入机器人底座中心，与正方形的左下角，的Y距离（毫米）')
y = float(input())

print('请输入机器人底座中心，与正方形的左下角，的X距离（毫米，在左为负数，在右为正数）')
x = float(input())

print('请输入机器人末端的高度（与J2等高为0，更低、更靠近桌面则为负，默认-100')
height = float(input() or -100)

d = {
    0:[x+side, y+side, height],
    1:[x+side, y, height],
    2:[x, y, height],
    3:[x, y+side, height],
}

d1 = {str(k):v for k,v in d.items()}

print(d1)

import json
with open(fn, 'w') as f:
    json.dump(d1, f)

print('written to', fn)
