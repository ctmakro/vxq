fn='positions.csv'

import sys
fn = sys.argv[1]


with open(fn, 'r') as f:
    csv = f.read()

lines = csv.split('\n')
lines = [l.strip() for l in lines if len(l.strip())>0]
lines = [l for l in lines
    if (not l.startswith('#')) \
    and len([n for n in l.split(',') if len(n)>0])>=2]

# print(lines)

lines = [[j.strip() for j in l.split(',')] for l in lines]
print('read {} valid line(s) from {}...'.format(
    len(lines), fn))


from vxq_hid import VXQHID as VH

v = VH(quiet=True)

import time

for line in lines:
    print('line:', line)
    if len(line)==2 and line[0].strip().lower()=='wait':
        t = float(line[1].strip())
        print('wait for', t)
        time.sleep(t)

    elif len(line)>=6:
        print('moving to', line[0:6])
        v.g1_joint(line[0:6], speed=350)

    else:
        print('unknown line')
