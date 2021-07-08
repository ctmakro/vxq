import numpy as np
npa = lambda k: np.array(k)

# cartesian coords of the end of the arm above each of the markers.
# measured by moving the arm above each marker and read the cartesian coords.
# $ python javis_ui_2021.py

if __name__ == '__main__':
    import sys
    fn = sys.argv[1] if len(sys.argv)>=2 else 'table_config.json'
else:
    fn = 'table_config.json'

marker_coordinates_in_robot_cartesian_frame = mcrcf = \
{
    0:npa([8,936,-130]),
    1:npa([8,372,-130]),
    3:npa([-622,918,-130]),
    2:npa([-581,328,-130]),
}

import json
try:
    j = json.load(open(fn,'r'))
except FileNotFoundError:
    print(fn, 'not found')
except Exception as e:
    print(f'Error while reading {fn}:',e)
else:
    q = {}
    # for k in list(j.keys()):
    for k in [0,1,2,3]:
        q[int(k)] = j[str(k)]

    print(q)

    for k in q:
        print(f'override {k} in mcrcf with {q[k]} from {fn}')
        mcrcf[k] = npa(q[k])

    mcrcf['home_pos'] = j['home_pos']
    mcrcf['move_height'] = j['move_height']
