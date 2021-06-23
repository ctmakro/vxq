import sys
fn = sys.argv[1] if len(sys.argv)>=2 else 'positions.csv'

import time,math,random, numpy as np

# read the positions list from file
from javis_ui_play_all import points as all_points

print(f'read {len(all_points)} points from {fn}')

# index of position list we're on
pind = -1

# limit travel of each axis
def clip(a):
    return max(min(a,4095),0)

class JointController():
    def __init__(self):
        self.jindex = 0 # joint setting selected index
        self.jsettings = [-1]*6 # joint settings
        self.jreadings = [-1]*6 # joint actual readings

        self.pind = -1

    # go interpolated from current location to all_points[idx].
    def slide_to_point(self, idx):
        print('going to point #{}/{}'.format(idx+1, len(all_points)))
        p = all_points[idx]
        v.g1_joint(p, speed=300)
        print('done #{}/{}'.format(idx+1, len(all_points)))

        self.jsettings = p.copy()

    def process_char(self, c):

        if type(c)==type(b''):
            c = c.decode('ascii')[0] # windows hack

        # column select
        if  c=='[': # left
            self.jindex = (self.jindex-1)%6
        elif c==']':
            self.jindex = (self.jindex+1)%6

        # +- values
        elif c=='=':
            self.jsettings[self.jindex] = clip(self.jsettings[self.jindex]+1)
        elif c=='-':
            self.jsettings[self.jindex] = clip(self.jsettings[self.jindex]-1)
        elif c=='+':
            self.jsettings[self.jindex] = clip(self.jsettings[self.jindex]+50)
        elif c=='_':
            self.jsettings[self.jindex] = clip(self.jsettings[self.jindex]-50)
        elif c=='q':
            exit()

        # stage performance
        elif c==',':
            # prev point
            if not len(all_points):
                print('no waypoints to move to')
            else:
                self.pind = (self.pind-1) % len(all_points)
                self.slide_to_point(self.pind)

        elif c=='.':
            # next point
            if not len(all_points):
                print('no waypoints to move to')
            else:
                self.pind = (self.pind+1) % len(all_points)
                self.slide_to_point(self.pind)

        elif c=='r':
            # set setting to reading
            self.jsettings = self.jreadings.copy()

        elif c=='h':
            self.display_help()

        elif c=='u':
            self.go_standup()

        elif c=='s':
            self.save_current()
        else:
            pass

    # def go_standup(self):
    #     pass
    #
    # def save_current(self):
    #     pass

    def update_readings(self, readings):
        self.jreadings = readings
        if -1 in self.jsettings:
            self.jsettings = readings.copy()

    def display_joint_info(self,):

        print(' '.join([
            '{}j{:1d}={:4d}({:4d})'.format(
                ('>'if i==self.jindex else ' '),
                i+1,
                int(self.jsettings[i]),
                int(self.jreadings[i]),
            )
        for i in range(6)]))

    def display_help(self):
        print('press q to quit, [] to select, +- to adjust, shift to accelerate')
        print('press <> to goto prev/next waypoint')
        print('press u to revert to default position')
        print('press r to set current position as commanded position')
        print('press h to show this message')
        print('press s to append current joint setting to csv')

from vxq_hid import VXQHID, arrayfmt

v = VXQHID(quiet=True)

class JCV(JointController):
    def go_standup(self):
        print('going standup position...')
        self.jsettings = [2048]*6

    def save_current(self):
        with open(fn, 'a+') as f:
            f.write(','.join([str(n) for n in self.jsettings])+'\n')
        print(self.jsettings,'written to',fn)


import readchar

jc = JCV()

jc.update_readings(v.get_joints()[:6])

jc.display_help()
jc.display_joint_info()

while 1:
    c = readchar.readchar()
    jc.process_char(c)
    v.go(jc.jsettings)
    jc.update_readings(v.get_joints()[:6])

    jc.display_joint_info()
    print(arrayfmt(v.k.fk(v.k.count2rad(jc.jreadings))))
