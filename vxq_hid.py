# pip install hidapi
import hid
from utils import *
import time,re,threading,math

import numpy as np

def arrayfmt(a, to_int=False):
    if to_int:
        return f"[{' '.join([f'{int(ai):4d}' for ai in a])}]"
    else:
        return f"[{' '.join([f'{ai:.2f}' for ai in a])}]"

# everything is little endian
def i2b(i, length=1): return i.to_bytes(length, 'little')
import struct
def f2b(f):return struct.pack('<f', float(f))
def b2f(b):return struct.unpack('<f', b)[0]

# assert b2f(f2b(3.14)) == 3.14
# print(b2f(f2b(3.14)))

def intify(s):
    if 'x' in s.lower(): return int(s, 16)
    return int(s)

with open('vxqusb.h', 'r') as f:
    vxqusbh = f.read()

defines = re.findall(
    r'^#define\s+?([A-z_]+?)\s+([0-9x]+)',
    vxqusbh,
    flags = re.M)

from types import SimpleNamespace as SN
vxqusbh = vuh = SN(**{match[0]:intify(match[1]) for match in defines})

u32 = 2**32 - 1
def getts(): return int(time.time()*1000) & u32

class VXQHID:
    def __init__(self, serial_number=None, **k):
        self.k = vxq_kinematics_gen(**k)
        self.connected = False
        sn = self.connect(serial_number=serial_number)

        time.sleep(1)

        self.startup_interval = time.time()

        self.start_monitoring_thread()
        self.wait_for_readouts()

        # # https://github.com/bitcoin-core/HWI/blob/master/hwilib/devices/trezorlib/transport/hid.py
        # self.hid_version = self.probe_hid_version()
        # print('hid_version:', self.hid_version)

    def probe_hid_version(self) -> int:
        n = self.h.write(bytearray([0, 63] + [0xFF] * 63))
        if n == 65:
            return 2
        n = self.h.write(bytearray([63] + [0xFF] * 63))
        if n == 64:
            return 1
        raise Exception(f"Unknown HID version: ({n}) ")

    def close(self):
        print('trying to disconnect...')
        self.connected = False
        time.sleep(0.5)
        self.h.close()
        print(f'disconnected from {self.sn}.')

    def find_all(self):
        # find all hid devices that are vxq robots
        found = []
        devices = hid.enumerate()
        for device in devices:
            if device['product_string'].lower().startswith('vxq'):
                found.append(device)
        return found

    def connect(self, serial_number=None):
        # connect to a vxq device.
        # specified a serial number,
        # or connect to first in list that's not occupied

        devices = self.find_all()
        if not devices:
            raise Exception('no vxq-named device found')
        print(f'{len(devices)} vxq-named device(s) found')

        sn_found = []
        sn_map = {}
        for device in devices:
            sn_found.append(device['serial_number'])
            sn_map[device['serial_number']] = device

        def tryopen(device):
            vid = device['vendor_id']
            pid = device['product_id']

            hidd = hid.device()
            try:
                hidd.open(vid, pid, serial_number)
            except Exception as e:
                print(e)
                # hidd.close()
                return False

            self.connected = True
            hidd.set_nonblocking(False)

            manufs, ps, sns = \
            hidd.get_manufacturer_string(), hidd.get_product_string(), hidd.get_serial_number_string()
            print(f'connection successful. \n#Manufacturer: {manufs} \n#Product: {ps} \n#SN: {sns}')

            return hidd

        if serial_number is None:
            for i in range(len(devices)):
                print(f'serial_number not specified, try #{i} in list of all devices')
                serial_number = sn_found[i]
                device = devices[i]

                hidd = tryopen(device)
                if not hidd:
                    print(f'connection of device #{i} failed (occupied?)')
                    continue
                else:
                    break
            if not hidd:
                raise Exception('all device(s) failed to connect')

        else:
            print('serial_number specified:', serial_number)
            if serial_number not in sn_map:
                raise Exception('vxq device with given serial_number not found')
            device = sn_map[serial_number]

            hidd = tryopen(device)
            if not hidd:
                raise Exception(f'device {serial_number} failed to connect (occupied?)')

        self.h = h = hidd
        # h.set_nonblocking(1)

        self.joints = [None]*8
        self.jlp = [None]*8

        self.sn = serial_number

    def write_packet(self, ba):
        # n = 0
        # while self.connected==False and n<4:
        #     n+=1
        #     print('not connected while write()', n)
        #     time.sleep(0.5)

        # fill the packet to 64 bytes
        if len(ba)<64:
            ba = ba+b'\0'*(64-len(ba))

        if self.connected:
            # always append 0x00 to the beginning
            self.h.write(b'\0'+ba)
        else:
            raise Exception('device disconnected while trying to write().')

    def go_va(self, *args):
        return self.go(args)

    def go(self, targets):
        # target: float[6]
        '''
        void VXQUsbHost::sendCmdMotionGo(bool cartesian, float target[], quint8 speedset)
        {
            struct usb_cmd_motion_s cmd;
            cmd.ts = getTs();
            cmd.type = USB_CMD_MOTION;
            cmd.mode = USB_CMD_MOTION_MODE_GO;
            cmd.flags = ((speedset == 0xff) ? 0 : USB_CMD_MOTION_F_SPEEDSET)
                      | (cartesian ? USB_CMD_MOTION_F_CARTESIAN : 0);
            cmd.speedset = speedset;
            memcpy(cmd.target, target, sizeof(cmd.target));
            sendRaw(&cmd, sizeof(cmd));
        }
        '''

        ts = i2b(getts(), 4)
        # ts = b'\x00'*4
        mtype = i2b(vuh.USB_CMD_MOTION, 1)
        mode = i2b(vuh.USB_CMD_MOTION_MODE_GO, 1)
        flags = i2b(0)
        speedset = b'\xff'

        # angles not specified in 'targets' are replaced
        # with their current value
        ja = self.joints.copy()[0:6]

        for i,t in enumerate(targets):
            if t is not None:
                ja[i] = t

        if None in ja:
            raise Exception('no readouts yet, not sure what to send')

        print('writing:', arrayfmt(ja, to_int=True))

        c = ts + mtype + mode + flags + speedset + \
            b''.join([f2b(a) for a in ja])

        self.write_packet(c)

    # read packet with retries and reconnections.
    def read_packet(self):
        # read loop

        acc = 0
        while True:
            if not self.connected:
                raise Exception('device disconnected')

            try:
                d = self.h.read(64)
            except Exception as e:
                print(e, f'({time.time()-self.startup_interval:.2f}s)')
                if 'read error' in str(e):
                    acc+=1
                    if acc>3:
                        print('too much read errors')
                        self.connected = False
                        raise(e)
                    else:
                        time.sleep(0.5)
                        continue
                else:
                    acc = 0
                    self.connected = False
                    raise e

            # if no packet to be read
            if not d:
                time.sleep(0.02)
                # try read again
                continue
            else:
                return d

    def start_monitoring_thread(self):
        '''
        void MainWindow::processData(QByteArray buf)
        {
            //ui->rxdata->setPlainText(buf.toHex(' '));
            //qDebug() << buf.toHex();

            struct usb_header_s *header = (struct usb_header_s *)buf.data();
            switch(header->type) {
            case USB_MSG_SYS_REPORT: {
                auto msg = (struct usb_msg_sys_report_s *)buf.data();
                parseUsbMsgSysReport(msg);
            } break;
            case USB_MSG_CMD_ACK: {
                auto msg = (struct usb_msg_cmd_ack_s *)buf.data();
                parseUsbMsgCmdAck(msg);
            } break;
            case USB_MSG_JOINT: {
                auto msg = (struct usb_msg_joint_s *)buf.data();
                parseUsbMsgJoint(msg);
            } break;
            case USB_MSG_PARAM_LIST: {
                auto msg = (struct usb_msg_param_list_s *)buf.data();
                parseUsbMsgParamList(msg);
            } break;
            case USB_MSG_PARAM_ACK: {
                auto msg = (struct usb_msg_param_ack_s *)buf.data();
                parseUsbMsgParamAck(msg);
            } break;
            default: qWarning() << "Unknown message type " << header->type;
            }
        }
        '''

        def parse_joint_data(arr):
            assert len(arr)==10
            '''
            struct usb_msg_joint_data_s {
                uint8_t options; [0]
                uint8_t status; [1]
                uint16_t pos; [2-3]
                uint16_t fb; [4-5]
                uint16_t current; [6-7]
                uint8_t mode; [8]
                int8_t temp; [9]
                //uint8_t ack;
                //uint8_t rsvd;
            };  //10B
            '''

            options = arr[0]
            status = arr[1]
            pos = arr[2] + arr[3]*256
            fb = arr[4] + arr[5]*256

            mode = arr[8]
            temp = arr[9]

            return fb
            # return arr

        def msgloop():
            joints_cache = self.joints.copy()
            joints_buffer = self.joints.copy()

            while not self.connected:
                time.sleep(0.2)

            # main loop
            while True:
                if not self.connected:
                    break

                d = self.read_packet()

                # got packet
                mtype = d[4]
                # print('received message type', mtype)

                if mtype==vuh.USB_MSG_SYS_REPORT:
                    pass
                elif mtype==vuh.USB_MSG_CMD_ACK:
                    pass
                elif mtype==vuh.USB_MSG_JOINT:
                    '''
                    struct usb_msg_joint_s {
                        uint32_t ts; [0-3]
                        uint8_t type; [4]
                        uint8_t subframe;   //0 = j1~j5, 1 = j6-j8 [5]

                        union usb_msg_joint_u_u { [6]
                            struct usb_msg_joint_s0_s {
                                struct usb_msg_joint_data_s joint[5]; [6..6+50]
                            } s0;   //50B

                            struct usb_msg_joint_s1_s {
                                struct usb_msg_joint_data_s joint[3]; [6..6+30]
                                uint8_t motion_status; //bit0 = running, bit1 = reached [6+30]
                                uint8_t speedset; [6+31]

                                float lastpos[NJOINTS];
                            } s1;   //56B
                        } u;
                    };  //62B
                    '''
                    # above code is disaster
                    subframe = d[5]
                    starting_offset = so = 6+2

                    # readouts are split into 2 parts,
                    # each "subframe" contains only one part
                    if subframe == 0:
                        for i in range(5):
                            joints_buffer[i] = parse_joint_data(d[so+10*i: so+ 10*(i+1)])
                    else:
                        # print('sf1')
                        for i in range(3):
                            joints_buffer[i+5] = parse_joint_data(d[so+10*i: so+10*(i+1)])

                        if None not in joints_buffer:
                            self.joints = joints_buffer.copy()

                        motion_status = d[so+30]
                        speedset = d[so+31]

                        # print(f'sf:{subframe} mo_stat:{motion_status}, speedset:{speedset}')

                        for i in range(6):
                            self.jlp[i] = b2f(bytes(d[so+32+i*4: so+32+(i+1)*4]))

                        if None not in joints_buffer:
                            # print array only if content changed
                            for i in range(6):
                                jci = joints_cache[i]
                                jbi = joints_buffer[i]
                                if (jci is None) or abs(jci-jbi)>2:
                                    rads = self.k.count2rad(joints_buffer)
                                    print(
                                    'readback[joints][angles][coords]:',
                                    arrayfmt(joints_buffer, to_int=True),
                                    arrayfmt(list(self.k.r2a(rads)),to_int=True),
                                    arrayfmt(list(self.k.fk(rads)), to_int=True)
                                    )
                                    joints_cache = joints_buffer.copy()
                                    break

                    # print([j for j in self.joints])
                    # print([f'{j:.2f}' for j in self.jlp if j])
                else:
                    # do nothing
                    pass

        print('starting recv loop thread')
        t = threading.Thread(target=msgloop, daemon=True)
        t.start()
        # msgloop()

    def wait_for_readouts(self):
        while 1:
            if None in self.joints[0:6]:
                time.sleep(0.1)
            else:
                break

    def go_standup(self):
        return self.go([2048]*6)

    def go_j(self, idx, count):
        return self.go([None]*(idx-1)+[count])

    def get_joints(self):
        return self.joints.copy()

# for k,v in vxqusbh.items():
#     print(k,v)

# print(intify('4'))
# print(intify('0xff'))

def vxq_kinematics_gen(
        # rod lengths, measured from robot
        l1 = 300.,
        l1_e = 30.,
        l2 = 560.,
        l2_e = 30.,
        l3 = 685., # to j5 center
    ):

    l1ex = - l1_e
    l1ey = 0.

    import autograd.numpy as np  # Thinly-wrapped numpy
    from autograd import grad, elementwise_grad as egrad
    import numpy as onp
    on = onp

    # convert radians 2 angle
    def r2a(r): return r / np.pi * 180
    def a2r(a): return a / 180 * np.pi

    r90 = a2r(90) # pi/2
    r180 = a2r(180) # pi

    neutral = 2048
    # all joints set to 2048 encoder count => standup (tallest) position

    rad_per_count = 1 / 1024 * r90 # 1024 count per 90 deg
    count_per_rad = 1 / rad_per_count

    # convert encoder count to radians
    def count2rad(joints):
        def c2r(j):
            return (j-neutral) * rad_per_count

        j = joints

        # angle definitions: refer to Qin's hand drawing
        a1 = - c2r(j[0]) + r90
        # a2 = c2r(j[1]) + r90 # old definition
        a2 = -c2r(j[1]) + r90 # per 20's definition
        a3 = c2r(j[2]) + r180

        if len(joints)==3:return np.array([a1,a2,a3], dtype='float32')

        a4 = - c2r(j[3])
        a5 = - c2r(j[4])
        a6 = - c2r(j[5])

        return np.array([a1, a2, a3, a4, a5, a6], dtype='float32')

    # back
    def rad2count(rads):
        def r2c(r):
            c = (r * count_per_rad + neutral)
            if c>=4096: c-=4096
            if c<0: c+=4096
            return c

        r = rads
        j1 = r2c(-r[0] + r90)
        # j2 = r2c(r[1] - r90) # old definition
        j2 = r2c(-r[1] + r90) # per 20's definition
        j3 = r2c(r[2] - r180)

        if len(rads)==3:return np.array([j1,j2,j3], dtype='float32')

        j4 = r2c(r[3])
        j5 = r2c(r[4])
        j6 = r2c(r[5])

        return np.array([j1, j2, j3, j4, j5, j6], dtype='float32')

    countlimits = np.array([
    [1025,3071],
    [1025,2960],
    [1,2047],
    [1025,3071],
    [1025,3071],
    [1025,3071],
    ], dtype='float32')

    clr0 = count2rad(countlimits[:, 0])
    clr1 = count2rad(countlimits[:, 1])
    radlimits_high = np.maximum(clr0, clr1)
    radlimits_low =  np.minimum(clr0, clr1)

    if __name__ == '__main__':

        # use as limits in optimization

        print('lower and higher limit of each joint, in sensor count (1024 counts per 90 degrees)')
        print(countlimits)

        print('lower and higher limit of each joint, in converted radians')
        print(arrayfmt(radlimits_low))
        print(arrayfmt(radlimits_high))

    def fk(angles):
        a = angles
        a1 = a[0]; a2 = a[1]; a3 = a[2]
        # a1, a2, a3 = a[0:3]

        # l1, l2 and l3 form an imaginary plane that rotates with j1
        # imaginary plane uses imx and imy as axis
        # refer to Qin's hand drawing

        a2x = np.cos(a2) * l2
        a2y = np.sin(a2) * l2

        # ta0 = r90 - (r180 - a2)
        ta0 = a2 - r90
        ta0x = np.cos(ta0) * l2_e
        ta0y = np.sin(ta0) * l2_e

        # ta1 = a3 - (r180 - a2)
        ta1 = a3 + a2 - r180
        ta1x = np.cos(ta1) * l3
        ta1y = np.sin(ta1) * l3

        imx = l1ex + a2x + ta0x + ta1x
        imy = l1ey + a2y + ta0y + ta1y

        #
        x = np.cos(a1) * imx
        y = np.sin(a1) * imx
        z = imy

        return np.array([x, y, z], dtype='float32')

    def ik_analytical(target_coords, ig=None):
        x,y,z = target_coords

        if y<=0:
            raise Exception('target coordinate y < 0 (back of machine)')

        a1 = np.arctan2(y,x)

        sqxy = x*x+y*y
        sqz = z*z
        distxy = np.sqrt(sqxy)
        dist = np.sqrt(sqxy+sqz)
        dist1e = np.sqrt((distxy + l1_e)**2 + sqz)

        l2s = np.sqrt(l2*l2 + l2_e*l2_e)

        if dist1e >= (l3+l2s):
            print('xyz dist1e, l3, l2s',x,y,z, dist1e, l3, l2s)
            raise Exception('target coordinate too far.')
        if dist1e <= abs(l3-l2s):
            print('xyz dist1e, l3, l2s',x,y,z, dist1e, l3, l2s)
            raise Exception('target coordinate too close.')

        a2e, a3ex , _ = triangle_sss(l3, dist1e, l2s)

        a3e = np.arccos(l2/l2s)
        a3 = a3ex-a3e

        a2s = np.arcsin(z/dist1e)
        a2 = a2e+a2s+a3e

        return np.array([a1, a2, a3], dtype='float32')

    def triangle_sss(a,b,c):
        # law of sines, https://www.calculator.net/triangle-calculator.html
        # sss, https://www.mathsisfun.com/algebra/trig-solving-sss-triangles.html
        a2 = a*a
        b2 = b*b
        c2 = c*c
        A = np.arccos((b2+c2-a2) / (2*b*c))
        B = np.arccos((a2+c2-b2) / (2*a*c))
        C = np.arccos((a2+b2-c2) / (2*a*b))
        return A,B,C

    if __name__ == '__main__':
        print('equi-triangle',[r2a(r) for r in triangle_sss(1,1,1)])

    def error_function(angles, target_coords):
        result_coords = fk(angles)
        return np.sum(np.square(result_coords - target_coords))

    gef = grad(error_function)

    default_initial_guess = a2r(np.array([45, 90, 90], dtype='float32'))
    default_mmt = np.zeros_like(default_initial_guess)

    def ik_approximate(target_coords, ig=None):
        c = target_coords
        initial_guess = default_initial_guess.copy() if ig is None else ig
        x = initial_guess

        rate = .5e-5
        decay = 0.9
        onemd = 1-decay
        mmtdir = default_mmt

        interval = interval_gen()

        errx = 0
        errs = []

        err_limit = 3 ** 2

        rlml = radlimits_low[0:3]
        rlmh = radlimits_high[0:3]

        for i in range(1000):
            dir = gef(x,c)
            mmtdir = mmtdir * decay + dir * onemd
            x = x - mmtdir * rate
            x = np.clip(x, rlml, rlmh)

            if i>4 and i % 5 == 0:
                errx = error_function(x,c)
                # print(f'{i:4d}, rate {rate:.7f}, err {errx}, {x}')

                errs.append(errx)
                if errx < err_limit:
                    break

                if len(errs)>10 and (errx >= errs[-2]
                    and errx >= errs[-4]
                    and errx >= errs[-6]
                    ): # error not decreasing anymore
                    break

        elap = interval()

        if i>20:
            print(f'{i} steps, costs {elap:.2f}s, abs error {np.sqrt(errx):.2f}')
        return x

    if __name__ == '__main__':
        examples = np.array([
            [-100,100,700],
            [100,100,700],
            [0,100,700],
        ], dtype='float32')

        for idx, example in enumerate(examples):
            print(f'===== example #{idx} =====')
            angles = ik_analytical(example)
            print('solved angles',arrayfmt(r2a(angles)),'solved joint counts', arrayfmt(rad2count(angles)))
            print('original requested pos', example,'fked pos',arrayfmt(fk(angles)))

    def smooth_interpolate(x):
        # map 0..1 to 0..1; first derivatives continuous and zero at both ends
        return -2 * x**3 + 3 * x**2

    def planpath_ik(c1, c2, speed=100, ig=None, dt=0.08):
        interval = interval_gen()

        rawpath = planpath_raw(c1, c2, speed, dt)

        waypoints_ik = []
        for pos in rawpath:
            solution = ik_analytical(pos)
            # solution = ik(pos, ig)
            # ig = solution.copy()
            waypoints_ik.append(solution)

        print(f'ik path planned ({len(rawpath)} steps) in {interval():.2f} seconds')

        return np.array(waypoints_ik, dtype='float32')

    def planpath_raw(p1, p2, speed=100, dt=0.08):
        interval = interval_gen()

        delta = p2 - p1
        dist = np.sqrt(np.sum(np.square(delta)))
        t = dist / speed
        total_steps = int(t / dt) + 2

        waypoints = []

        steplet = 1/(total_steps-1)
        for i in range(1,total_steps):
            progress = i*steplet
            pos = p1 + delta * smooth_interpolate(progress)
            waypoints.append(pos)
            # print(arrayfmt(pos))

        print(f'raw path planned ({total_steps} steps) in {interval():.2f} seconds')

        return np.array(waypoints, dtype='float32')

    return SN(
        planpath_raw=planpath_raw,
        planpath_ik=planpath_ik,
        fk = fk,
        ik_analytical = ik_analytical,
        ik = ik_analytical,
        ika = ik_analytical,
        ik_approximate = ik_approximate,
        rad2count = rad2count,
        count2rad = count2rad,

        r2a = r2a, a2r = a2r,
    )

vxq_kinematics = vxq_kinematics_gen()

if __name__ == '__main__':

    k = vxq_kinematics

    dots = np.array([
        [0,200,100],
        [0,1100,100],
        [0,200,1100],
        [0,200,100],

        [0,300,-100],
        [800, 300, -100],
        [800,600, -100],
        [-800, 600, -100],
        [-800, 300, -100],

        [0,200,100]
    ], dtype='float32')

    # check if above waypoints reachable
    ik_dots = [k.rad2count(k.ik(dot)) for dot in dots]
    # print('dots, ik_dots')
    # print(dots)
    # print(ik_dots)

    v = VXQHID(        # rod lengths, measured from robot
            l1 = 300.,
            l1_e = 30.,
            l2 = 560.,
            l2_e = 30.,
            l3 = 685., # to j5 center
    )
    print('serial number is', v.sn)

    # test close functionality
    v.close()
    v = VXQHID(        # rod lengths, shorter(newer) version
            l1 = 300.,
            l1_e = 30.,
            l2 = 510.,
            l2_e = 30.,
            l3 = 640., # to j5 center
    )
    print('serial number is', v.sn)

    here = v.get_joints()
    here_c = k.fk(k.count2rad(here[0:3]))
    print('bot at', here, arrayfmt(here_c))

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
    def ik_demo(r=1):
        # move to starting point in joint space
        starting_point_j = k.rad2count(k.ik(dots[0]))
        print('spj', starting_point_j)
        joint_goto(starting_point_j, speed=400)

        for j in range(r):
            for i in range(1, len(dots)):
                ik_move(dots[i-1], dots[i], speed=200)

    pi = np.pi
    def swing(r=1):
        for j in range(r):
            joint_goto(k.rad2count([pi*0.5, pi*0.75, pi*0.5]), speed=200)
            joint_goto(k.rad2count([pi*0.5, pi*0.5, pi*1.]), speed=200)

    # time.sleep(3)
