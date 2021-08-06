import time, sys, threading

def interval_gen(f=None):
    t0 = time.time()

    def get_interval():
        nonlocal t0
        t1 = time.time()
        interval = t1-t0
        t0 = t1
        if f is None:
            return interval
        else:
            return int(interval*f)
    return get_interval

def lp1_gen(k = 0.8):
    buf = None; ook = 1-k
    def update(v):
        nonlocal buf
        buf = v if buf is None else buf * k + v * ook
        return buf
    return update

def lpn_gen(n = 2, k = 0.8, integerize=False):
    ks = [lp1_gen(k) for i in range(n)]
    def update(v):
        for i in range(n): v = ks[i](v)
        return int(v) if integerize else v
    return update

def run_threaded(f):
    t = threading.Thread(target=f, daemon=True)
    t.start()
    return t

class MailWaiter:
    def __init__(self):
        self.cond = threading.Condition()
        self.mail = None

    def send(self, mail):
        with self.cond:
            self.mail = mail
            self.cond.notify()

    def recv(self, keep=False):
        while 1:
            # with flock:
            tt = 0.0
            dt = 0.5

            with self.cond:
                if self.cond.wait(dt):
                    mail = self.mail
                    if not keep:
                        self.mail = None
                    return mail
                else:
                    tt+=dt
                    print(f'MailWaiter waited for {tt:.1f}s')
                    dt*=2

    def gotmail(self):
        return self.mail is not None

clip = lambda a,b: lambda x: min(b,max(a,x))
clip255 = clip(0,255)

def frequency_regulator_gen(freq, gain=10):
    itvl = interval_gen()

    target_dt = 1/freq

    lasterr = 0
    ierr = 0

    kp= .0
    ki= .8

    def tick():
        nonlocal ierr,lasterr

        dt = itvl()
        err = target_dt - dt
        ierr += err

        ierr = max(-10,min(ierr,10))

        sleeptime = err*kp+ierr*ki

        if sleeptime>0:
            time.sleep(sleeptime)
        else:
            ierr = 0

        lasterr = err

    return tick

if __name__ == '__main__':
    fr = frequency_regulator_gen(10)
    iv = interval_gen()
    k = 0

    erra = 0
    while 1:
        k+=1
        j = iv()
        print(f'{k:5d} {j:.5f} {abs(j-.1):.6f}')

        erra+=(j-.1)**2
        fr()

        time.sleep(0.005)

        if k>30:
            print('erra',erra/30)
            break
