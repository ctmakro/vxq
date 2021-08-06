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
