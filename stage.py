import numpy as np

class Stage:
    def __init__(self, fn='positions.csv'):
        with open(fn, 'r') as f:
            csv = f.read()

        lines = csv.split('\n')
        lines = [l.strip() for l in lines if len(l.strip())>0]
        lines = [l for l in lines if (not l.startswith('#')) and len([n for n in l.split(',') if len(n)>0])>=6]

        # print(lines)
        print('read {} valid line(s) from {}...'.format(
            len(lines), fn))

        self.lines = lines
        self.index = 0

    def decode(self, line):
        l = line
        items = l.split(',')
        items = items[0:6]
        assert len(items)==6
        return items

    def decode_all(self):
        return np.array([self.decode(l) for l in self.lines],dtype='float32')

stage = Stage()

# print(s.decode_all())

all_points = stage.decode_all()
print(all_points)
