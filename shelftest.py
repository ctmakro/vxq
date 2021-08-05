import shelve

def get_shelf():
    d = shelve.open('data',flag='c')
    return d

if __name__ == '__main__':
    d = get_shelf()
    d['a'] = 1
    d.sync()
    print(d['a'])
