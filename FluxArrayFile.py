import numpy as np
import time

def float_to_3_bytes(x):
    assert x >= 0
    i = int(round(1.e5 * x))
    b0 = i//65536
    q = b0*65536
    b1 = (i - q)//256
    b2 = i - q - b1*256
    return [b0, b1, b2]

def bytes_to_float(b, i):
    return 1.e-5 * (b[i]*65536 + b[i+1]*256 + b[i+2])

class FluxArrayFile:

    def __init__(self, path):
        self.path = path
        
    def save(self, arr):
        bb = []
        for x in arr:
            bb.extend(float_to_3_bytes(x))
        with open(self.path, 'wb') as f:
            f.write(bytearray(bb))

    def load(self):
        with open(self.path, 'rb') as f:
            bb = f.read()
        assert len(bb)%3 == 0
        ff = []
        for i in range(len(bb)//3):
            ff.append(bytes_to_float(bb, i*3))
        return ff


if __name__=='__main__':
    fn = 'FA_test.flux'
    rnd = np.random.rand(1000)
    FA = FluxArrayFile(fn)
    FA.save(rnd)
    arr = FA.load()

    for z in zip(rnd, arr):
        s1 = '%.5f'%z[0]
        s2 = '%.5f'%z[1]
        assert s1==s2
        
    print('test passed')
    
    N = 333333
    rnd = np.random.rand(N)
    FA = FluxArrayFile(fn)
    t1 = time.time()
    FA.save(rnd)
    t2 = time.time()
    arr = FA.load()
    t3 = time.time()
    
    print('benchmark with '+str(N)+' numbers')
    print('saved in ', t2-t1, 'sec')
    print('loaded in ', t3-t2, 'sec')
    









