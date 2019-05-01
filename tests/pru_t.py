import utils.CardUtils as cu
import utils.PyRandomUtils as pru
from datetime import datetime
import numpy as np
import os


def time_test(gen, n):
    t = datetime.now()
    pru.make_ts_no_batch(gen, n)
    dt = datetime.now() - t

    return dt

def time_test_suite(gen_func, maxexp, name):
    print("{}:".format(name))
    for i in range(maxexp):
        gen = gen_func()
        print("10**{}: {}".format(i, time_test(gen, 10**i)))


class urand_gen(pru.PRNG):

    def make_buffer(self):
        self.buffer = np.frombuffer(os.urandom(self.bufsize), dtype=np.uint16)

    def __init__(self, bufsize = 1024 * 8):
         self.bufsize = bufsize
         self.make_buffer()
         self.randcount = 0

    def rand(self):
        ret = self.buffer[self.randcount%len(self.buffer)] / np.iinfo(np.uint16).max
        self.randcount += 1
        if self.randcount%len(self.buffer) == 0:
            self.make_buffer()
        return ret

    def seed(self, x):
        pass

    def reset(self):
        pass

if __name__ == '__main__':
    n = 7
    # time_test_suite(lambda: pru.LCG_RANDU(1), n, "lcg")
    # time_test_suite(pru.PyRandGen, n, 'python')
    # time_test_suite(pru.HaltonGen, n, 'Halton')
    # time_test_suite(lambda: pru.HaltonGen_Deck(batch_size=10**6), n, 'HaltonDeck')
    # time_test_suite(lambda: urand_gen(bufsize=1024), n, '2**10')
    time_test_suite(lambda: urand_gen(bufsize=2**12), n, '2**12')
    time_test_suite(lambda: urand_gen(bufsize=2**14), n, '2**14')
    time_test_suite(lambda: urand_gen(bufsize=65536), n, '2**16')
    time_test_suite(lambda: urand_gen(bufsize=10**7 * 5000), n, '2**16')
