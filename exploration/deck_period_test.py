import utils.CardUtils as cu
import utils.PyRandomUtils as pru
import random
import numpy as np
import matplotlib.pyplot as plt

class CountGen(pru.PRNG):
    def __init__(self, len, seed=0):
        self.len = len
        self.seed = seed

    def rand(self):
        self.seed = (self.seed+1)
        return (self.seed%self.len)/float(self.len)

class CountRandGen(pru.PRNG):
    def __init__(self, rand, len, seed=-1):
        self.len = len
        self.seed = seed

        self.arr = [rand.rand() for i in range(len)]

    def rand(self):
        self.seed = (self.seed+1)
        return self.arr[(self.seed%self.len)]/float(self.len)


if __name__ == '__main__':
    n = 50
    # N = n * 100
    ret = []
    test = []
    for i in range(1, n+1):
        gen = CountRandGen(len=i, rand=pru.PyRandGen())
        decks = {tuple(pru.deck(gen)) for i in range(n * 52)}
        ret.append(len(decks))
        test.append( int(np.lcm(i, 50)/50))
        if ret[-1] != test[-1]:
            print("{}: {}, {}".format(i, ret[-1], test[-1]))
    plt.plot(ret)
    plt.show()

