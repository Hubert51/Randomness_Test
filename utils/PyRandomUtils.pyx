#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
import numpy as np
import random
import utils
import utils.CardUtils as cu
from multiprocessing import Pool, Queue, Lock, Process

from   utils.CardUtils cimport get_features, card_t, deck_t
import matplotlib.pyplot as plt

from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX
cimport numpy as np


cdef class PRNG(object):

    cpdef uint32_t randi(self) except *:
        return 4

    cpdef float rand(self) except *:
        return 0.4 #guaranteed to be random by fair dice roll

    def randint(self, lo, hi):
        """random int in [lo, hi)"""
        cdef int r
        r = int(self.rand() * (hi-lo) + lo) #idk the best way
        while r==hi:
            r = int(self.rand() * (hi-lo) + lo)
        return r

cdef class PyRandGen(PRNG):

    def __init__(self, seed=None):
        random.seed(seed)

    cpdef float rand(self) except *:
        return random.random()

    cpdef randint(self, lo, hi):
        return random.randrange(start=lo, stop=hi)


cdef class LCG(PRNG):
    cdef readonly uint64_t state
    cdef uint64_t mod
    cdef uint64_t a
    cdef uint64_t c

    def __init__(self, uint64_t mod, uint64_t a, uint64_t c,uint64_t seed):
        if seed%2==0:
            raise ValueError("Seed must be odd")
        self.state=seed
        self.mod = mod
        self.a = a
        self.c = 0

    cpdef uint32_t randi(self):
        '''Returns an int between 0 and self.mod'''
        self.state = (self.a * self.state + self.c) % self.mod
        return self.state

    cpdef float rand(self):
        '''returns a float between 0 and 1'''
        return (<float> self.randi()) / self.mod

cpdef LCG LCG_RANDU(seed):
    return LCG(1<<31, 65539, 0, seed)

cdef class MiddleSquare_WeylSequence(PRNG):
    """Middle Square generator using the Weyl sequence
    https://en.wikipedia.org/wiki/Middle-square_method"""
    cdef readonly uint64_t x, seed, w


    def __init__(self, int seed=0xb5ad4eceda1ce2a9):
        if seed%2==0:
            raise ValueError("Seed must be odd")
        self.seed=seed
        self.x = self.w = 0

    cpdef uint32_t randi(self):
        self.x *= self.x
        self.w += self.seed
        self.x += self.w
        self.x = (self.x >> 32) | (self.x << 32)
        return self.x


    cpdef float rand(self):
        return (<float>self.randi())/UINT32_MAX


CARD = np.int8

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
# ctypedef np.int32_t card_t
#
ctypedef np.npy_intp IDX_t
#
# ctypedef CARD_T[:] deck_t

def shuffle(PRNG gen, np.ndarray[card_t, ndim=1] arr):
    cdef IDX_t i, j, n=len(arr)
    for i in range(0, n-2):
        j = gen.randint(i, n)
        arr[i], arr[j] = arr[j], arr[i]

cdef np.ndarray DECK = np.array(range(1,53), dtype=CARD)

cpdef deck_t deck(gen=None):
    tmp = np.array(DECK, dtype=CARD)
    if gen:
        shuffle(gen, tmp)
    return tmp

#Halton sequence
def next_prime():
    def is_prime(int num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True

    cdef int prime = 3
    while 1:
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(int n, int base=2):
    cdef double vdc = 0
    cdef int denom = 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def halton_sequence(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return seq

class HaltonGen(PRNG):

    def __init__(self, base=2, count=0):
        self.base = base
        self.count = count

    def rand(self):
        self.count += 1
        return vdc(self.count, self.base)

    def seed(self, n):
        # skips the first n results
        self.count = n


#Common used functions
def make_ts(gen, batches=1000, batch_size=36):
    ts = [sum((get_features(deck(gen)) for i in range(batch_size))) / batch_size
           for j in range(batches)]
    return np.array(ts).T

def make_ts_no_batch(PRNG gen, int decks=5):
    cdef Py_ssize_t i
    cdef np.ndarray ret = np.zeros((decks, len(cu.theoretical_probabilities)), dtype=float)
    for i in range(decks):
        ret[i] = np.array(cu.get_features(deck(gen)))
    return ret.T

def f(deck):
    return list(cu.get_features(deck))

def make_ts_no_batch_mp(PRNG gen, int n=5, int cores=4):
    cdef Py_ssize_t i
    cdef np.ndarray ret = np.zeros((n, len(cu.theoretical_probabilities)), dtype=float)

    decks = np.array([deck(gen) for i in range(n)])


    with Pool(cores) as pool:
        ret_ = pool.map(f, decks)
    # ret_ = map(f, decks)

    for i, e in enumerate(ret_):
        ret[i] = e

    return ret.T


def make_graphs(ts):
    dim = ts.shape[0]
    fig, ax = plt.subplots(dim,1, figsize = (15, 100))

    for i in range(dim):
        plt.subplot(dim,1,i+1)
        plt.title(cu.feature_string[i],fontsize=16)
        plt.plot(ts[i] - cu.theoretical_probabilities[i])

def print_means(ts):
    means = np.apply_along_axis(np.mean, 1, ts)
    tp = cu.theoretical_probabilities
    print("{:22}{:^22}{:^22}{:^22}".format("Feature", "p", "p_hat", "p-p_hat"))
    for i in range(len(cu.feature_string)):
        print("{:20} {: 20.18f} {: 20.18f} {: 20.18f}".format(cu.feature_string[i],
                                           cu.theoretical_probabilities[i],
                                           means[i],
                                           cu.theoretical_probabilities[i]-means[i]))