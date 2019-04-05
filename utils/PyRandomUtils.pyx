#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION" [-Wcpp]
import numpy as np
import random
import utils
import utils.CardUtils as cu
from utils.CardUtils import card, result
from multiprocessing import Pool, Queue, Lock, Process

from   utils.CardUtils cimport get_features, card_t, deck_t, result_t, timeseries_t
import matplotlib.pyplot as plt

from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX
cimport numpy as np

import cython
cimport cython



ctypedef np.npy_intp IDX_t

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



def shuffle(PRNG gen, np.ndarray[card_t, ndim=1] arr):
    cdef IDX_t i, j, n=len(arr)
    for i in range(0, n-2):
        j = gen.randint(i, n)
        arr[i], arr[j] = arr[j], arr[i]

cdef np.ndarray DECK = np.array(range(1,53), dtype=card)

cpdef deck_t deck(gen=None):
    tmp = np.array(DECK, dtype=card)
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

cpdef vdc(size_t n, size_t base=2):
    cdef double vdc = 0
    cdef int denom = 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]

cdef long maxprime = 1000
cdef long[:] primes = primesfrom2to(maxprime)


cdef float[:,:] halton_sequence(size_t size, size_t dim, size_t start=0):

    cdef float[:,:] seq
    cdef size_t base, d, i, j

    if len(primes) < dim:
        global primes, maxprime
        maxprime *= 2
        primes = primesfrom2to(maxprime)

    seq = np.zeros((dim, size), dtype=np.float32)
    for d in range(dim):
        base = primes[d]
        j = 0
        for i in range(start, start+size):
            seq[d][j] = vdc(i, base)
            j += 1
        # seq[d] = [vdc(i, base) for i in range(start, start+size)]
    return np.array(seq)

cdef class HaltonGen(PRNG):

    cdef size_t base, count

    def __init__(self, base=2, count=0):
        self.base = base
        self.count = count

    cpdef float rand(self):
        self.count += 1
        return vdc(self.count, self.base)

    def seed(self, n):
        # skips the first n results
        self.count = n

cdef class HaltonGen_Deck(PRNG):

    cdef size_t batch_size
    cdef size_t nprimes, count, idx
    cdef float[:,:] sequence

    def __init__(self, size_t count=100, size_t nprimes=52, size_t batch_size=10**6):

        self.batch_size = batch_size
        self.nprimes = nprimes
        self.count = count

        self.sequence=halton_sequence(size=self.batch_size, dim=self.nprimes, start=self.count)
        self.idx=0

    cpdef float rand(self):
        self.idx += 1
        if self.idx==self.nprimes:
            self.idx=0
            self.count += 1

        if self.count%self.batch_size==0 and self.idx==0:
            self.sequence=halton_sequence(self.batch_size, dim=self.nprimes, start=self.count)

        return self.sequence[self.idx][self.count % self.batch_size]


    def seed(self, n):
        # skips the first n results
        self.count = n




#Common used functions
def make_ts(gen, batches=1000, batch_size=36):
    ts = [sum((get_features(deck(gen)) for i in range(batch_size))) / batch_size
           for j in range(batches)]
    return np.array(ts).T

@cython.boundscheck(False)
@cython.wraparound(False)
def make_ts_no_batch(PRNG gen, int decks=5):
    cdef Py_ssize_t i, j, l = len(cu.theoretical_probabilities)
    cdef timeseries_t ret = np.zeros((decks, l), dtype=result)
    cdef result_t[:] features

    for i in range(decks):
        features = cu.get_features(deck(gen))
        for j in range(l):
            ret[i, j] = features[j]
    return ret.T

def f(deck):
    return list(cu.get_features(deck))

def make_ts_no_batch_mp(PRNG gen, int n=10**6, int cores=4):
    cdef Py_ssize_t i
    cdef np.ndarray ret = np.zeros((n, len(cu.theoretical_probabilities)), dtype=result)

    decks = np.array([deck(gen) for i in range(n)])


    with Pool(cores) as pool:
        ret_ = pool.map(f, decks, chunksize=4096)
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