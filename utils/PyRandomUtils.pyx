
from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX
import numpy as np
cimport numpy as np
import random
from collections import Collection


cdef class PRNG(object):

    cpdef uint32_t randi(self) except *:
        return 4

    cpdef float rand(self) except *:
        return 0.4 #guaranteed to be random by fair dice roll

    def randint(self, lo, hi):
        """random int in [lo, hi)"""
        cdef float r
        r = self.rand() #idk the best way
        while r==1:
            r = self.rand()
        return int(self.rand() * (hi-lo) + lo)

cdef class PyRandGen(PRNG):

    def __init__(self, seed=None):
        random.seed(seed)

    cpdef float rand(self) except *:
        return random.random()

    cpdef randint(self, lo, hi):
        return random.randrange(start=lo, stop=hi)


cdef class LCG(PRNG):
    cdef readonly uint32_t state
    cdef uint32_t mod
    cdef uint32_t a
    cdef uint32_t c

    def __init__(self, uint32_t mod, uint32_t a, uint32_t c,uint32_t seed):
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
    return LCG(1<<31, 65539, seed, c=0)

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


# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
CARD = np.int8

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.int8_t CARD_T

ctypedef np.npy_intp IDX_t

def shuffle(PRNG gen, np.ndarray[CARD_T, ndim=1] arr):
    cdef IDX_t i, j, n=len(arr)
    for i in range(0, n-2):
        j = gen.randint(i, n)
        arr[i], arr[j] = arr[j], arr[i]

cdef np.ndarray DECK = np.array(range(1,53), dtype=np.int8)

cpdef deck(gen=None):
    tmp = np.array(DECK)
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
