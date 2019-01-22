
from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX
import numpy as np
cimport numpy as np
import random



cdef class PRNG(object):

    cpdef uint32_t randi(self) except *:
        return 4

    cpdef float rand(self) except *:
        return 0.4 #guaranteed to be random by fair dice roll

    def randint(self, lo, hi):
        """random int in [lo, hi)"""
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

    def __init__(self, uint32_t mod, uint32_t a,uint32_t seed):
        if seed%2==0:
            raise ValueError("Seed must be odd")
        self.state=seed
        self.mod = mod
        self.a = a

    cpdef uint32_t randi(self):
        '''Returns an int between 0 and self.mod'''
        self.state = (self.a * self.state) % self.mod
        return self.state

    cpdef float rand(self):
        '''returns a float between 0 and 1'''
        return (<float> self.randi()) / self.mod

cpdef LCG LCG_RANDU(seed):
    return LCG(1<<31, 65539, seed)

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

