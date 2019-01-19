
from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX




cdef class PRNG(object):

    cpdef uint32_t randi(self) except *:
        return 4

    cpdef float rand(self) except *:
        return 4 #guaranteed to be random by fair dice roll


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
        self.state = (self.a * self.state) % self.mod
        return self.state

    cpdef float rand(self):
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
