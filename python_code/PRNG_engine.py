import struct
import random
from abc import ABC

class PRNG(ABC):
    '''Abstract base class for all generators, here to define an interface for all random generators'''
    def __init__(self, seed=None):
        pass

    def rand(self):
        pass

   #def randchunk(self, chunksize=1024):
   #   pass

'''
    This is first random generator implemented by ourselves. 
    The algorithm of this RNG is middle-square method, suggested 
    by John von Neumann in 1946. 
    Code adapted from https://en.wikipedia.org/wiki/Middle-square_method
'''
class MS_generator(object):

    def __init__(self, seed=None, digits=4):
        if seed is None:
            seed=(int)(random.random() * (10**digits-2)+1) #get 4 digit number between 1 and 9999
        self.number = seed
        self.digits = digits

    def rand(self):
        if self.number%2==1: #make sure number is even
            self.number+=1
        d2 = self.digits*2
        self.number = int(str(self.number**2).zfill(d2)[self.digits//2:d2-self.digits//2])
        return self.number/(10**self.digits-1) #scaling between 0 and 1




class LCG(object):
    '''
    static unsigned long int next = 1;

    int rand(void) // RAND_MAX assumed to be 32767
    {
        next = next * 1103515245 + 12345;
        return (unsigned int)(next/65536) % 32768;
    }

    void srand(unsigned int seed)
    {
        next = seed;
    }
    '''
