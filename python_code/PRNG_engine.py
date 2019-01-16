
import random



'''
    This is first random generator implemented by ourselves. 
    The algorithm of this RNG is middle-square method, suggested 
    by John von Neumann in 1946. 
'''
class MS_generator(object):

    def __init__(self):
        pass


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
