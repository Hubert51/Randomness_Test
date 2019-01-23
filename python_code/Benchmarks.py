import PyRandomUtils as prt
import timeit as tt
import numpy as np
import random

def msws():
    x = 0
    w = 0
    s = 0xb5ad4eceda1ce2a9
    while True:
      x *= x
      x %= 2**64-1
      w = (w+s)%2**64-1
      x += w
      x = (x>>32) | (x<<32) % (2**32-1)
      yield x


def shuffle(arr):
   n = len(arr)
   for i in range(0, n - 2):
      j = random.randrange(i, n)
      arr[i], arr[j] = arr[j], arr[i]



setup = "from __main__ import msws; gen=msws()"
script = "next(gen)"
reps = 10**7
msws_python = tt.repeat(script, setup, number=reps, repeat=3)
print("{} repititions of msws in Python (best of 3): {}".format(reps, min(msws_python)))
print("Time per repetition: {}".format(min(msws_python)/reps))
print()

msws_cython = tt.repeat("gen.randi()",
             "from PyRandomUtils import MiddleSquare_WeylSequence as msws; gen = msws()", number=reps, repeat=3)
print("{} repititions of msws in Cython (best of 3): {}".format(reps, min(msws_cython)))
print("Time per repetition: {}".format(min(msws_cython)/reps))

deck = np.array(range(1,53), dtype=np.int8)
def make_deck():
   return np.array(deck)

reps2 = 10**5
shuffles_py = tt.repeat("shuffle(deck)", setup = "from __main__ import shuffle, make_deck;deck=make_deck()", number=reps2, repeat=3)
print("{} reps of shuffling in Python: {}".format(reps2, min(shuffles_py)))
print()
shuffles_cy = tt.repeat("shuffle(gen, deck)",
                        setup = "import numpy as np;"
                                "from PyRandomUtils import MiddleSquare_WeylSequence as msws, shuffle; "
                                "gen = msws();"
                                "deck = np.array(range(1,53), dtype=np.int8)", number=reps2, repeat=3)
print("{} reps of shuffling in Cython: {}".format(reps2, min(shuffles_cy)))


