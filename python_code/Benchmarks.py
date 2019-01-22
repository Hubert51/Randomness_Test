import PyRandomUtils as prt
import timeit as tt


x = 0
w = 0
s = 0xb5ad4eceda1ce2a9
def msws():
   global x, w
   x *= x
   x %= 2**64-1
   w = (w+s)%2**64-1
   x += w
   x = (x>>32) | (x<<32) % (2**32-1)
   return x

setup = "from __main__ import msws"
script = "msws()"
print(tt.timeit(script, setup, number=100))
