import utils.CardUtils as cu
import utils.PyRandomUtils as pru
from datetime import datetime

if __name__ == '__main__':
    n = 2
    gen = pru.LCG_RANDU(1)

    pru.make_ts_no_batch(gen, n)

    print("Done")