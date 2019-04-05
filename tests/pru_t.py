import utils.CardUtils as cu
import utils.PyRandomUtils as pru
from datetime import datetime


def time_test(gen, n):
    t = datetime.now()
    pru.make_ts_no_batch(gen, n)
    dt = datetime.now() - t

    return dt

def time_test_suite(gen_func, maxexp, name):
    print("{}:".format(name))
    for i in range(maxexp):
        gen = gen_func()
        print("10**{}: {}".format(i, time_test(gen, 10**i)))


if __name__ == '__main__':
    n = 7
    # time_test_suite(lambda: pru.LCG_RANDU(1), n, "lcg")
    # time_test_suite(pru.PyRandGen, n, 'python')
    # time_test_suite(pru.HaltonGen, n, 'Halton')
    # time_test_suite(lambda: pru.HaltonGen_Deck(batch_size=10**6), n, 'HaltonDeck')
    print(time_test(pru.HaltonGen(base=3), 10**5))
    print(time_test(pru.HaltonGen(base=31), 10**5))
    print(time_test(pru.HaltonGen(base=51), 10**5))
    print(time_test(pru.HaltonGen(base=3), 10**5))
    print(time_test(pru.HaltonGen(base=31), 10**5))
    print(time_test(pru.HaltonGen(base=51), 10**5))