import numpy as np
import itertools
import random
import pdb
from collections import Collection
import scipy.stats as st



def binom_var(n,p):
    return np.sqrt(n*p*(1-p))

def z_test(sigma, n, mean, mu):
    error = sigma/np.sqrt(n)
    z = (mean-mu)/error
    return 1-2*(1-st.norm.cdf(abs(z)))

def word_test(arr, word_len=3, verbose=False, alpha=0):

    N = len(arr)

    counts = {}

    for l in range(1, word_len+1):
        counts.update({tuple(x): 0 for x in itertools.product((0, 1), repeat=l)})
        #         pdb.set_trace()
        for i in range(0,N - l,l):
            key = tuple(arr[i: i + l])
            counts[key] = counts[key] + 1

    if verbose:
        print("{:10}|{:10}|{:10}|{:10}".format("Tuple", "Observed", "Expected", "Diff"))
        for key in counts.keys():
            if not any(key):
                print('-' * 50)

            n = N//len(key)
            p = 1/(2 ** len(key))
            expected = n*p
            test_result = z_test(binom_var(n, p),n,counts[key]/n,p)
            print("{:10}|{:10}|{:10}|{:+10}|{:.10f}{}".format(str(key),
                                                    str(counts[key]),
                                                    str(expected),
                                                    counts[key] - expected,
                                                    test_result,
                                                    "" if alpha == 0 else "|"+str((1-alpha) < test_result)))
    return counts
