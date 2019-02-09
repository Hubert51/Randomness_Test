from learning.Ruijie import exploration_test as et
import python_code.PyRandomUtils as pru
import numpy as np
import itertools


def sign( feature ):
    feature[ np.where(feature>0) ] = 1
    feature[ np.where( (feature<0) | (feature==0) ) ] = 0


def bit_test( feature, bit_len ):
    sign(feature)
    prod = list(itertools.product([1, 0], repeat=bit_len))
    prod = list(map(list, prod))  # => [1,2,3]
    stats = [ 0 ] * len( prod )
    print(feature)
    print(prod)


    # prod = np.array( prod ).astype( np.int8 )

    for i in range( len( feature ) - bit_len + 1 ):
        bit_stream = ( feature[i:i+bit_len] )
        bit_stream = list(map(int, bit_stream))  # => [1,2,3]

        index = prod.index(bit_stream)
        stats[index] += 1
        # print(bit_stream)
    return stats


if __name__ == '__main__':
    good = pru.PyRandGen(1)
    bad = pru.LCG(mod=2 ** 10, a=1140671485, c=128201163, seed=1)

    ts_good = et.make_ts(gen=bad, batch_size=300, batches=100)

    tp = et.theoretical_probabilities.reshape(20,1)

    result = ts_good - tp
    one_feature = result[0,:]
    stats = bit_test(one_feature, 3)

    print(stats)