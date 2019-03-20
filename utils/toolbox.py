import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
import pyximport
pyximport.install()


import importlib
import numpy as np
from pdb import pm
import utils.pbn_parse as pbn_parse
from pprint import pprint
import collections
import matplotlib.pyplot as plt
import utils.CardUtils as cu
import utils.CardUtils as CardUtils
import datetime
import utils.PyRandomUtils as pru
import utils.sobol_seq as ss
from rdrand import RdRandom
from matplotlib.ticker import FuncFormatter
import random


DECK = np.array(range(1,53), dtype=np.int8)


class SobolGen(pru.PRNG):
    def __init__(self, seed):
        self.seed = seed

    def rand(self):
        r, self.seed = ss.i4_sobol(1, self.seed)
        return r


def shuffled(gen):
    tmp = np.array(DECK)
    pru.shuffle(gen, tmp)
    return tmp


def make_ts(gen, batch_size, num_batches):
    if ( type(gen) == RdRandom ):
        ts = []
        for j in range(num_batches):
            temp = np.zeros( cu.get_num_features() )
            for i in range(batch_size):
                one_deal = DECK.copy()
                gen.shuffle( one_deal )
                temp += cu.get_features( (np.array(one_deal) ) )
            temp /= batch_size
            ts.append(temp)
    else:
        ts = [sum((cu.get_features(shuffled(gen)) for i in range(batch_size))) / batch_size
              for j in range(num_batches)]
    return np.array(ts).T


def generate_ts( batch_size, num_of_batch, para ):
    """
    :param batch_size:
    :param num_of_batch:
    :param para: a list [ "name of RNG", para1 for this RNG ]
    :return:
    """

    if para[0] == "LCG":
        LCD = pru.LCG(mod=2**para[1], a=1140671485, c=128201163, seed=1)
        ts = np.swapaxes( make_ts(LCD, batch_size, num_of_batch ), 0, 1 )

    elif para[0] == "Sobol":
        # sobol PRNG
        sobol = SobolGen(1)
        ts = make_ts(sobol, batch_size, num_of_batch)
        ts = np.swapaxes(ts, 0, 1)

    elif para[0] == "MT":
        # test for good PRNG
        MT = pru.PyRandGen(100)
        ts = np.swapaxes(make_ts(MT, batch_size, num_of_batch), 0, 1)

        # In[10]:

        # test for hardware RNG
        hardware = RdRandom()
        ts = np.swapaxes(make_ts(hardware, batch_size, num_of_batch), 0, 1)

    elif para[0] in "real game":
        # real game
        result = pbn_parse.get_all_files(tod=["Morning", "Afternoon", "Evening"])
        ts = []
        for day in sorted(result.keys()):
            ts.append(
                sum((CardUtils.get_features(deal)
                     for deal in result[day])) / len(result[day]))
        ts = np.array(ts)

    elif para[0] == "hardware":
        hardware = RdRandom()
        ts = np.swapaxes(make_ts(hardware, batch_size, num_of_batch), 0, 1)

    else:
        # big deal game
        size_of_bigdeal = int( para[1] )
        if size_of_bigdeal < batch_size * num_of_batch:
            raise Exception("size of bigdeal is not enough")



        result = pbn_parse.get_deals_from_file("../hand records/{}.pbn".format(size_of_bigdeal))

        # In[13]:

        total_deal = batch_size * num_of_batch
        last_index = size_of_bigdeal - total_deal - 1
        start = random.randint(0, last_index)
        end = start + total_deal
        result = result[start:end, :]

        result = result.reshape((num_of_batch, batch_size, 52))
        result = dict(enumerate(result))

        ts_bigdeal = []
        for day in sorted(result.keys()):
            ts_bigdeal.append(
                sum((CardUtils.get_features(deal)
                     for deal in result[day])) / len(result[day]))
        ts = np.array(ts_bigdeal)

    return ts


