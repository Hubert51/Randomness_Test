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

from matplotlib import style
style.use('fivethirtyeight')



DECK = np.array(range(1, 53), dtype=np.int8)
BOTH = 0
MAX = 1
AVG = 2

REAL_GAME = 0
LCG = 1
SOBOL = 2
HALTON = 3
Mersenne_Twister = 4
HARDWARD = 5
BIG_DEAL = 6

MODEL_NAME = {}
MODEL_NAME[0] = "Real_Game"
MODEL_NAME[1] = "LCG"
MODEL_NAME[2] = "SOBOL"
MODEL_NAME[3] = "HALTON"
MODEL_NAME[4] = "Mersenne_Twister"
MODEL_NAME[5] = "HARDWARE"
MODEL_NAME[6] = "BIG_DEAL"




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
    if (type(gen) == RdRandom):
        ts = []
        for j in range(num_batches):
            temp = np.zeros(cu.get_num_features())
            for i in range(batch_size):
                one_deal = DECK.copy()
                gen.shuffle(one_deal)
                temp += cu.get_features((np.array(one_deal)))
            temp /= batch_size
            ts.append(temp)
    else:
        ts = [sum((cu.get_features(shuffled(gen)) for i in range(batch_size))) / batch_size
              for j in range(num_batches)]
    return np.array(ts).T


def generate_ts(batch_size, num_of_batch, para):
    """
    :param batch_size:
    :param num_of_batch:
    :param para: a list [ "name of RNG", para1 for this RNG ]
    :return:
    """

    if para[0] == LCG:
        LCD = pru.LCG(mod=2 ** para[1], a=1140671485, c=128201163, seed=1)
        ts = np.swapaxes(make_ts(LCD, batch_size, num_of_batch), 0, 1)

    elif para[0] == SOBOL:
        # sobol PRNG
        sobol = SobolGen(1)
        ts = make_ts(sobol, batch_size, num_of_batch)
        ts = np.swapaxes(ts, 0, 1)


    elif para[0] == HALTON:
        halton = pru.HaltonGen()
        ts = make_ts(halton, batch_size, num_of_batch)
        ts = np.swapaxes(ts, 0, 1)


    elif para[0] == Mersenne_Twister:
        # test for good PRNG
        MT = pru.PyRandGen(100)
        ts = np.swapaxes(make_ts(MT, batch_size, num_of_batch), 0, 1)

        # In[10]:

        # test for hardware RNG
        hardware = RdRandom()
        ts = np.swapaxes(make_ts(hardware, batch_size, num_of_batch), 0, 1)

    elif para[0] == REAL_GAME:
        # real game
        result = pbn_parse.get_all_files(tod=para[2])
        ts = []
        for day in sorted(result.keys()):
            ts.append(
                sum((CardUtils.get_features(deal)
                     for deal in result[day])) / len(result[day]))
        ts = np.array(ts)

    elif para[0] == "hardware":
        hardware = RdRandom()
        ts = np.swapaxes(make_ts(hardware, batch_size, num_of_batch), 0, 1)

    elif para[0] == "bigdeal":
        # big deal game
        size_of_bigdeal = int(para[1])
        if size_of_bigdeal < batch_size * num_of_batch:
            raise Exception("size of bigdeal is not enough")

        result = pbn_parse.get_deals_from_file("../hand records/{}.pbn".format(size_of_bigdeal))

        total_deal = batch_size * num_of_batch
        last_index = size_of_bigdeal - total_deal
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

    else:
        raise Exception("Unknown RNG")

    return ts


def draw_histgram_slow(tp, seq, window_size=500, bin_size=30):
    start_index = 0
    end_index = start_index + window_size
    x = []
    while (end_index <= len(seq)):
        sub_seq = np.cumsum(seq[start_index:end_index])
        max = 0
        for i in range(window_size):
            diff = np.abs(sub_seq[i] - tp * (i + 1))
            if (diff > max):
                max = diff
        # if index == window_size:
        x.append(max)

        start_index += window_size
        end_index += window_size
    n, bins, patches = plt.hist(x, bin_size, facecolor='blue', alpha=0.5)
    # print("X is {}".format(x))


def draw_histgram(tp, ts_seq, window_size=500, bin_size=30, Max_Avg_flag=BOTH):
    """
    This function will draw one histogram to show deviation of simulation probability
    from theoretical probability in specific window size within a long sequence. The
    data for histogram can be maximum deviation or average deviation.
    :param tp: theoretical probability
    :param ts_seq: simulation probability
    :param window_size: the window size among string sequence
    :param bin_size: the number of bin in histogram
    :param type: the data in diagram is whether "max" or "mean"
    """
    start_index = 0
    end_index = start_index + window_size

    tp_seq = tp * (np.arange(len(ts_seq)) + 1)

    sub_ts_seq = ts_seq[start_index:end_index]
    sub_tp_seq = tp * (np.arange(window_size) + 1)

    x = []
    # x.append( np.max(np.abs(sub_ts_seq - sub_tp_seq) ) )

    while (end_index <= len(ts_seq)):
        sub_ts_seq = np.cumsum(ts_seq[start_index:end_index])
        if Max_Avg_flag == MAX:
            x.append(np.max(np.abs(sub_ts_seq - sub_tp_seq)))
        elif Max_Avg_flag == AVG:
            x.append(np.mean(np.abs(sub_ts_seq - sub_tp_seq)))

        else:
            raise Exception("Unknown Flag!")
        start_index += window_size
        end_index = start_index + window_size

    n, bins, patches = plt.hist(x, bin_size, facecolor='blue', alpha=0.5)
    # print("X is {}".format(x))


def animate(i):
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    pullData = open("sampleText.txt", "r").read()
    dataArray = pullData.split('\n')
    xar = []
    yar = []
    zar = []
    for eachLine in dataArray:
        if len(eachLine) > 2:
            x, y, z = eachLine.split(',')
            xar.append(int(x))
            yar.append(int(y))
            zar.append(int(z))

        ax1.clear()
        ax1.plot(xar, yar)
        ax1.plot(xar, zar)


def draw_feature_distribution(folder, model, Max_Avg_flag=BOTH, window_size=10, bin_size=25):
    """
    This function tries to draw the bin distribution figure. The data is calculating
    deviation
    :param Max_Avg_flag:
    :param folder:
    :param model:
    :param window_size:
    :param bin_size:
    """
    # file to write the feature output. the file is in the same file with model data
    f = open("{}/output.txt".format(folder), "a")
    # theoretical probability for each feature.
    tp = cu.theoretical_probabilities
    model_name = (model.split("."))[0]
    print("The model is {}".format(model_name))

    f.write("The model is {}\n".format(model_name))

    # load data from file
    print("Start loading data from file")
    # time1 = time.time()
    data = np.loadtxt("{}/{}".format(folder, model))
    # print("End loading data from file, spend {} time".format(time.time()-time1))
    # time1 = time.time()

    length = len(data[:, 0])
    xar = np.arange(length)
    dim = len(tp)
    if Max_Avg_flag == BOTH:
        fig, ax = plt.subplots(dim, 2, figsize=(30, 90))
    else:
        fig, ax = plt.subplots(dim, 1, figsize=(15, 90))
    plt.suptitle("The model is {} with window size{}, bin size {}".format(model_name, window_size, bin_size))


    for i in range(len(tp)):
        # time1 = time.time()
        # toolbox.draw_histgram_slow( tp[i], data[:,i], window_size, bin_size )
        # print("End draw slow histgram, spend {} time".format(time.time() - time1))

        yar = tp[i] * (np.arange(data.shape[0]) + 1)
        zar = np.cumsum(data[:, i])

        diff = np.abs(yar - zar)
        S_avg = np.sum(diff) / length
        S_max = np.max(diff)
        # diff = diff.astype(int)
        S_max_index = np.where(diff == np.max(diff))[0][0]

        if Max_Avg_flag == BOTH:
            plt.subplot(dim, 2, 2*i+1 )
            plt.title("{}({}): {}".format(model_name, "max", cu.feature_string[i]), fontsize=16)
            draw_histgram(tp[i], data[:, i], window_size, bin_size, Max_Avg_flag=MAX)
            plt.subplot(dim, 2, 2*i + 2)
            plt.title("{}({}): {}".format( model_name, 'avg', cu.feature_string[i]), fontsize=16)
            draw_histgram(tp[i], data[:, i], window_size, bin_size, Max_Avg_flag=AVG)
            # print("End draw normal histgram, spend {} time".format(time.time() - time1))

        elif Max_Avg_flag == AVG:
            plt.subplot(dim, 1, i)
            plt.title("{}({}): {}".format(Max_Avg_flag, model_name, cu.feature_string[i]), fontsize=16)
            draw_histgram(tp[i], data[:, i], window_size, bin_size, Max_Avg_flag=AVG)

        elif Max_Avg_flag == MAX:
            plt.subplot(dim, 1, i)
            plt.title("{}({}): {}".format(Max_Avg_flag, model_name, cu.feature_string[i]), fontsize=16)
            draw_histgram(tp[i], data[:, i], window_size, bin_size, Max_Avg_flag=MAX)


        print("Feature {:2}: S_avg is {:8.3f}, index of S_avg is {:8}, S_max is {:8.3f}".format(i, S_avg, S_max_index,
                                                                                                S_max))
        f.write(
            "Feature {:2}: S_avg is {:8.3f}, index of S_avg is {:8}, S_max is {:8.3f}\n".format(i, S_avg, S_max_index,
                                                                                                S_max))
    print()
    f.write("\n")
    f.close()
    index = 0
    file_name = "{}{}.png".format(model_name, index)
    all_files = os.listdir(folder)

    # handle situation when file name is already exists.
    while file_name in all_files:
        index += 1
        file_name = "{}{}.png".format(model_name, index)
    file_name = "{}/{}".format(folder, file_name)
    plt.savefig(file_name)
    plt.show()



'''
Unused function.
'''
def TBT_bit_analysis(bit_string, block_len):
    '''
    :param bit_string:
    :param block_len:
    :return:
    '''

    block_num = len(bit_string) // block_len
    new_string = (bit_string[0:block_len*block_num]).reshape((block_num, block_len))


    print(new_string)




