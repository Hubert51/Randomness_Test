"""
This is the main.py
"""
# import matplotlib
# matplotlib.use('Agg')
import utils.toolbox as toolbox
import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from matplotlib import style
import utils.CardUtils as cu
import os, sys

style.use('fivethirtyeight')


#
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)


def expect(tp, ts):
    xar = np.arange(len(ts))
    yar = [tp]  # theoretical_probabilities
    zar = [ts[0]]  # simulation_probabilities
    for i in range(1, len(ts)):
        yar.append(tp * (i + 1))
        zar.append(zar[i - 1] + ts[i])
    ax1.plot(xar, yar)
    ax1.plot(xar, zar)


def draw_histgram(diff, window_size, bin_size):
    index = window_size
    x = []
    while (index <= len(diff)):
        # if index == window_size:
        x.append(np.max(diff[index - window_size:index]))
        index += window_size
    n, bins, patches = plt.hist(x, bin_size, facecolor='blue', alpha=0.5)


if __name__ == '__main__':

    """
    Current task is to check whether the random sequence is agreeable with the expected value
    """

    # generate the data
    # batch_size = 36
    # batch_num = 800
    # model = [ "MT" ]
    # result = toolbox.generate_ts(batch_size, batch_num, model)
    # print(result.shape)

    # models = [ "MT", "hardware" ]
    folder = "function_test"
    # models = ["LCG25.txt", "LCG15.txt" ]
    models = [ "Real_Game3_month.txt", "Real_Game1_year.txt"]
    # models = [ "HALTON.txt" ]
    tp = cu.theoretical_probabilities
    window_size = 5
    bin_size = 25

    for model in models:
        toolbox.draw_feature_distribution(folder, model, toolbox.BOTH, window_size, bin_size)
    #
    # fig, ax = plt.subplots(15, 2, figsize=(30, 100))
    # plt.show()
