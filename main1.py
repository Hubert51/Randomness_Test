"""
This is the main.py
"""
# import matplotlib
# matplotlib.use('Agg')
import csv

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
import pylab as p
from pylab import figure
import warnings
import scipy.stats as st
from scipy import stats
import multiprocessing

KL = 1
KS = 2


# style.use('fivethirtyeight')

CSV_FILE = 0
TXT_FILE = 1


#
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)


def fit_distribution(data):
    y, x = np.histogram(data, bins=15, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    DISTRIBUTIONS = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
        st.cosine,
        st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f, st.fatiguelife,
        st.fisk,
        st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto, st.gennorm, st.genexpon,
        st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat, st.gompertz, st.gumbel_r,
        st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm, st.hypsecant, st.invgamma,
        st.invgauss,
        st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign, st.laplace, st.levy, st.levy_l,
        st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax, st.maxwell, st.mielke, st.nakagami, st.ncx2,
        st.ncf,
        st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw, st.powerlognorm, st.powernorm, st.rdist, st.reciprocal,
        st.rayleigh, st.rice, st.recipinvgauss, st.semicircular, st.t, st.triang, st.truncexpon, st.truncnorm,
        st.tukeylambda,
        st.uniform, st.vonmises, st.vonmises_line, st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    ]
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # print(distribution.name)
                # print(sse)

                # if axis pass in add to plot
                # try:
                #     if ax:
                #         pd.Series(pdf, x).plot(ax=ax)
                #     end
                # except Exception:
                #     pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except:
            pass
    print(best_distribution.name)

    # for distribution in DISTRIBUTIONS:


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


def writeCsv(RList, GList, BList, lists):
    outString = '\r\n'.join([
        ';'.join(map(str, RList)),
        ';'.join(map(str, GList)),
        ';'.join(map(str, BList))
    ])
    print(outString)
    f = open('csv_file.csv', 'wb')
    f.write(outString.encode())
    f.close()


def loadCsv():
    f = open('csv_file.csv', 'rb')
    out = f.read()
    f.close()
    out = out.split('\r\n')
    out = [x.split(';') for x in out]
    return out

# writeCsv(l1, l2, l3)
# print('--')
# print(loadCsv())


def load_histogram_data(folder, models):
    """
    :param folder:
    :param models:
    :return: all_x: [[MT:feature0], [MT: Feature1], ...[LCG: feature0],...]
    """
    all_x = []
    file_flag = TXT_FILE
    # print( filename.split(".")[-1])
    # if ( filename.split(".")[-1] == "txt" ):
    #     file_flag = TXT_FILE
    # elif ( filename.split(".")[-1] == "csv" ):
    #     file_flag = CSV_FILE
    # else:
    #     raise Exception("Unknown file extension")

    files = os.listdir(folder)
    for model in models:
        # if file == "latex_table.txt":
        #     continue
        filename = "{}/{}.txt".format(folder, toolbox.MODEL_NAME[model[0]])
        with open(filename) as f:
            if file_flag == CSV_FILE:
                reader = csv.reader(f, delimiter=' ')
            while (True):
                one_feature_x = []
                line = 0
                for i in range(cu.num_features):
                    one_feature_x.append([])
                try:
                    if file_flag == CSV_FILE:
                        model_info = (next(reader)).split(" ")
                    else:
                        model_info = f.readline().strip()
                    # line += 1
                except:
                    break
                if model_info == '':
                    break

                group_num = 1
                if file_flag == TXT_FILE:
                    model_info = model_info.split(" ")
                    model_name, total_num, group_num = model_info[0], int(model_info[1]), int(model_info[2])
                    # print(model_info)
                # for special case
                # group_num = 1

                for i in range(group_num):
                    for j in range(cu.num_features):
                        # x = next(reader)
                        x = f.readline().strip().split(" ")
                        x = list(map(float, x))
                        one_feature_x[j] += x
                all_x += one_feature_x
    return all_x


def print_KL_result(KL_result, models):
    dash = '-' * 55
    row_num, col_num = KL_result.shape

    for i in range(row_num):
        if i == 0:
            print(dash)
            output = " "*14
            # output += "{:<16s}".format(toolbox.MODEL_NAME[models[i][0]])
            for j in range( len(models)):
                output += "{:<10s}".format(toolbox.MODEL_NAME[models[j][0]])
                # print('{:<10s}{:>4s}{:>12s}{:>12s}'.format(models[i][0], models[i][0], models[i][0], models[i][0]))
            print(output)
            print(dash)

        # print('{:<10s}{:>4d}{:^12s}{:>12.1f}'.format(data[i][0], data[i][1], data[i][2], data[i][3]))
        output = "{:<10s} | ".format(toolbox.MODEL_NAME[models[i][0]])
        for j in range(col_num):
            output += "{:<10f}".format(KL_result[i,j])
        print(output)
    print()


def statics_test(all_y, method, feature_num):
    feature_number, model_num, _ = all_y.shape
    KL_result = np.zeros((model_num, model_num))
    for j in range(model_num):
        for k in range(j+1, model_num):
            sum = 0
            for i in range(feature_num):
                if method == KL:
                    entropy = stats.entropy( all_y[i,j,:], all_y[i,k,:] )
                elif method == KS:
                    entropy = stats.ks_2samp( all_y[i,j,:], all_y[i,k,:] ).statistic
                else:
                    raise Exception("Unknown statics method")
                if entropy != float("inf"):
                    sum+=entropy
            sum /= feature_number
            KL_result[j,k] = sum
            KL_result[k,j] = sum

    # print(KL_result)
    return KL_result


def calculate_chi_square(all_y):
    feature_number, model_num, _ = all_y.shape
    chi_square_result = np.zeros((model_num, model_num))
    for j in range(model_num):
        for k in range(model_num):
            if j == k:
                continue
            sum = 0
            for i in range(3,4):
                obversed = all_y[i,j,:]
                expected = all_y[i,k,:]
                index = np.where(expected==0)
                expected = np.delete(expected, index)
                obversed = np.delete(obversed, index)
                sum += stats.chisquare( obversed, expected ).statistic
            chi_square_result[j,k] = sum
    return chi_square_result


def KS_test(all_x, models, cal_feature_num):
    model_num = len(models)
    total_feature_num = cu.num_features
    pvalue_result = np.ones((model_num, model_num))
    statics_result = np.ones((model_num, model_num))

    for j in range(model_num):
        for k in range(j+1, model_num):
            pvalue_sum = 0
            statics_sum = 0
            for i in range(cal_feature_num):
                result = stats.ks_2samp( all_x[i+j*total_feature_num], all_x[i+k*total_feature_num] )
                if result != float("inf"):
                    pvalue_sum += result.pvalue
                    statics_sum += result.statistic
            pvalue_sum /= cal_feature_num
            statics_sum /= cal_feature_num

            pvalue_result[j,k] = pvalue_result[k,j] = pvalue_sum
            statics_result[j,k] = statics_result[k,j] = statics_sum
    return pvalue_result, statics_result


def get_bin_data(all_x, models, bin_size, folder, draw_flag = False, cumsum = False):
    all_y = []
    all_bincenters = []
    length = len(all_x)
    model_size = int(length/cu.num_features)
    for i in range(cu.num_features):
        one_feature_data = []
        # calculate the binEdges for all data for one feature
        for j in range(model_size):
            x = all_x[i+cu.num_features*j]
            one_feature_data += x
        _, binEdges = np.histogram(one_feature_data, bins=bin_size)
        binEdges_simple, binEdges = toolbox.find_concentrated_bin_edge(one_feature_data, bin_size)
        binEdges_partial = np.copy(binEdges)
        temp_index = 0
        for edge in [binEdges_simple, binEdges_partial, binEdges]:
            temp_y = []
            for j in range(model_size):
                model = models[j]
                x = all_x[i+cu.num_features*j]
                y = np.digitize(x, edge)
                y, wrong_edge = np.histogram(x, bins=edge)
                y = [float(k) / sum(y) for k in y]
                if temp_index == 1:
                    y = np.array(y)
                    y[np.where((y>0.5) )] = 0
                bincenters = 0.5 * (edge[1:] + edge[:-1])
                wrong_bincenters = 0.5 * (wrong_edge[1:] + wrong_edge[:-1])

                temp_y.append(y)
                all_bincenters.append(bincenters)
                if draw_flag == True:
                    if model[0] == toolbox.THREE_MONTH or model[0] == toolbox.ONE_YEAR:
                        if model[1] == '3_month':
                            p.plot(bincenters, y, ':k', label=model[1], linewidth=1, density=True, histtype='step', cumulative=True)
                        else:

                            p.plot(bincenters, y, '-', label=model[1], linewidth=1,density=True, histtype='step',
                           cumulative=True)
                    else:
                        p.plot(bincenters, y, '-', label=toolbox.MODEL_NAME[model[0]], linewidth=1, density=True, histtype='step',
                           cumulative=True)
                # print("bin center is ", bincenters)
                # print("wrong bin center is ", wrong_bincenters)

            if (temp_index == 0):
                title = "{}/Feature{}-{}-simple".format(folder, i, cu.feature_string[i])
                all_y.append(temp_y)


            elif (temp_index == 1):
                title = "{}/Feature{}-{}-partial".format(folder, i, cu.feature_string[i])

            else:
                # contains full data
                title = "{}/Feature{}-{}-full".format(folder, i, cu.feature_string[i])

            if draw_flag == True:
                leg = p.legend()
                for line in leg.get_lines():
                    line.set_linewidth(1)
                p.title(title)
                p.savefig(title + ".png")
                p.show()
            temp_index += 1

    return np.array(all_y)



def get_bin_data2(all_x, models, bin_size, folder, draw_flag = False, cumsum = False):
    # plt.style.use('ggplot')

    all_y = []
    all_bincenters = []
    all_mean = []
    all_std = []
    all_X = []
    length = len(all_x)
    model_size = int(length/cu.num_features)
    fig, ax = plt.subplots(figsize=(20, 10))

    for i in range(cu.num_features):
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.rcParams['lines.linewidth'] = 3.5
        X = []
        label = []
        std = []
        mean = []
        for j in range(model_size):
            model = models[j]
            x = all_x[i+cu.num_features*j]
            x = np.array( sorted(x) )
            std.append(np.std(x))
            mean.append(np.mean(x))
            # user 99% of data, remove the tail data.
            X.append( x[0:int(0.99*len(x))] )
            # X.append(x[:])
            label.append( toolbox.MODEL_NAME[model[0]] )
        if draw_flag:
            ax.hist(X, linewidth=2, bins=1000000, density=True, histtype='step', cumulative=True, label=label)
            ax.legend()
            ax.set_title("PMF for Feature{}-{}".format(i, cu.feature_string[i]))
            file_name = "{}/PMF for Feature{}-{}".format(folder, i, cu.feature_string[i])
            p.savefig(file_name + ".png")
            plt.show()
        all_std.append(std)
        all_mean.append(mean)
        all_X.append(X)

    return all_X, all_mean, all_std


if __name__ == '__main__':
    """
    Current task is to check whether the random sequence is agreeable with the expected value
    all t
    """


    flag = 1 # analysis
    # flag = 0 # get data
    # flag = 2 # get data and analysis

    batch_num = 320000
    toolbox.TIME_PERIOD = 16000   # 1600000
    batch_size = 1
    window_size = 32
    bin_size = 20
    folder = "data{}-{}".format(batch_num, window_size)

    models = [
        [toolbox.Mersenne_Twister],
        # [toolbox.LCG, 20],
        [toolbox.LCG, 25],
        [toolbox.BIG_DEAL, "100000000"],
        [toolbox.HARDWARD],
        [toolbox.HALTON],
        [toolbox.THREE_MONTH, "3_month", ["Morning", "Afternoon", "Evening"] ],
        # [toolbox.ONE_YEAR, "1_year", ["AFT"]],
    ]

    if flag == 0 or flag == 2:
        print("the data point is {}, the window size is {}".format(batch_num, window_size))
        try:
            os.mkdir(folder)
        except:
            pass
        # batch_num = 2000000
        jobs = []
        for model in models:
            args = (model, batch_size, batch_num, folder, window_size, bin_size)
            process = multiprocessing.Process(target=toolbox.generate_histogram_data, args=args)
            jobs.append(process)

        for j in jobs:
            j.start()
            # j.join()

        # Ensure all of the threads have finished
        for j in jobs:
            j.join()

        print("List processing complete.")

    if flag == 1 or flag == 2:
        csv.field_size_limit(256 << 15)
        # all_x = load_histogram_data("csv_file3_11.csv")
        # cu.num_features = 19
        temp = []
        all_x = load_histogram_data(folder, models)

        pvalue_result, statics_result = KS_test(all_x, models, 10)
        # toolbox.latex_table_generation(folder, pvalue_result, models, "pvalue for origin data")
        # toolbox.latex_table_generation(folder, pvalue_result, models, "statistic for origin data")


        # print_KL_result(pvalue_result, models)
        # print_KL_result(statics_result, models)

        [all_x, all_mean, all_std] = get_bin_data2(all_x, models, bin_size = 50, folder=folder, draw_flag=True )
        # toolbox.latex_table_generation(folder, [all_mean,all_std], models, "Mean and standard deviation", False)
        sys.exit(1)
        KL_result = statics_test(all_y, KL, feature_num=3)
        # toolbox.latex_table_generation(folder, KL_result, models, caption="KL distance")
        print_KL_result(KL_result, models)


        # chi_square_result = calculate_chi_square(all_y)
        # print_KL_result(chi_square_result, models)

        ks_result = statics_test(all_y, KS, feature_num=3)
        print_KL_result(ks_result, models)
        # toolbox.latex_table_generation(folder, ks_result, models, caption="KS test")



    # for i in range(len(all_x)):
    #     fit_distribution( all_x[i] )
    # sys.exit(1)


        # toolbox.generate_histogram_data(model, batch_size, batch_num, folder, window_size, bin_size)

    # analysis_model(all_ts)
    # exit(1)
    #
    # # generate the data   
    # # batch_size = 36
    # # batch_num = 800
    # # model = [ "MT" ]
    # # result = toolbox.generate_ts(batch_size, batch_num, model)
    # # print(result.shape)
    #
    # models = [ "MT800.txt" ]
    # folder = "function_test"
    # # models = ["LCG25.txt", "LCG15.txt" ]
    # # models = [ "Real_Game3_month.txt", "Real_Game1_year.txt"]
    # # models = [ "HALTON.txt" ]
    # tp = cu.theoretical_probabilities
    # window_size = 20
    # bin_size = 20
    #
    #
    # for model in models:
    #     toolbox.draw_feature_distribution(folder, model, toolbox.BOTH, window_size, bin_size)
    # #
    # # fig, ax = plt.subplots(15, 2, figsize=(30, 100))
    # # plt.show()



"""

        # for i in range(0, len(models), len(all_x)):
        #     temp.append( [all_x[i:i+len(models)]] )
        #
        # data = np.array(temp)
        # print(data.shape)
        # np.reshape(data, (len(models), cu.num_features, len(all_x[0])))
        # ks_result = statics_test(data, KS, feature_num=3)"""