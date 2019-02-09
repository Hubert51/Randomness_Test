import importlib
import numpy as np
from pdb import pm
import utils.pbn_parse as pbn_parse
from pprint import pprint
import collections
import matplotlib.pyplot as plt
import utils.CardUtils as CardUtils
import datetime


if __name__ == '__main__':
    result = pbn_parse.get_all_files(tod=["Morning", "Afternoon", "Evening"])
    # print(result)

    ts = []
    for day in sorted(result.keys()):
        ts.append(
            sum((CardUtils.get_features(deal)
                 for deal in result[day])) / len(result[day]))
    ts = np.array(ts)

    print()

