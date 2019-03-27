import sys, os

sys.path.append(os.path.dirname(os.getcwd()))
import utils.toolbox as toolbox
import unittest
import numpy as np

if __name__ == '__main__':
    # models = [ ["Halton"], ]
    # models = [ ["MT"] ]
    models = [["bigdeal", "10000000"]]
    models = [[toolbox.REAL_GAME, "1_year", ["AFT"]]]
    models = [[toolbox.HALTON]]
    # for i in range(10, 30, 5):
    #     models.append(["LCG", i])
    batch_size = 1
    batch_num = int(10000000)

    for model in models:
        result = toolbox.generate_ts(batch_size, batch_num, model)
        if len(model) >= 2:
            file_name = toolbox.MODEL_NAME[model[0]] + model[1] + ".txt"
        else:
            file_name = toolbox.MODEL_NAME[model[0]] + ".txt"

        np.savetxt(file_name, result)
