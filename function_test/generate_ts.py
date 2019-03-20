import utils.toolbox as toolbox
import unittest
import numpy as np



if __name__ == '__main__':
    models = [ ["MT"], ["hardware"] ]
    # models = []
    for i in range(10, 30, 5):
        models.append(["LCG", i])

    batch_size = 1
    batch_num = int(10 ** 7)

    for model in models:
        result = toolbox.generate_ts(batch_size, batch_num, model)
        file_name = ''.join(str(x) for x in model) + ".txt"
        np.savetxt(file_name, result)
