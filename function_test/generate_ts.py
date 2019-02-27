import utils.toolbox as toolbox
import unittest
import numpy as np



if __name__ == '__main__':
    models = [["Sobol"], ["MT"], ["real game"], ["hardware"], ["bigdeal", 3000000]]
    for i in range(10, 30):
        models.append(["LCG", i])

    batch_size = 36
    batch_num = 800

    for model in models:
        result = toolbox.generate_ts(batch_size, batch_num, model)
        file_name = ''.join(str(x) for x in model)
        np.savetxt(file_name, result)
