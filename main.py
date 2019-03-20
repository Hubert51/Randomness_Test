"""
This is the main.py

"""
import utils.toolbox as toolbox
import numpy as np









if __name__ == '__main__':
    """
    Current task is to check whether the random sequence is agreeable with the expected value
    """

    # generate the data
    batch_size = 36
    batch_num = 800
    model = [ "MT" ]
    result = toolbox.generate_ts(batch_size, batch_num, model, window_size)
    print(result.shape)

    # load data from other file
    result = np.loadtxt("function_test/hardware")
    print(result.shape)
