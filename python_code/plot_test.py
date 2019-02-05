import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    # a = np.loadtxt( "plot_data.txt" )

    a = [0,1,1,0,1,0,1]
    x = list(range(len(a)))
    plt.plot( x, a )
    plt.show()