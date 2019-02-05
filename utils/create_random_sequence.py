import random
import sys
import numpy as np

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Please use create_random_sequence.py file_name, number, seed")

    file_name = sys.argv[1]
    number = int(sys.argv[2])
    seed = int(sys.argv[3])

    f = open('test.txt', 'ab')

    file_name = file_name + str(seed) + ".txt"
    random.seed(seed)

    np.random.seed(seed)
    a = np.random.permutation(52).reshape((1,52))
    print(a)
    # np.savetxt(f, a, fmt='%1d')

    for i in range( int(1e8)):
        a = np.random.permutation(52).reshape((1, 52))
        np.savetxt(f, a, fmt='%1d', newline='\n')



