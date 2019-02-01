import os
import random
import sys
from random import SystemRandom
import time
from Card import Card
from Table import Table
import numpy as np


def feature_extraction_test():
    cards = list(range(52))
    random.seed(10)
    results = np.arange(13)
    while True:
        random.shuffle(cards)
        my_table = Table(cards)
        result = my_table.extract_feature()
        if result == None:
            continue
        results += np.array(result)




if __name__ == '__main__':
    feature_extraction_test()

    cards = list(range(52))
    for i in range(10):
        print(i)

    print(i)


    for i in range(100):
        print(i)
    print()
    # n = int(1e7)
    #
    # start = time.time()
    # list1 = [ random.random() for i in range(n)]
    # end = time.time()
    # print("the time for generate {} software random number is {}".format(n, end - start))
    #
    #
    # cryptogen = SystemRandom()
    # list3 = [cryptogen.randrange(3) for i in range(20)]  # random ints in range(3)
    #
    # start = time.time()
    # list2 = [cryptogen.random() for i in range(n)]  # random floats in [0., 1.)
    # end = time.time()
    #
    # print("the time for generate {} hardware random number is {}".format(n, end - start))
    #


    # print(Table.card)

    random.seed(10)
    list1 = list(range(52))
    data = [0, 0, 0, 0]
    itr = 0
    while True:
        random.shuffle(list1)
        my_table = Table(list1)
        for j in range(4):
            points = my_table.calculate_point(j)
            data[j] += points

        itr += 1

        if itr % 1000 == 0 and itr != 0:
            f = open("data1.txt", "a")
            f.write(str(data) + "\n")
            f.close()
            data = [0, 0, 0, 0]

    # print(my_table.show())