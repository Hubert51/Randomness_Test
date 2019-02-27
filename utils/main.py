import os
import random
import sys
from random import SystemRandom
import time
from Card import Card
from Table import Table
import numpy as np
from python_code.Database import Database
from python_code.PyRandomUtils import MiddleSquare_WeylSequence as MS
import python_code.PyRandomUtils as Tool
from distutils.core import setup
from Cython.Build import cythonize
import matplotlib.pyplot as plt


ext_options = {"compiler_directives": {"profile": True, "language_level" : "3"}, "annotate": True}
setup(
    ext_modules = cythonize("PyRandomUtils.pyx", **ext_options),
    include_dirs=[np.get_include()]
)

FEATURE_NUM = 15


def set_db(RNG, seed):
    DB_NAME = "Randomness_test"
    TABLE = "descriptor"
    db = Database("Ruijie","12345678","142.93.59.116",DB_NAME)
    state = ""

    if "descriptor" not in db.show_tables(DB_NAME):
        if RNG == "MT":
            random.seed(seed)
            state = random.getstate()
            state = "".join( [str(x) for x in state] )

        db.create_tables( "descriptor",   [ ("id", "MEDIUMINT NOT NULL AUTO_INCREMENT"),
                                            ("key1", "VARCHAR(40)"),
                                            ("seed", "int"),
                                            ("RNG", "VARCHAR(20)"),
                                            ("lastIndex","int"),
                                            ("status", "VARCHAR(8000)"),
                                            ("Notes", "VARCHAR(200)") ], ["id", "key1"] )
        db.insert_data([ RNG+str(seed), seed, RNG, 0, state, "" ], "descriptor")

    TABLE = RNG+str(seed)
    if TABLE not in db.show_tables(DB_NAME):
        db.create_tables(   TABLE,
                            [   ("id", "BIGINT NOT NULL AUTO_INCREMENT"),
                                ("LS4", "CHAR(1)"),
                                ("LS5", "CHAR(1)"),
                                ("LS6", "CHAR(1)"),
                                ("LS7", "CHAR(1)"),
                                ("LS8", "CHAR(1)"),
                                ("LS9", "CHAR(1)"),
                                ("LSMore10", "CHAR(1)"),
                                ("SS3", "CHAR(1)"),
                                ("SS2", "CHAR(1)"),
                                ("SS1", "CHAR(1)"),
                                ("SS0", "CHAR(1)"),
                                ("score0to7", "CHAR(1)"),
                                ("score8to12", "CHAR(1)"),
                                ("score13toAll", "CHAR(1)"),
                                ("hand1Win", "CHAR(1)"),
                                ("hand2Win", "CHAR(1)"),
                                ("hand3Win", "CHAR(1)"),
                                ("hand4Win", "CHAR(1)")
                            ],
                            "id" )
    # state = db.queryColsData("descriptor", [ ["key1", "'MT10'"] ])
    # print(state[0][5])
    # print()


def get_state(RNG, seed):
    pass


def feature_extraction(RNG="MT", seed=0):
    """
    :param RNG:
    :param seed:
    :return:

    The status of MT RNG is almost 7400 characters
    """
    # if RNG == "MT":
    #     random.seed(seed)
    set_db(RNG, seed)

    # check current state of MT RNG using the seed.
    if RNG =="MT":
        state = get_state("MT", seed)

    elif RNG == "MS":
        model = MS()
        # model =

    cards = list(range(52))
    my_table = Table(cards)

    n = int(1e4)
    string = ""
    poses = 0
    negs = 0
    plot_data = []

    for j in range(1000):
        results = np.array([0] * FEATURE_NUM)
        # if string == "10-5":
        #     print()
        i = 0
        while i < n:
            cards = np.arange(52).astype(np.int8)
            # if i == 3778:
            #     print()
            try:
                Tool.shuffle( model, cards )
            except:
                print("ERROR!")
                continue
            my_table = Table(cards)
            result = my_table.extract_feature()
            results += np.array(result)
            i += 1
        true_value = np.array(my_table.get_feat_prob())

        # print(results)

        data = (results/n - true_value) / true_value
        plot_data.append(data[0])

        np.set_printoptions(suppress=True,
                            formatter={'float_kind': '{:0.2f}'.format})
        # print( data )
        # print( np.where( data>0)[0]  )
        pos = len( np.where( data>=0 )[0] )
        neg = len( np.where( data<=0 )[0] )
        if pos > neg :
            poses += 1
        elif pos < neg :
            negs += 1

        string = "{}-{}".format(pos, neg)
        print( "{}-{}".format(pos, neg) )
    print("Poses is {}, Negs is {}".format(poses, negs))
    np.savetxt("plot_data.txt", np.array(plot_data))
    plt.plot( list(range(len(plot_data))), plot_data )
    plt.show()

    sys.exit(1)


if __name__ == '__main__':

    RNG = "MS"
    seed = 10
    file_name = "MS10.txt"


    feature_extraction(RNG, seed)
    feature_extraction(RNG="MT", seed=10 )
'''
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
    total_time = 0
    while True:
        random.shuffle(list1)
        my_table = Table(list1)
        for j in range(4):
            points = my_table.calculate_point(j)
            data[j] += points
        itr += 1

        if itr % 10000000 == 0 and itr != 0:
            f = open("data.txt", "a")
            f.write(str(data) + "\n")
            f.close()
            data = [0, 0, 0, 0]
            total_time += 1
        if total_time == 10000:
            break
   
    # print(my_table.show())

'''