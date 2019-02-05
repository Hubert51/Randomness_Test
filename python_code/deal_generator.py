from python_code.Database import Database
import numpy as np
import random


def createCourseInfoTable():
    # if no this database, the code will create one.
    myDB = "Randomness_test"
    db = Database("Ruijie", "12345678", "142.93.59.116", myDB)
    table_name = "contents"
    element = [
        ["id", "varchar(35)"],
        ["courseName", "varchar(35)"],
        ["department", "varchar(5)"],
        ["courseCode", "varchar(10)"],
        ["professor", "varchar(35)"],
        ["max", "int"],
        ["time", "varchar(5)"],
        ["comment1", "varchar(30)"]
    ]
    key = "id"
    db.create_tables( table_name, element, key)

if __name__ == '__main__':
    index = int(1e8)
    random.seed(10)

    cards = list(range(52))
    # state1_str = str(random.getstate())
    # print(len(state1_str))
    random.shuffle(cards)
    random.shuffle(cards)
    random.shuffle(cards)
    print(cards)

    # state = eval(state1_str)
    # random.setstate(state)
    cards = list(range(52))

    random.shuffle(cards)
    print(cards)

    # state = random.getstate()
    #
    # cards = list(range(52))
    # random.shuffle(cards)
    # print(cards)
    #
    # cards = list(range(52))
    # random.setstate(state)
    # random.shuffle(cards)
    # print(cards)

