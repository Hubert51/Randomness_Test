import sys

import matplotlib.pyplot as plt


def cal_point(file_name, plot_flag):
    f = open(file_name, "r")
    lines = f.read().strip().split("\n")
    array = []
    data = [0,0,0,0]
    y = [[], [], [], []]
    #
    # with open("data.txt", "r") as ins:
    #     for line in ins:
    #         array.append(line)

    for element in lines:
        line = element.strip()
        one_iteration = eval(line)
        index = one_iteration.index(max(one_iteration))
        data[index] += 1
        for index in range(len(y)):
            y[index].append(data[index])

    x = list( range(len(y[0])) )
    plt.plot(x, y[0])
    plt.plot(x, y[1])
    plt.plot(x, y[2])
    plt.plot(x, y[3])

    if plot_flag:
        plt.show()
        print(data)


if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("use data_handle.py file_name plot_flag")
        print("1 is true, 0 is false")
        sys.exit(1)

    if not sys.argv[2].isdigit() :
        print("plot_flag must be integer")
        sys.exit(1)

    file_name = sys.argv[1]
    plot_flag =bool(int(sys.argv[2]))

    cal_point(file_name, plot_flag)
