

if __name__ == '__main__':

    f = open("data.txt", "r")
    data = [0,0,0,0]
    array = []

    with open("data.txt", "r") as ins:
        for line in ins:
            array.append(line)

    for element in array:
        line = element.strip()
        list = eval(line)
        # print(max(list))
        index = list.index(max(list))
        data[index] += 1
    print(data)
