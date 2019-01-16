import numpy as np


def create_card_dict():
    all_suit = ["Heart", "Spade", "Diamond", "Club"]
    card = dict()
    suit_index = 0
    value = 0

    for number in range(52):
        card[number] = dict()
        card[number]["suit"] = suit_index
        card[number]["suit_name"] = all_suit[suit_index]
        card[number]["value"] = value + 1
        if value == 0:
            card[number]["point"] = 4
        elif value == 10:
            card[number]["point"] = 1
        elif value == 11:
            card[number]["point"] = 2
        elif value == 12:
            card[number]["point"] = 3
        else:
            card[number]["point"] = 0

        value += 1
        if value == 13:
            value = 0
            suit_index += 1
    return card


class Table(object):
    heard_index = np.arange(13)
    spade_index = np.arange(13,26)
    diamond_index = np.arange(26,39)
    clover_index = np.arange(40,52)
    all_suit = ["Heart", "Spade", "Diamond", "Club"]
    all_hand = ["hand1", "hand2", "hand3", "hand4"]

    card = create_card_dict()


    def __init__(self, rdn_list):
        self.hand = np.array(rdn_list).reshape((4,13))


    def show_one_hand(self, array=None):
        print(type(None))
        if type(array) == "NoneType":
            array = self.hand[0,:]
        result = dict()
        for suit, suit_name in zip(range(4), Table.all_suit):
            one_suit = array[array//13 == suit]
            result[suit_name] = sorted((one_suit%13 + 1))

        return result

    def show(self):
        result = dict()
        for index, hand_name in zip(range(4), Table.all_hand):
            array = self.hand[index,:]
            result[hand_name] = self.show_one_hand(array)
        return result


    def calculate_point(self, hand=0):
        one_hand = self.hand[hand,:]
        points = 0
        for num in one_hand:
            points += Table.card[num]["point"]
        return points





