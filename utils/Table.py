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


all_suit = ["Heart", "Spade", "Diamond", "Club"]
all_hand = ["hand1", "hand2", "hand3", "hand4"]
feature_dict = {
        "longest": {"4":0, "5":1, "6":2, "7":3, "8":4, "9":5, "10": 6},
        "shortest": {"3":7 , "2": 8, "1": 9, "0": 10},
        "score" : [ 11 ] * 7 + [ 12 ] * 5 + [ 13 ] * 25,
        "highest": 14
}
feature_des = ['longestSuit==4', 'longestSuit==5', 'longestSuit==6', 'longestSuit==7', 'longestSuit==8',
               'longestSuit==9', 'shortestSuit==3', 'shortestSuit==2', 'shortestSuit==1', 'shortestSuit==0',
               '0 <= score <= 7', '8 <= score <= 12', '13 <= score']
                  " hand1 == highest score ]"
    feat_prob = [
        0.3508, 0.4434, 0.1655, 0.0353, 0.0047, 0.00037, 0.000017303006,
        0.1054, 0.5380, 0.3055, 0.0512,
        0.285846, 0.446251, 0.267, 0.25
    ]

# card = create_card_dict()      

class Table(object):
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
        one_hand = self.hand[hand, :]
        one_hand = self.hand[hand,:]
        points = 0
        for num in one_hand:
            points += Table.card[num]["point"]
        return points

    def calculate_point2(self, hand=0):
        one_hand = self.hand[hand, :]

        temp =  4 * len(one_hand[one_hand % 13 == 0] ) + \
                3 * len(one_hand[one_hand % 13 == 12] ) + \
                2 * len(one_hand[one_hand % 13 == 11]) + \
                1 * len(one_hand[one_hand % 13 == 10])

        return temp


    def extract_hand_feature(self, hand=0):
        """
        :param hand:
        :return: a list of feature  the size is 1X13 list
        this list has three features
        [ [longest suit], [shortest suit], [score range] ]
        in longest suit list: [4card, 5card, 6card, 7card, 8card, 9card]
        in shortest suit list: [Three card, Doubleton, Singleton, Void]
        in score list: [0~7, 8~14, 15~37]

        """
        suit_list = []
        features = [ 0 ] * 15
        pattern_ = [0,0,0,0]
        for card in one_hand:
            pattern_[(card-1)//13]+=1
        pattern_.sort(reverse=True)
        pattern = "".join( str(x) for x in pattern_)
        # print(pattern)

        # calculate the longest
        if len(pattern) >= 5:
            element = "10"
        else:
            element = pattern[0]
        index = feature_dict["longest"][element]
        # print(index)
        features[index] = 1

        # calculate the shortest
        element = pattern[-1]
        index = feature_dict["shortest"][element]
        # print(index)
        features[index] = 1

        # calculate the score
        score = self.calculate_point(hand)
        index = feature_dict["score"][score]
        features[index] = 1

        # calculate whether this hand is highest score:
        scores = []
        for i in range(4):
            scores.append(self.calculate_point2(i))
        if max(scores) == scores[0]:
            features[self.feature_dict["highest"]] = 1

        # print(features)
        return features

        # temp = np.array(["4432", "4432", "4434"])
        # print(np.where(temp == pattern)[0][1] )
        #
        # print()


    def extract_features(self):
        return sum(self.extract_hand_feature(i) for i in range(4))

    def get_feat_prob(self):
        return self.feat_prob

    def describe_feature(self):
        return feature_des

