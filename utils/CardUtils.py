# from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX
import numpy as np
# cimport numpy as np

# ctypedef np.int8_t CARD_T
CARD = np.int8
# ctypedef np.npy_intp IDX_t

suits = "SHDC"
names = "23456789TJQKA"

feature_string = [
    'ls4',
    'ls5',
    'ls6',
    'ls7',
    'ls8',
    'ls9',
    'ls10',
    'ls11',
    'ls12',
    'ls13',
    'ss0',
    'ss1',
    'ss2',
    'ss3',
    '6-5',
    '6-6',
    'p<8',
    'p8-13',
    'p14-18',
    'p19+'
]

fdict = {item : index for index, item in enumerate(feature_string) }

num_features = len(feature_string)


def val_card(card):
    return (card+1)%13

def val_card_name(card):
    return names[card%13]

def suit_card(card):
    return suits[(card - 1) // 13]

def name_card(card):
    return '{}{}'.format(val_card_name(card), suit_card(card))

def get_dist(hand):
    dist = [0, 0, 0, 0]
    for c in hand:
        dist[(c - 1) // 13] += 1
    return sorted(dist, reverse=True)

def print_all_dist(deal):
    for hand in np.reshape(deal, (4,13)):
        print(get_dist(hand))

def get_points(hand):
    return sum((max(0, (card-1) % 13 - 8) for card in hand))


def get_features(deal): 
    features = np.zeros(num_features, dtype=np.float)
    for hand in np.reshape(deal, (4,13)):
        dist = get_dist(hand)

        longest = dist[0]
        features[fdict['ls'+str(longest)]]+=1

        if longest == 6 and dist[1] == 5:
            features[fdict['6-5']] += 1

        if longest == 6 and dist[1] == 6:
            features[fdict['6-6']] += 1

        shortest = dist[-1]
        features[fdict['ss'+str(shortest)]]+=1

        points = get_points(hand)
        if points < 8: 
            features[fdict['p<8']] += 1
        elif points < 14:
            features[fdict['p8-13']] += 1
        elif points < 19:
            features[fdict['p14-18']] += 1
        else:
            features[fdict['p19+']] += 1

    return features / 4

