# distutils: language=c++

from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX
from libcpp.algorithm cimport sort
from libcpp cimport bool
import numpy as np
cimport numpy as np
cimport cython



# ctypedef np.int8_t card_t
ctypedef np.npy_intp IDX_t
card = np.int8
result = np.float32

suits = "SHDC"
names = "23456789TJQKA"

feature_string = np.array([
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
    '7-5',
    '7-6',
    'p<8',
    'p8-13',
    'p14-18',
    'p19+'
])

theoretical_probabilities = np.array([
    0.3508,
    0.4434,
    0.1655,
    0.0353,
    0.0047,
    0.00037,
    0.000017,
    0.0000003,
    0.000000003,
    0.000000000006,
    0.0512,
    0.3055,
    0.5380,
    0.1054,
    0.00705+0.00651,
    0.00072,
    0.00109,
    0.000056,
    0.285846,
    0.801244- 0.285846,
    0.975187 - 0.801244,
    1 - 0.975187
    ])

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
    return get_dist(hand)

cdef inline bool greater(int x, int y):
    return x > y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.int8_t[:] __get_dist(card_t[:] hand):
    cdef np.int8_t c
    cdef IDX_t i, j
    cdef np.int8_t[:] dist = np.zeros(4, dtype=np.int8)

    for i in range(13):
        c = hand[i]
        j = (c - 1) // 13
        dist[j] += 1

    sort(&(dist[0]), &(dist[4]), greater)

    return dist

def print_all_dist(deal):
    for hand in np.reshape(deal, (4,13)):
        print(get_dist(hand))


cpdef int get_points(deck_t hand):
    cdef IDX_t i
    cdef int sum_ = 0
    for i in range(len(hand)):
        sum_ += max(0, (((hand[i]-1) % 13)- 8))


    return sum_


def print_features():
    for i in range(len(feature_string)):
        print("{:3d}: {:8}|{:0.13f}".format(i, feature_string[i], theoretical_probabilities[i]))

cdef inline void insert_feature(result_t[:] features, str fname):
    cdef IDX_t idx
    idx = fdict[fname]
    features[idx] += 0.25


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef result_t[:] get_features(card_t[:] deal):
    cdef result_t[:] features = np.zeros(num_features, dtype=result)
    cdef card_t[:] hand
    cdef np.int8_t[:] dist
    cdef IDX_t i, idx
    cdef long longest, shortest, points


    for i in range(4):

        hand = deal[13 * i: 13 * (i+1)]
        dist = __get_dist(hand)
        # print([x for x in dist])

        longest = dist[0]
        points = get_points(hand)
        shortest = dist[3]

        insert_feature(features, 'ls'+str(longest))
        insert_feature(features, 'ss'+str(shortest))

        if longest == 6 and dist[1] == 5:
            insert_feature(features, '6-5')
        if longest == 6 and dist[1] == 6:
            insert_feature(features, '6-6')
        if longest == 7 and dist[1] == 5:
            insert_feature(features, '7-5')
        if longest == 7 and dist[1] == 6:
            insert_feature(features, '7-6')


        if points < 8:
            insert_feature(features, 'p<8')

        elif points < 14:
            insert_feature(features, 'p8-13')
        elif points < 19:
            insert_feature(features, 'p14-18')
        else:
            insert_feature(features, 'p19+')


    return features

