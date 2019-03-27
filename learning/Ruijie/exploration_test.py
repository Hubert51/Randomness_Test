
import numpy as np

# from libc.stdint cimport uint32_t, uint64_t, UINT32_MAX
import numpy as np
import python_code.PyRandomUtils as pru
from pdb import pm
import utils.CardUtils as cu
import numpy as np
import matplotlib.pyplot as plt
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
    0.285846,
    0.801244- 0.285846,
    0.985549 - 0.801244,
    1 - 0.985549
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
    hand = np.reshape(deal, (4,13))[0]
#     for hand in np.reshape(deal, (4,13)):
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
#     print(hand)
#     print(points)
    if points < 8:
        features[fdict['p<8']] += 1
    elif points < 14:
        features[fdict['p8-13']] += 1
    elif points < 19:
        features[fdict['p14-18']] += 1
    else:
        features[fdict['p19+']] += 1
#     print(features)
#     print(hand)

    return features



batch_size = int(36) #number of deals in a batch
num_batches = 500
DECK = np.array(range(1,53), dtype=np.int8)
ts = theoretical_probabilities



def shuffled(gen):
    tmp = np.array(DECK)
    pru.shuffle(gen, tmp)
    # print(tmp)
    return tmp


def make_ts(gen, batches=num_batches, batch_size=batch_size):
    ts = [sum((cu.get_features(shuffled(gen)) for i in range(batch_size))) / batch_size
           for j in range(num_batches)]

    # for i in range(num_batches):
    #     for j in range(batch_size):
    #         deal = shuffled(gen)
    #         ts = cu.get_features( deal )
    return np.array(ts).T



def make_graphs(ts):
    dim = ts.shape[0]
    fig, ax = plt.subplots(dim,1, figsize = (15, 100))
    tp = cu.theoretical_probabilities


    for i in range(dim):
        plt.subplot(dim,1,i+1)
        plt.title(cu.feature_string[i],fontsize=16)
        plt.plot(ts[i] - tp[i])
    plt.show()
    fig.savefig('foo.png' )


def print_means(ts):
    means = np.apply_along_axis(np.mean, 1, ts)
    tp = cu.theoretical_probabilities
    print("{:22}{:^22}{:^22}{:^22}".format("Feature", "p", "p_hat", "p-p_hat"))
    for i in range(len(cu.feature_string)):
        print("{:20} {: 20.18f} {: 20.18f} {: 20.18f}".format(cu.feature_string[i],
                                           tp[i],
                                           means[i],
                                           tp[i]-means[i]))



if __name__ == '__main__':
    # bad prng
    # for i in range(10,20):
    # bad = pru.LCG(mod=2 ** i, a=1140671485, c=128201163, seed=1)

    # deterministic gen
    # sobol = SobolGen(11)

    # good prng
    good = pru.PyRandGen(1)

    # ts_bad = make_ts(bad)
    ts_good = make_ts(good)
    # ts_sobol = make_ts(sobol)

    make_graphs(ts_good)
    # print_means(ts_bad)
