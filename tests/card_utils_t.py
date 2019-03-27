import utils.CardUtils as cu
import utils.PyRandomUtils as pru

if __name__ == '__main__':
    deck = pru.deck()
    print([x for x in cu.get_features(deck)])