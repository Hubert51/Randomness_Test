#/usr/bin/python3

import numpy as np 
from Card import Card
from os import listdir
from os.path import join
import datetime
import re

file_root = '../hand records'

face_to_val = {str(x + 2): x + 1 for x in range(8)}
face_to_val.update({'T': 9, 'J': 10, 'Q': 11, 'K': 12, 'A': 13})

re_ymd = re.compile(r'^(20)?\d\d\d\d\d\d\D.*')
re_mdy = re.compile(r'^\d\d\d\d\d\d\d\d\D.*')

bad_filenames = {
    "1810110CGWedAft.pbn": datetime.datetime(2018, 10, 11),
    "0752017CGWedAft.pbn": datetime.datetime(2017, 7, 5)
}


def card_to_int(suit, val):
    '''suit in [0123]
    val in [23456789TJQKA]'''
    return suit * 13 + face_to_val[val]


def get_deals_from_file(fname):
    deals = []
    with open(fname) as fin:
        for line in fin:
            if line.startswith("[Deal "):
                deal = []
                dstring = line.split('"')[1][2:]
                hands = dstring.split(' ')
                for hand in hands:
                    for i, suit in enumerate(hand.split('.')):
                        for card in suit:
                            deal.append(card_to_int(i, card))
                deals.append(np.array(deal, dtype=np.int8))

    return np.array(deals)


def date_from_fname(fname):
    if re_ymd.match(fname):
        if fname.startswith('20'):
            ret = datetime.datetime.strptime(fname[:8], "%Y%m%d")
        else:
            ret = datetime.datetime.strptime(fname[:6], "%y%m%d")
    elif re_mdy.match(fname):
        ret = datetime.datetime.strptime(fname[:8], "%m%d%Y")
    elif fname in bad_filenames:
        return bad_filenames[fname]
    else:
        ret = None
    if not ret:
        raise RuntimeError("Failed to parse filename {}".format(fname))
    return ret


def get_all_files(file_root=file_root, tod="Afternoon"):
    files = listdir(join(file_root, tod))

    return {date_from_fname(fname):
            get_deals_from_file(join(file_root, tod, fname))
            for fname in files
            if fname.endswith('.pbn')}
