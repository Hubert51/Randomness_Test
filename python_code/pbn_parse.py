#/usr/bin/python3

import numpy as np 
from Card import Card

file_root = '/home/thom/Downloads/hand records'
fname = '/home/thom/Downloads/hand records/Morning/181030m.pbn'

face_to_val = {str(x+2) : x+1 for x in range(8)}
face_to_val.update({'A':1,'T':10,'J':11,'Q':12,'K':13})


def card_to_int(suit, val):
	'''suit in [0,1,2,3]
	val in [23456789TJQKA]'''
	return suit*13+face_to_val[val]

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
				deals.append(deal)

	return deals

