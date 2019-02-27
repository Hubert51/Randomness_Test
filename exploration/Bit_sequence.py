
# coding: utf-8

# In[1]:


import sys, os
sys.path.append(os.path.dirname(os.getcwd()))


# In[2]:


import importlib
import numpy as np
from pdb import pm
import utils.pbn_parse as pbn_parse
from pprint import pprint
import collections
import matplotlib.pyplot as plt
import utils.CardUtils as cu
import utils.CardUtils as CardUtils
import datetime
import PyRandomUtils as pru
import utils.sobol_seq as ss
from rdrand import RdRandom
from matplotlib.ticker import FuncFormatter
import random



# In[3]:


# Global variable
'''
this is the data size for professor's data, which is real game
batch_size = 36 #number of deals in a batch
num_batches = 717
'''

# following data is larger than real game
batch_size = 100 #number of deals in a batch
num_batches = 3500

# size of deal in file file_name
big_deal_number = 1000000
file_name = "{}.pbn".format(big_deal_number)



DECK = np.array(range(1,53), dtype=np.int8)


# In[4]:


def generate_bit_sequence(ts, tp):
    '''
        ts: is actually probability for one feature
        tp: is theoretical probability for one feature
    '''
    bit_series = (ts-tp).T
    bit_series[ np.where(bit_series>0) ] = 1
    bit_series[ np.where( (bit_series<0) | (bit_series==0) )  ] = 0
    return bit_series

def print_result(feature_series, data_type):
    one_value = len(np.where(feature_series == 1)[0])
    zero_value = len(np.where(feature_series == 0)[0])

    #     print("The theoretical value of ratio between 1 and 0 is 0.5")
    print("The value in {} is {}".format(data_type, one_value / (one_value + zero_value)))
    

def make_ts(gen, batches=num_batches, batch_size=batch_size):
    if ( type(gen) == RdRandom ):
        ts = []
        for j in range(num_batches):
            temp = np.zeros(20)
            for i in range(batch_size):
                one_deal = DECK.copy()
                gen.shuffle( one_deal )
                temp += cu.get_features( (np.array(one_deal) ) )
            temp /= batch_size
            ts.append(temp)
    else:
        ts = [sum((cu.get_features(shuffled(gen)) for i in range(batch_size))) / batch_size
              for j in range(num_batches)]
    return np.array(ts).T


def shuffled(gen):
    tmp = np.array(DECK)
    pru.shuffle(gen, tmp)
    return tmp


class SobolGen(pru.PRNG):
    def __init__(self, seed):
        self.seed = seed
        
    def rand(self):
        r, self.seed = ss.i4_sobol(1, self.seed)
        return r
    
    
def feature_compare(RS, RS_name, tp ):
    result = []
    for index in range(len(RS[0][0,:])):
        print("Compare feature: {}".format(index))
        temp = []
        for i in range(len(RS)):
            feature_series = generate_bit_sequence(RS[i][:, index], tp[index])
            one_value = len(np.where(feature_series == 1)[0])
            zero_value = len(np.where(feature_series == 0)[0])
            print("The value in {} is {}".format(RS_name[i], one_value / (one_value + zero_value)))            
            temp.append(one_value / (one_value + zero_value))
        result.append(temp)
    return result


# In[5]:


def make_graphs_hist(feature_result, RS_name):
    dim = feature_result.shape[0]
    fig, ax = plt.subplots(dim,1, figsize = (15, 100))
    x =  list(range(len(feature_result[0])))

    for i in range(dim):
        plt.subplot(dim,1,i+1)
        plt.bar( x, feature_result[i])
        plt.xticks(x, RS_name)
        coord_x1 = -1
        coord_y1 = 0.5

        coord_x2 = 11
        coord_y2 = 0.5

        plt.plot([coord_x1, coord_x2], [coord_y1, coord_y1], '--', color="red")
        plt.title(CardUtils.feature_string[i])



# In[6]:


tp = CardUtils.theoretical_probabilities
RS = []
RS_name = []



# In[ ]:


# bad PRNG
ts_bads = []
for i in range(10,30,3):
    bad_model = pru.LCG(mod=2**i, a=1140671485, c=128201163, seed=1)
    ts_bad = np.swapaxes( make_ts(bad_model), 0, 1 )
    RS.append(ts_bad)
    RS_name.append("bad2^{}".format(i))
    # ts_bads.append(ts_bad)
    # feature_series = generate_bit_sequence( ts_bad[:,0], tp[0] )
    # print_result(feature_series)





# In[ ]:


# sobol PRNG
sobol = SobolGen(1)
ts_sobol = make_ts(sobol)
ts_sobol = np.swapaxes(ts_sobol, 0, 1)
RS.append(ts_sobol)
RS_name.append("Sobol")


# In[ ]:


# test for good PRNG
good = pru.PyRandGen(100)
ts_good = np.swapaxes( make_ts(good), 0, 1)
RS.append(ts_good)
RS_name.append("Good")


# In[ ]:


# test for hardware RNG
hardware = RdRandom()
ts_hardware = np.swapaxes( make_ts(hardware), 0, 1)
RS.append(ts_hardware)
RS_name.append("Hardware")


# In[ ]:


# real game
result = pbn_parse.get_all_files(tod=["Morning", "Afternoon", "Evening"])
ts = []
for day in sorted(result.keys()):
    ts.append(
        sum((CardUtils.get_features(deal)
             for deal in result[day])) / len(result[day]))
ts = np.array(ts)
RS.append(ts)
RS_name.append("real game")



# In[ ]:


# big deal game
result = pbn_parse.get_deals_from_file("../hand records/{}".format(file_name))



# In[ ]:


total_deal = batch_size*num_batches
last_index = big_deal_number - total_deal - 1
start = random.randint(0,last_index)
end = start + total_deal
result = result[ start:end, : ]


# In[ ]:


result = result.reshape((num_batches, batch_size, 52))
# print(result.shape)


# In[ ]:


result = dict(enumerate(result)) 


# In[ ]:


ts_bigdeal = []
for day in sorted(result.keys()):
    ts_bigdeal.append(
        sum((CardUtils.get_features(deal)
             for deal in result[day])) / len(result[day]))
ts_bigdeal = np.array(ts_bigdeal)
RS.append(ts_bigdeal)
RS_name.append("BigDeal")


# In[ ]:


feature_result = np.array( feature_compare(RS, RS_name, tp) )



# In[ ]:


make_graphs_hist(feature_result, RS_name)


# In[ ]:


# plt.savefig("test1.png")
print(feature_result)


# In[ ]:


error_sum = np.zeros( ( len(RS_name)) )

for one_feature in feature_result:
    for i in range( len(one_feature) ):
        error_sum[i] += abs( one_feature[i] - 0.5 )
print("batch_size is {}".format(batch_size))
print("num_batches is {}".format(num_batches))
data = []

for i in range( len(RS_name) ):
    data.append( (error_sum[i], RS_name[i]) )
data.sort()

for element in data:
    print("{:10}: Sum of deviation is {:.4f}".format(element[1], element[0] ) )

        
        

