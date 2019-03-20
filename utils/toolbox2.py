

def TBT_bit_analysis(bit_string, block_len):
    '''
    :param bit_string:
    :param block_len:
    :return:
    '''

    block_num = len(bit_string) // block_len
    new_string = (bit_string[0:block_len*block_num]).reshape((block_num, block_len))


    print(new_string)



