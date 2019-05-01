# distutils: language=c++
cimport numpy as np
cimport cython

# card = np.int8
ctypedef np.int8_t card_t
ctypedef card_t[:] deck_t

ctypedef np.float32_t result_t
ctypedef result_t[:,:] timeseries_t

cpdef result_t[:] get_features(card_t[:] deal)