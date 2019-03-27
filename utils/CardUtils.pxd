# distutils: language=c++
cimport numpy as np

ctypedef np.int8_t card_t
ctypedef card_t[:] deck_t

cpdef np.float64_t[:] get_features(card_t[:] deal)