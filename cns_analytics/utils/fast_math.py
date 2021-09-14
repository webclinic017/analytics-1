import numba
import numpy as np


@numba.njit((numba.float64[:], ), nogil=True)
def expanding_mean(arr):
    """Returns mean for all elements past current (including current) for every point
    """
    total_len = arr.shape[0]
    return ((arr / total_len).cumsum() / np.arange(1, total_len + 1)) * total_len
