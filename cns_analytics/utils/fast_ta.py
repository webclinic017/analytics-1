from typing import Union

import numpy as np
import numba


@numba.jit((numba.float64[:], numba.int64), nopython=True, nogil=True)
def _ewma(arr_in, window):
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=numba.float64)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1 - alpha) ** i
        ewma_old = ewma_old * (1 - alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    ewma[:window - 1] = np.nan
    return ewma


@numba.jit((numba.float64[:], numba.int64), nopython=True, nogil=True)
def _ewma_infinite_hist(arr_in, window):
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=numba.float64)
    alpha = 2 / float(window + 1)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = arr_in[i] * alpha + ewma[i - 1] * (1 - alpha)
    return ewma


@numba.njit
def _ema(arr, period, alpha: Union[float, bool]=False):
    if alpha is True:
        alpha = 1 / period
    elif alpha is False:
        alpha = 2 / (period + 1)

    exp_weights = np.zeros(len(arr))
    exp_weights[period - 1] = np.mean(arr[:period])
    for i in range(period, len(exp_weights) + 1):
        exp_weights[i] = exp_weights[i - 1] * (1 - alpha) + ((alpha) * (arr[i]))
    exp_weights[:period - 1] = np.nan
    return exp_weights


@numba.jit((numba.float64[:], numba.int64, numba.int64, numba.int64), nopython=True, nogil=True)
def macd(s, window_slow, window_fast, window_sign):
    _emafast = _ewma_infinite_hist(s, window_fast)
    _emaslow = _ewma_infinite_hist(s, window_slow)
    _macd = _emafast - _emaslow
    _macd_signal = _ewma_infinite_hist(_macd[window_slow - 1:], window_sign)
    nans = np.zeros((window_slow - 1,))
    nans[:] = np.nan
    _macd_signal = np.concatenate((nans, _macd_signal))

    return _macd, _macd_signal


def _mad(x):
    return np.mean(np.abs(x - np.mean(x)))


@numba.jit(nopython=True)
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


@numba.njit
def sma(arr, period):
    results = np.zeros(len(arr))
    for i in range(period - 1, len(results)):
        window = arr[i - period + 1: i + 1]
        results[i] = np.sum(window)
    results[:period - 1] = np.nan
    return results / period


# @numba.njit
def cci(df_high, df_low, df_close, period=20, scaling=0.015):
    typicalPrice = (df_high + df_close + df_low) / 3
    rolling_windows = rolling_window(typicalPrice, period)
    central_tendency_arr = sma(typicalPrice, period)[period - 1:]
    abs_deviation_arr = np.abs((rolling_windows.T - central_tendency_arr).T)
    mean_abs_deviation = np.zeros(len(abs_deviation_arr))

    # once numba has a way of reducing along axes, can switch this away
    for i in range(len(rolling_windows)):
        mean_abs_deviation[i] = np.mean(abs_deviation_arr[i])

    result = (typicalPrice[period - 1:] - central_tendency_arr) / (mean_abs_deviation * scaling)
    result = np.concatenate((np.array([np.nan] * (period - 1)), result))
    return result


@numba.njit
def rsi(df_close, period=14):
    delta = np.diff(df_close)
    up, down = np.copy(delta), np.copy(delta)
    up[up < 0] = 0
    down[down > 0] = 0

    # Exponential Weighted windows mean with centre of mass = period - 1 -> alpha = 1 / (period)
    alpha = 1 / period
    rUp = _ema(up, period, alpha=alpha)
    rDown = np.abs(_ema(down, period, alpha=alpha))
    result = 100 - (100 / (1 + rUp / rDown))

    result = np.concatenate((np.array([np.nan]), result))
    return result
