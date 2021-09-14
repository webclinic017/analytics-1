import contextlib
import itertools
from typing import Optional

import math
import random
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cns_analytics.entities import DropLogic, Duration


def get_ols_regression(x, y):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values

    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.values

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    ones = np.ones(shape=y.shape[0]).reshape(-1, 1)

    y = np.concatenate((ones, y), 1)

    try:
        coefs = np.linalg.inv(y.transpose().dot(y)).dot(y.transpose()).dot(x)
    except np.linalg.linalg.LinAlgError:
        raise

    return coefs


def get_trend(x, out_size=None):
    lx = len(x)
    out_size = out_size or lx
    line = np.linspace(0, 1 / lx * out_size, out_size)
    coef = get_ols_regression(x, line[:lx])

    return line * coef[1] + coef[0]


def detrend(x):
    line = np.linspace(0, 1, len(x))
    coef = get_ols_regression(x, line)

    return x - line * coef[1]


def get_symbol_pairs(symbols, shuffle=True):
    if shuffle:
        # changes order of symbols in spreads
        # for example, without this only UNI-INCH could be possible if UNI is first in symbols list
        # but shuffling allows to bypass initial order of symbols
        random.shuffle(symbols)

    result = list(itertools.combinations(symbols, 2))

    if shuffle:
        random.shuffle(result)

    return result


_TIMEIT_SUM = defaultdict(float)
_TIMEIT_START = {}
_TIMEIT_HISTORY = defaultdict(list)


@contextlib.contextmanager
def timeit(name='timeit', on=True, pct=True, reset=False, mean=False):
    if not on:
        yield
        return

    if name not in _TIMEIT_START:
        _TIMEIT_START[name] = time.time()

    t1 = time.perf_counter()
    yield
    t2 = time.perf_counter()

    _TIMEIT_SUM[name] += t2 - t1

    time_total = time.time() - _TIMEIT_START[name]

    _TIMEIT_HISTORY[name].append(_TIMEIT_SUM[name])

    if mean:
        time_total = np.mean(_TIMEIT_HISTORY[name])
    elif not pct:
        time_total = _TIMEIT_START[name]

    if pct:
        print(name, f"{round(_TIMEIT_SUM[name] / time_total*100, 2)}%", end='         \r')
    else:
        print(name, f"{round(time_total * 1e3, 3)}ms", end='         \r')

    if reset:
        _TIMEIT_SUM[name] = 0
        del _TIMEIT_START[name]


def get_correlation(x, y):
    return spearmanr(x, y).correlation


def get_line_angle(x1, y1):
    """Returns angle for line that crosses (0, 0) and (x1, y1)"""
    dx = 0 - x1
    dy = 0 - y1
    radians = math.atan2(dy, dx)
    angle = math.degrees(radians)

    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180

    return angle


def get_drop(data: pd.Series, logic: DropLogic,
             window: Optional[Duration] = None,
             growth_by: Optional[float] = None,
             growth_during: Optional[Duration] = None,
             std_period: Duration = '30d',
             std_limit: Optional[float] = None) -> pd.Series:
    if logic is DropLogic.SIMPLE:
        return data.cummax() - data
    elif logic is DropLogic.SKIP_AFTER_UPDATE:
        if window is None:
            raise Exception(f"window prameter is required for {logic}")

        if growth_during is not None:
            growth_during = pd.Timedelta(growth_during)

        assert bool(growth_by) + bool(growth_during) in {0, 2}

        window = pd.Timedelta(window)

        result = []
        result_time = []
        skip_until = None

        data = data.to_frame(name='px')

        if growth_during and growth_by:
            data['look_back'] = data.shift(freq=growth_during).ffill()
        else:
            data['look_back'] = data

        data_freq = data.index.to_series().diff().mode().iloc[0]
        new_index = pd.date_range(data.index[0], data.index[-1],
                                  freq=data_freq)
        data = data.reindex(new_index, fill_value=np.nan).ffill()

        std_period = pd.Timedelta(std_period)
        data['std'] = data.px.rolling(std_period, min_periods=int(std_period / data_freq)).std()

        last_high = None

        for _time, px, look_back, std in data.reset_index().values:
            result_time.append(_time)

            is_working = not skip_until or _time > skip_until

            if is_working:
                if not result or np.isnan(result[-1]):
                    result.append(px)
                    continue

                if px > result[-1]:
                    if not growth_by or px > look_back + growth_by:
                        skip_until = _time + window
                        last_high = result[-1]
                        result.append(np.nan)
                        continue

                result.append(max(result[-1], px))
                pass
            else:
                if px < last_high and (std_limit is None or std < std_limit):
                    skip_until = None
                result.append(np.nan)

        return (pd.Series(result, index=pd.DatetimeIndex(result_time)))# - data['px']).clip(0)
    elif logic is DropLogic.WINDOWED:
        return data.rolling(window=pd.Timedelta(window)).max() - data
    else:
        raise NotImplementedError()
