from dataclasses import dataclass
from typing import List, Optional

import numba
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from cns_analytics.entities import Duration
from cns_analytics.timeseries import TimeSeries


@numba.njit()
def running_max(x):
    _max = x[0]
    y = np.empty_like(x)
    for i, val in enumerate(x):
        if val > _max:
            _max = val
        y[i] = _max
    return y


@numba.njit()
def running_min(x):
    _min = x[0]
    y = np.empty_like(x)
    for i, val in enumerate(x):
        if val < _min:
            _min = val
        y[i] = _min
    return y


@numba.njit
def _get_loss_points(arr: np.ndarray, width: float):
    exited_max = (running_max(arr) - arr) > width
    exited_min = (arr - running_min(arr)) > width
    exited_max[-1] = 1
    exited_min[-1] = 1
    arg_max = np.argmax(exited_max)
    arg_min = np.argmax(exited_min)
    return arg_max, arg_min


@dataclass
class TimeTillLossReport:
    timestamp: pd.Timestamp
    days_buy: float
    days_sell: float
    past_days_buy: float
    past_days_sell: float
    spread_value: float

    @property
    def days_both(self):
        return min(self.days_buy, self.days_sell)

    @property
    def past_days_both(self):
        return min(self.past_days_buy, self.past_days_sell)


def get_next_loss_time(df, width, reverse=False):
    if reverse:
        df = df.iloc[::-1]

    loss_idx_buy, loss_idx_sell = _get_loss_points(df.values, width)
    loss_idx_buy = df.index[loss_idx_buy]
    loss_idx_sell = df.index[loss_idx_sell]

    if reverse:
        # past has reversed buy and sell values
        days_buy = len(pd.date_range(loss_idx_sell, df.index[0], freq=BDay()))
        days_sell = len(pd.date_range(loss_idx_buy, df.index[0], freq=BDay()))
    else:
        days_buy = len(pd.date_range(df.index[0], loss_idx_buy, freq=BDay()))
        days_sell = len(pd.date_range(df.index[0], loss_idx_sell, freq=BDay()))

    return days_buy, days_sell


def get_time_till_loss(
        ts: TimeSeries,
        width: float,
        max_days: int = 500,
        framed=False,
        symbol='SPREAD',
        step: Duration = '7d',
        calc_past: bool = False,
        past_width: Optional[float] = None
) -> List[TimeTillLossReport]:
    results = []

    all_df = ts.get_raw_df()[symbol]

    max_ = pd.Timedelta(f'{max_days}d')
    past_window = pd.Timedelta('7d')

    past_width = past_width or width

    past_days_buy = 0
    past_days_sell = 0

    skip_start = '30d' if calc_past else '0m'

    for point in ts.get_datetime_iterator(step=step, framed=framed, skip_start=skip_start):
        if calc_past:
            past_df = all_df[point - max_: point]
            past_days_buy, past_days_sell = get_next_loss_time(past_df, past_width, reverse=True)

        df = all_df[point: point + max_]
        days_buy, days_sell = get_next_loss_time(df, width)

        results.append(TimeTillLossReport(
            timestamp=point,
            days_buy=days_buy,
            days_sell=days_sell,
            past_days_buy=past_days_buy,
            past_days_sell=past_days_sell,
            spread_value=df.iloc[0]))

    return results
