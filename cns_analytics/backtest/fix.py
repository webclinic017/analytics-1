from datetime import datetime

import matplotlib.pyplot as plt
import pytz
import pandas as pd
import numpy as np

from cns_analytics.entities import Duration
from cns_analytics.timeseries import TimeSeries


def get_fix(*, data, step, trend=None, width=None, sl_pos=None, buy=True, reverse=False, reval=True,
            book_spread=0, one_side_fee=0, early_exit=False, initial_pos=0,
            exit_on_sl=True, entry_mask=None):
    sell = not buy

    if isinstance(data, TimeSeries):
        data = data.get_raw_df()[data.expect_one_symbol()]

    if trend is None:
        assert width is None
        trend = data
        width = -1e9

    if entry_mask is None:
        entry_mask = data != np.nan

    data = data.to_frame('px')
    # print('step', step)

    data['trend'] = trend
    data['mask'] = entry_mask
    next_buy = None
    next_sell = None
    reverse = -1 if reverse else 1

    open_money = 0
    position = None
    fee = 0

    reval_history = []
    closes_history = []
    closes = 0

    if step < 2 * one_side_fee:
        raise Exception()

    data = data.reset_index()
    last_ts = None

    last_fix = 0
    last_day_fix = 0
    fix_per_day = {}

    for ts, px, _trend, _entry_mask in data.values:
        buy_px = px + book_spread * reverse
        sell_px = px - book_spread * reverse

        last_ts = ts

        if not position:
            if early_exit and closes > 0:
                break
            if _entry_mask:
                if buy:
                    next_buy = buy_px
                    next_sell = buy_px + step * 2
                    if position is None:
                        open_money -= initial_pos * buy_px
                        position = initial_pos
                else:
                    next_buy = sell_px - step * 2
                    next_sell = sell_px
                    if position is None:
                        open_money += initial_pos * sell_px
                        position = -initial_pos

        if next_buy is not None:
            if buy_px <= next_buy and ((buy_px < _trend - width and buy) or
                                       (sell and position * reverse < 0)):
                if position != 0 or sell or _entry_mask:
                    open_money -= buy_px * reverse
                    position += 1 * reverse
                    next_buy -= step
                    next_sell -= step
                    fee += one_side_fee
                    # print('buy', round(px, 3), position)
                    if sell:
                        closes += 1

            elif sell_px >= next_sell and ((sell_px > _trend + width and sell) or
                                           (buy and position * reverse > 0)):
                if position != 0 or buy or _entry_mask:
                    open_money += sell_px * reverse
                    position -= 1 * reverse

                    next_buy += step
                    next_sell += step
                    fee += one_side_fee
                    # print('sell', round(px, 3), position)
                    if buy:
                        closes += 1

        reval_history.append(open_money + (position or 0) * px - fee)
        closes_history.append(closes)

        if False:
            date = ts.date()
            if date not in fix_per_day:
                last_day_fix = last_fix

            fix_per_day[date] = closes - last_day_fix
            last_fix = closes

        if sl_pos is not None and abs(position or 0) >= sl_pos:
            # print('loss', 'buy' if buy else 'sell')
            if exit_on_sl:
                break
            else:
                open_money += (position or 0) * (buy_px if (position or 0) < 0 else sell_px)
                position = 0

    if isinstance(last_ts, datetime):
        last_ts = last_ts.astimezone(pytz.UTC)

    closes_history = np.asarray(closes_history)
    reval_history = np.asarray(reval_history)

    if not reval:
        return (closes_history * step - fee) * reverse, last_ts

    return reval_history, last_ts


def get_intersections(data, trend, width: float, reset_on_zero: bool = True):
    detrended = data - trend

    intersections_up = list(detrended[(detrended > width).astype(int).diff() == -1].index)
    intersections_down = list(detrended[(detrended < -width).astype(int).diff() == 1].index)
    intersections_zero = list(detrended[(detrended < 0).astype(int).diff() == 1].index)

    intersections = [(x, 'up') for x in intersections_up] + \
                    [(x, 'down') for x in intersections_down] + \
                    [(x, 'zero') for x in intersections_zero]

    intersections.sort(key=lambda x: x[0])

    state = 0
    count_up = 0
    count_down = 0

    for timestamp, side in intersections:
        if side == 'up' and (state == 0 or state == 1):
            state = -1
            count_up += 1
        elif side == 'down' and (state == 0 or state == -1):
            state = 1
            count_down += 1
        elif side == 'zero' and reset_on_zero:
            state = 0

    return count_up, count_down


def get_time_spaced_fix(ts: TimeSeries, time_step: Duration, loss_position: int, symbol=None,
                        buy=True):
    time_step = pd.Timedelta(time_step)

    df = ts.get_raw_df()

    if symbol is None:
        [symbol] = df.columns

    side = 1 if buy else -1

    df = df[symbol]

    last_px = df.iloc[0]

    money = -last_px * side
    position = 1 * side

    position_history = []
    reval_history = []
    px_history = []

    for point in ts.get_datetime_iterator(step=time_step):
        px = df.iloc[df.index.get_loc(point, method='nearest')]

        if abs(position) >= loss_position:
            money += position * px
            position = 0

        if (px < last_px and position < 0) or (position >= 0 and buy):
            money -= px
            position += 1
            last_px = px
        if (px > last_px and position > 0) or (position <= 0 and not buy):
            money += px
            position -= 1
            last_px = px
        last_px = px

        position_history.append(position)
        reval_history.append(money + position * px)
        px_history.append(px)

    return np.asarray(reval_history), np.asarray(position_history), np.asarray(px_history)
