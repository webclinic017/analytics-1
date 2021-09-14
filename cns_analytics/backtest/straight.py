import pandas as pd
import contextlib
import time

import numpy as np
import asyncio
import numba as nb
import matplotlib.pyplot as plt
import pytz
from matplotlib.widgets import MultiCursor
from numba.experimental import jitclass

from cns_analytics.database import DataBase
from cns_analytics.entities import Exchange, Symbol
from cns_analytics.timeseries import TimeSeries
from cns_analytics.utils import fast_ta, get_symbol_pairs, timeit

SHOULD_PRINT = False


class RSISignal:
    nan_value = 50

    def __init__(self, low_percentile=20, high_percentile=80):
        # last value of rsi
        self.last = None
        # rsi values for latest df_close
        self.data = np.empty(1)
        # history of last values for all past df_close
        self.history = []
        self.history_low_limit = []
        self.history_high_limit = []
        self.low_limit = 0
        self.high_limit = 0

        self._low_percentile = low_percentile
        self._high_percentile = high_percentile

    def calculate(self, df_close, period):
        rsi = fast_ta.rsi(df_close=df_close, period=period)

        self.data = rsi
        self.last = rsi[-1]

        if not self.history:
            init_rsi = rsi.copy()
            init_rsi[np.isnan(init_rsi)] = self.nan_value
            self.history = init_rsi.tolist()
            init_rsi[:] = self.nan_value
            self.history_low_limit = init_rsi.tolist()
            self.history_high_limit = init_rsi.tolist()
        else:
            self.history.append(rsi[-1])

        low = 50 - np.nanpercentile(np.abs(rsi), self._low_percentile)
        high = np.nanpercentile(np.abs(rsi), self._high_percentile) - 50

        best = max(low, high)

        self.low_limit = 50 - best
        self.high_limit = 50 + best

        self.history_low_limit.append(self.low_limit)
        self.history_high_limit.append(self.high_limit)

    def should_buy(self):
        if self.last is None:
            return None
        return self.last <= self.low_limit

    def should_sell(self):
        if self.last is None:
            return None
        return self.last >= self.high_limit


# @jitclass([
#     ('positions', nb.types.DictType(nb.types.string, nb.float64)),
#     ('prices', nb.types.DictType(nb.types.string, nb.float64)),
#     ('open_money', nb.float64),
#     ('iteration', nb.float64),
#     ('time_mult', nb.int8),
#
#     ('_coef', nb.float64),
#     ('_pos_coef', nb.float64),
#     ('_macd_last', nb.float64),
#     ('_spread_last', nb.float64),
#     ('_cci_last', nb.float64),
# ])
class MetaBack:
    def __init__(self):
        self.positions = nb.typed.Dict.empty(nb.types.string, nb.float64)
        self.prices = nb.typed.Dict.empty(nb.types.string, nb.float64)
        self.open_money = 0
        self.iteration = 0
        self.time_mult = 5

        self._coef = 0
        self._pos_coef = 0
        self._macd_diff_last = 0
        self._macd_diff_high = 0
        self._macd_diff_low = 0
        self._macd_fast_last = 0
        self._macd_slow_last = 0
        self._spread_last = 0
        self._cci_last = 0

        self.positions['1'] = 0
        self.positions['2'] = 0

        self._loop_i = 0
        self._pre_open_reval = 0
        self._sl = 0

        self.bought_at = []
        self.sold_at = []
        self._cci = []
        self._macd = []
        self._ccis = []
        self._macds = []
        self._spreads = []
        self._finrez_results = []
        self._spread_mean = 0
        self._cci_high_limit = 0
        self._cci_low_limit = 0
        self._waiting_after_loss = None
        self._tp_counter = 0
        self._sl_counter = 0

        self.rsi = RSISignal()

    def buy(self, sec: str, qty: float):
        # if SHOULD_PRINT:
        #     print('buy', sec, self.prices[sec])
        if sec in self.positions:
            self.positions[sec] += qty
        else:
            self.positions[sec] = qty
        self.open_money -= self.prices[sec] * qty

    def sell(self, sec: str, qty: float):
        # if SHOULD_PRINT:
        #     print('sell', sec, self.prices[sec])
        if sec in self.positions:
            self.positions[sec] -= qty
        else:
            self.positions[sec] = -qty
        self.open_money += self.prices[sec] * qty

    def get_revaluation(self):
        revaluation = self.open_money
        for sec in self.positions:
            revaluation += self.positions[sec] * self.prices[sec]
        return revaluation

    def set_prices(self, sec, px):
        self.prices[sec] = px

    def calc_coef(self, known_prices1, known_prices2):
        self.set_prices('1', known_prices1[-1])
        self.set_prices('2', known_prices2[-1])

        if self.positions['1'] == 0 and self.iteration % (1440 // self.time_mult) == 0:
            self._coef = known_prices1.mean() / known_prices2.mean()

        if self.iteration % (60 // self.time_mult) == 0:
            spread = known_prices1 - known_prices2 * self._coef

            # if self._ccis:
            #     spread_candles = spread[-8016*2:].reshape(-1, (60 // self.time_mult))
            # else:
            spread_candles = spread[len(spread) % 12:].reshape(-1, (60 // self.time_mult))
            spread_high = spread_candles.max(axis=1)
            spread_low = spread_candles.min(axis=1)
            spread_close = np.ascontiguousarray(spread_candles[:, -1])
            hours = 24
            cci = fast_ta.cci(df_high=spread_high, df_low=spread_low, df_close=spread_close, period=20 * hours)

            self.rsi.calculate(df_close=spread_close, period=14 * hours)

            macd = fast_ta.macd(spread_close, window_slow=26*hours, window_fast=12*hours, window_sign=9*hours)

            self._macd_fast_last = macd[0][-1]
            self._macd_slow_last = macd[1][-1]

            macd = macd[0] - macd[1]
            self._macd = macd

            # if self.iteration >= 20000:
            #     spread_candles = spread[-11520:].reshape(-1, 12)
            #     spread_high = spread_candles.max(axis=1)
            #     spread_low = spread_candles.min(axis=1)
            #     spread_close = spread_candles[:, -1]
            #     hours = 24
            #     cci2 = fast_ta.cci(df_high=spread_high, df_low=spread_low, df_close=spread_close,
            #                       period=20 * hours)
            #     breakpoint()

            self._cci = cci

            self._cci_last = cci[-1]
            if self._ccis:
                self._ccis.append(cci[-1])
                self._macds.append(macd[-1])
                self._spreads.append(spread_close[-1])
            else:
                cci[np.isnan(cci)] = 0
                macd[np.isnan(macd)] = 0
                self._ccis = cci.tolist()
                self._spreads = spread_close.tolist()
                self._macds = macd.tolist()

            self._spread_last = spread[-1]
            self._macd_diff_last = macd[-1]
            self._macd_diff_high = np.nanpercentile(np.abs(self._macd), 40)
            self._macd_diff_low = -self._macd_diff_high
            # self._spread_mean = np.mean(spread_close)
            self._sl = np.nanpercentile(self._spreads, 70) - np.nanpercentile(self._spreads, 30)

        if self._ccis and self.iteration % (5000 // self.time_mult) == 0:
            self._cci_high_limit = np.nanpercentile(np.abs(self._cci), 60)
            self._cci_low_limit = -self._cci_high_limit

    def should_close(self):
        sl = False

        current_reval = self.get_revaluation()
        fr = current_reval - self._pre_open_reval

        # if fr < -self._sl:
        #     sl = True

        macd_diff_says_sell = False#(self._macds[-8] < self._macds[-3] < self._macds[-2] > self._macds[
           #-1])# and self._macd_slow_last > 0 and self._macd_fast_last > 0
        macd_diff_says_buy = False#(self._macds[-8] > self._macds[-3] > self._macds[-2] < self._macds[
           #-1])# and self._macd_slow_last < 0 and self._macd_fast_last < 0

        if self.positions['1'] < 0 and (self._macd_diff_last <= 0 or sl or macd_diff_says_buy):
            if SHOULD_PRINT:
                print('close sell', self._spread_last)
            self.buy('1', 1)
            self.sell('2', self._coef)
            self.sold_at[-1].append(self._loop_i)
            self.sold_at[-1].append(sl)
            self._finrez_results.append(fr)

            if fr >= 0:
                self._tp_counter += 1
            else:
                self._sl_counter += 1

            if sl:
                self._waiting_after_loss = 'sell'

        if self.positions['1'] > 0 and (self._macd_diff_last >= 0 or sl or macd_diff_says_sell):
            if SHOULD_PRINT:
                print('close buy', self._spread_last)
            self.sell('1', 1)
            self.buy('2', self._coef)
            self.bought_at[-1].append(self._loop_i)
            self.bought_at[-1].append(sl)
            self._finrez_results.append(fr)

            if fr >= 0:
                self._tp_counter += 1
            else:
                self._sl_counter += 1

            if sl:
                self._waiting_after_loss = 'buy'

    def should_open(self):
        spread_lower_than_mean_spread = True#self._spread_last < self._spread_mean
        spread_higher_than_mean_spread = True#self._spread_last > self._spread_mean

        macd_diff_says_sell = (self._macd[-8] < self._macd[-3] < self._macd[-2] > self._macd[-1])# and self._macd_slow_last > 0 and self._macd_fast_last > 0
        macd_diff_says_buy = (self._macd[-8] > self._macd[-3] > self._macd[-2] < self._macd[-1])# and self._macd_slow_last < 0 and self._macd_fast_last < 0

        macd_diff_says_sell = macd_diff_says_sell and self._macd_diff_last > self._macd_diff_high
        macd_diff_says_buy = macd_diff_says_buy and self._macd_diff_last < self._macd_diff_low

        cci_says_buy = self._cci_last <= self._cci_low_limit
        cci_says_sell = self._cci_last >= self._cci_high_limit

        oscilator_says_buy = self.rsi.should_buy()
        oscilator_says_sell = self.rsi.should_sell()

        # l = len(self.rsi.history)
        #
        # if l >= 3075 and l % 25 == 0:
        #     breakpoint()

        if oscilator_says_buy and spread_lower_than_mean_spread and macd_diff_says_buy:
            self._pre_open_reval = self.get_revaluation()
            if SHOULD_PRINT:
                print('open buy', self._spread_last)
            self.buy('1', 1)
            self.sell('2', self._coef)
            self._pos_coef = self._coef

            self.bought_at.append([self._loop_i])

        if oscilator_says_sell and spread_higher_than_mean_spread and macd_diff_says_sell:
            self._pre_open_reval = self.get_revaluation()
            if SHOULD_PRINT:
                print('open sell', self._spread_last)
            self.sell('1', 1)
            self.buy('2', self._coef)
            self._pos_coef = self._coef

            self.sold_at.append([self._loop_i])

    def check_everything(self):
        assert round(self.positions['1'] * self._pos_coef, 6) == \
               -round(self.positions['2'], 6)

    def run(self, data):
        all_pxs1, all_pxs2 = data[:, 0], data[:, 1]

        # 12 points = 1 hour; 12 * 24 = 1 day
        start_idx = round(60 / self.time_mult * 24 * 40)

        len_all = len(all_pxs1)

        if start_idx >= len_all:
            return

        for i in range(start_idx, len_all):
            self._loop_i = i

            self.calc_coef(all_pxs1[:i+1], all_pxs2[:i+1])

            # with timeit():
            #     self.check_everything()

            if self.positions['1'] != 0:
                self.should_close()
            elif self.positions['1'] == 0:# and self._waiting_after_loss is None:
                self.should_open()

            if self._waiting_after_loss == 'buy' and self._cci_last > 0:
                self._waiting_after_loss = None

            if self._waiting_after_loss == 'sell' and self._cci_last < 0:
                self._waiting_after_loss = None

            if i % 1000 == 0:
                print(f'    [{self._tp_counter}|{self._sl_counter}] Finrez({i/len_all*100:0.2f}%) =', round(self.get_revaluation(), 3), end='   \r')

            self.iteration += 1

        print(f'    [{self._tp_counter}|{self._sl_counter}] Finrez(100%) =', round(self.get_revaluation(), 3), '      ')

        print(f'tp: {len([x for x in self._finrez_results if x >= 0])}, '
              f'sl: {len([x for x in self._finrez_results if x < 0])}')
        print([round(x, 3) for x in self._finrez_results])
        print()

        fig, axes = plt.subplots(nrows=3)

        axes[0].plot(self._spreads)

        for opn, cls, sl in [x for x in self.bought_at if len(x) == 3]:
            axes[0].axvspan(opn // 12, cls // 12, alpha=0.5, color='green' if sl else 'green')
        for opn, cls, sl in [x for x in self.sold_at if len(x) == 3]:
            axes[0].axvspan(opn // 12, cls // 12, alpha=0.5, color='red' if sl else 'red')

        axes[1].plot(self.rsi.history)
        axes[1].plot(self.rsi.history_low_limit, color='red')
        axes[1].plot(self.rsi.history_high_limit, color='red')
        axes[1].axhline(50)

        axes[2].bar(list(range(len(self._macds))), self._macds)
        axes[2].axhline(0)

        cur = MultiCursor(fig.canvas, axes)

        plt.show()
        breakpoint()
        pass


async def main():
    # warm up cache
    fast_ta.cci(np.random.random(100), np.random.random(100), np.random.random(100))

    symbols = await DataBase.get_all_symbols(exchange=Exchange.BinanceFutures)

    for s1, s2 in get_symbol_pairs(symbols):
        s1, s2 = Symbol('UNIUSDT'), Symbol('1INCHUSDT')
        ts = TimeSeries(s1, s2)
        await ts.load(start='2021-01-01', resolution='5m')

        try:
            if ts.get_raw_df().index[0] > pd.Timestamp('2021-01-15').tz_localize(tz=pytz.UTC):
                continue
        except IndexError:
            print()
            continue

        print(s1.name,  s2.name)

        data = ts.get_raw_df().values

        mb = MetaBack()
        mb.run(data)


    # for i in range(5):
    #     t1 = time.perf_counter()
    #     mb = MetaBack()
    #     mb.run(data)
    #     print(time.perf_counter() - t1)


if __name__ == '__main__':
    asyncio.run(main())

