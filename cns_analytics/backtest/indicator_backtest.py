import abc

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

from cns_analytics.backtest.indicators.macd import MACDDiffIndicator
from cns_analytics.backtest.indicators.rsi import RSIIndicator
from cns_analytics.database import DataBase
from cns_analytics.entities import Exchange, Symbol
from cns_analytics.timeseries import TimeSeries
from cns_analytics.utils import fast_ta, get_symbol_pairs, timeit


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
        self.time_mult = 60

        self.positions['1'] = 0
        self.positions['2'] = 0

        self._loop_i = 0
        self._pre_open_reval = 0

        self.bought_at = []
        self.sold_at = []
        self._finrez_results = []
        self._waiting_after_loss = None
        self._tp_counter = 0
        self._sl_counter = 0

        self.rsi = RSIIndicator()
        self.macd_diff = MACDDiffIndicator()

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

            hours = 4
            self.rsi.calculate(df_close=spread, period=14 * hours)
            self.macd_diff.calculate(df_close=spread,
                                     period_slow=26 * hours,
                                     period_fast=12 * hours,
                                     period_signal=9 * hours)

    def should_close(self):
        sl = False

        sell = False
        buy = False

        if self.positions['1'] < 0 and buy:
            self.buy('1', 1)
            self.sell('2', self._coef)
            self.sold_at[-1].append(self._loop_i)

            current_reval = self.get_revaluation()
            fr = current_reval - self._pre_open_reval

            self.sold_at[-1].append(sl)
            self._finrez_results.append(fr)

            if fr >= 0:
                self._tp_counter += 1
            else:
                self._sl_counter += 1

            if sl:
                self._waiting_after_loss = 'sell'

        if self.positions['1'] > 0 and sell:
            self.sell('1', 1)
            self.buy('2', self._coef)
            self.bought_at[-1].append(self._loop_i)

            current_reval = self.get_revaluation()
            fr = current_reval - self._pre_open_reval

            self.bought_at[-1].append(sl)
            self._finrez_results.append(fr)

            if fr >= 0:
                self._tp_counter += 1
            else:
                self._sl_counter += 1

            if sl:
                self._waiting_after_loss = 'buy'

    def should_open(self):
        breakpoint()
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

    def run(self, data):
        all_pxs1, all_pxs2 = data[:, 0], data[:, 1]

        len_all = len(all_pxs1)

        start_idx = int(len_all * 0.3)

        if start_idx >= len_all:
            return

        for i in range(start_idx, len_all):
            self._loop_i = i

            self.calc_coef(all_pxs1[:i+1], all_pxs2[:i+1])

            if self.positions['1'] != 0:
                self.should_close()
            elif self.positions['1'] == 0:
                self.should_open()

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

    s1, s2 = Symbol('BZ=F', exchange=Exchange.YFinance), Symbol('HG=F', exchange=Exchange.YFinance)
    ts = TimeSeries(s1, s2)
    await ts.load(resolution='1h')

    data = ts.get_raw_df().values

    mb = MetaBack()
    mb.run(data)


if __name__ == '__main__':
    asyncio.run(main())

