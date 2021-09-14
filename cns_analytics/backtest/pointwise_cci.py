import asyncio
import itertools
import random

import numba as nb
import numpy as np
import matplotlib.pyplot as plt
import ta.trend
from matplotlib.widgets import MultiCursor

from cns_analytics.database import DataBase
from cns_analytics.entities import Exchange
from cns_analytics.timeseries import TimeSeries, Spread
from cns_analytics.utils import fast_ta


# @nb.jit((nb.float64[:], nb.float64[:], nb.int64), nopython=True, nogil=True)
def backtest(px1, px2, hours):
    ta.trend.cci()
    
    macd_diff = []
    spread_hist = []

    last_macd_diff_change = None

    buys_opn = []
    buys_cls = []
    sells_opn = []
    sells_cls = []
    buys_px = []
    sells_px = []
    reval_hist = []

    sensitivity = 0.015
    max_change = 0.125

    # TODO: MACD don't open when bar change is too high (opposite sensitivity)
    # TODO: react on steep change (sensitivity)

    open_new_pos = False
    coef = None
    mean = None

    peaks_buy = []
    peaks_sell = []

    pos = 0
    pos_hist = []

    current_pos = None

    for i in range(26 * hours + hours, len(px1), hours):
        data1 = px1[:i+1]
        data2 = px2[:i+1]

        if current_pos is None or coef is None:
            coef = data1.mean() / data2.mean()

        spread_data = (data1 - data2 * coef)[-(26 * hours - hours) * 10:]
        spread_px = spread_data[-1]
        mean = (data1 - data2 * coef).mean()

        spread_hist.append(spread_data[-1])

        mlen = min(len(buys_px), len(sells_px))
        reval_hist.append(sum(sells_px[:mlen]) - sum(buys_px[:mlen]))

        macd_last_6 = fast_ta.macd(spread_data,
                                   window_slow=26*hours, window_fast=12*hours, window_sign=9*hours)
        diff = (macd_last_6[0] - macd_last_6[1])[-hours:].mean()

        min_peak = 0

        if macd_diff:
            macd_diff_change = diff - macd_diff[-1] > 0
            # if np.abs(diff - macd_diff[-1]) < sensitivity:
            #     macd_diff_change = last_macd_diff_change
            #
            # if np.abs(diff - macd_diff[-1]) > max_change:
            #     last_macd_diff_change = macd_diff_change

            if last_macd_diff_change is not None and last_macd_diff_change != macd_diff_change:
                if macd_diff_change:
                    # buy
                    if spread_px < mean and diff < -min_peak:
                        # don't move under open_new_pos check
                        # that condition will change
                        peaks_buy.append(round(diff, 3))
                        all_peaks = peaks_sell + [-x for x in peaks_buy]
                        if len(all_peaks) > 10:
                            min_peak = np.percentile(all_peaks[-10:], 70) * 0.4
                        elif len(all_peaks) == 10:
                            min_peak = np.percentile(all_peaks, 70)
                            peaks_buy = [x if x < -min_peak else -min_peak for x in peaks_buy]
                            peaks_sell = [x if x > min_peak else min_peak for x in peaks_sell]
                        else:
                            min_peak = np.inf

                        # print(round(min_peak, 2), peaks_buy, peaks_sell)
                    # open new pos?
                    open_new_pos = spread_px < mean and diff < 0#-min_peak

                    if sells_px:#current_pos == 'sell':
                        # close old pos
                        buys_cls.append(len(macd_diff))
                        buys_px.append(spread_px)
                        pos += 1

                    current_pos = None

                    if open_new_pos:
                        # open new pos
                        current_pos = 'buy'
                        buys_opn.append(len(macd_diff))
                        buys_px.append(spread_px)
                        pos += 1
                else:
                    # sell
                    if spread_px > mean and diff > min_peak:
                        # don't move under open_new_pos check
                        # that condition will change
                        peaks_sell.append(round(diff, 3))
                        all_peaks = peaks_sell + [-x for x in peaks_buy]
                        if len(all_peaks) > 10:
                            min_peak = np.percentile(all_peaks[-10:], 70) * 0.4
                        elif len(all_peaks) == 10:
                            min_peak = np.percentile(all_peaks, 70)
                            peaks_buy = [x if x < -min_peak else -min_peak for x in peaks_buy]
                            peaks_sell = [x if x > min_peak else min_peak for x in peaks_sell]
                        else:
                            min_peak = np.inf
                        # print(round(min_peak, 2), peaks_buy, peaks_sell)
                    # open new pos?
                    open_new_pos = spread_px > mean and diff > 0#min_peak

                    if buys_px:#current_pos == 'buy':
                        # close old pos
                        sells_cls.append(len(macd_diff))
                        sells_px.append(spread_px)
                        pos -= 1

                    current_pos = None

                    if open_new_pos:
                        # open new pos
                        current_pos = 'sell'
                        sells_opn.append(len(macd_diff))
                        sells_px.append(spread_px)
                        pos -= 1

                # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
                # ax1.plot(spread_hist)
                # ax2.bar(list(range(len(macd_diff))), macd_diff, alpha=0.7)
                # cur = MultiCursor(fig.canvas, (ax1, ax2))
                # # plt.bar(list(range(len(exp_macd[27:]))), exp_macd[27:], alpha=0.3)
                # plt.show()
            last_macd_diff_change = macd_diff_change

        macd_diff.append(diff)
        pos_hist.append(pos)

    breakpoint()

    if len(sells_px) != len(buys_px):
        mlen = min(len(sells_px), len(buys_px))
        sells_px = sells_px[:mlen]
        buys_px = buys_px[:mlen]
    #
    # breakpoint()
    #
    # fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    # plt.tight_layout()
    # ax1.plot(spread_hist)
    # ax2.bar(list(range(len(macd_diff))), macd_diff, alpha=0.7)
    # ax2.vlines(buys_opn, min(macd_diff), mean, colors=['green']*len(buys_opn))
    # ax2.vlines(buys_cls, mean, max(macd_diff), colors=['green']*len(buys_cls))
    # ax2.vlines(sells_opn, mean, max(macd_diff), colors=['red']*len(sells_opn))
    # ax2.vlines(sells_cls, min(macd_diff), mean, colors=['red']*len(sells_cls))
    #
    # cur = MultiCursor(fig.canvas, (ax1, ax2))
    # plt.show()
    # plt.close(fig)

    return buys_px, sells_px, spread_hist, reval_hist


    # plt.bar(list(range(len(exp_macd[27:]))), exp_macd[27:], alpha=0.3)


async def main():
    good_symbols = ['MATICUSDT', 'MKRUSDT', 'ALGOUSDT', 'ALPHAUSDT', 'ATOMUSDT', 'AVAXUSDT',
                    'AXSUSDT', 'BANDUSDT', 'BATUSDT', 'BELUSDT', 'SUSHIUSDT', 'BLZUSDT',
                    'BZRXUSDT', 'AAVEUSDT', 'ZRXUSDT', 'BALUSDT', 'UNIUSDT', 'CTKUSDT',
                    'CVCUSDT', 'DOTUSDT', 'EGLDUSDT', 'ENJUSDT', 'EOSUSDT', 'ETCUSDT',
                    'CRVUSDT', 'FILUSDT', 'FTMUSDT', 'FLMUSDT', 'GRTUSDT', 'HNTUSDT',
                    'ICXUSDT', 'IOSTUSDT', 'IOTAUSDT', 'KAVAUSDT', 'KNCUSDT', 'LINKUSDT',
                    'LRCUSDT', 'OCEANUSDT', 'ONTUSDT', 'QTUMUSDT', 'RENUSDT', 'RLCUSDT',
                    'RSRUSDT', 'KSMUSDT', 'SKLUSDT', 'SNXUSDT', 'SOLUSDT', 'SRMUSDT', 'STORJUSDT',
                    'SXPUSDT', 'THETAUSDT', 'TOMOUSDT', 'TRBUSDT', 'TRXUSDT', 'VETUSDT',
                    'WAVESUSDT', 'XLMUSDT', 'XMRUSDT', 'XTZUSDT', 'YFIIUSDT', 'YFIUSDT', 'ZECUSDT',
                    'ZENUSDT', 'ZILUSDT', '1INCHUSDT']
    pairs = list(itertools.combinations(good_symbols, 2))
    random.shuffle(pairs)

    for s1, s2 in pairs:
        print(s1, s2)
        # s1, s2 = 'UNIUSDT', '1INCHUSDT'
        spread = TimeSeries(s1, s2)
        await spread.load(start='2021-01-01', resolution='1m')
        data = spread.get_raw_df().values

        backtest(data[:, 0], data[:, 1], hours)


if __name__ == '__main__':
    asyncio.run(main())
