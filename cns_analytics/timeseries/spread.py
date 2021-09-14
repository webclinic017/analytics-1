""" Simplifies work with spreads
"""
import functools
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ta
from matplotlib.widgets import MultiCursor

from cns_analytics.entities import Symbol
from cns_analytics.timeseries.timeseries import TimeSeries


def _wrap_spread_method(method):
    """Parametrized decorator for spread methods that work with legs of spread,
    rather that spread itself"""
    @functools.wraps(method)
    def _decorator(*args, **kwargs):
        self = method.__self__

        self.include_symbol(self._leg1)
        self.include_symbol(self._leg2)
        self.exclude_symbol(self.SPREAD_SERIES_NAME)

        ret_val = method(*args, **kwargs)

        self.exclude_symbol(self._leg1)
        self.exclude_symbol(self._leg2)
        self.include_symbol(self.SPREAD_SERIES_NAME)

        self[self.SPREAD_SERIES_NAME] = self[self._leg1] - self[self._leg2]

        return ret_val

    return _decorator


class Spread(TimeSeries):
    """Represents time series of a spread

    | Simplifies function calls to spread time series
    | Adds new method for spread analysing
    """

    SPREAD_SERIES_NAME = 'SPREAD'

    def __init__(self, leg1, leg2, op='-'):
        if isinstance(leg1, str):
            leg1 = Symbol(leg1)
        if isinstance(leg2, str):
            leg2 = Symbol(leg2)

        super().__init__(leg1, leg2)
        self._leg1 = leg1.name
        self._leg2 = leg2.name
        self._op = op

        self.scale_ols = _wrap_spread_method(self.scale_ols)
        self.scale_mean = _wrap_spread_method(self.scale_mean)
        self.get_correlation = _wrap_spread_method(self.get_correlation)
        self.get_sensitivity = _wrap_spread_method(self.get_sensitivity)
        self._coefs = {}

    async def load(self, start: Optional[str] = None, end: Optional[str] = None, resolution='1h'):
        res = await super().load(start=start, end=end, resolution=resolution)
        self.exclude_symbol(self._leg1)
        self.exclude_symbol(self._leg2)
        self._coefs = self.scale_mean()
        self.dropna()
        if self._op == '-':
            self[self.SPREAD_SERIES_NAME] = self[self._leg1] - self[self._leg2]
        elif self._op == '/':
            self[self.SPREAD_SERIES_NAME] = self[self._leg1] - self[self._leg2]

        return res

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> 'Spread':
        if isinstance(df, pd.Series):
            df = df.to_frame()

        spread = cls(df.columns[0], df.columns[1])

        spread.scale_mean()
        spread.exclude_symbol(spread._leg1)
        spread.exclude_symbol(spread._leg2)
        spread[cls.SPREAD_SERIES_NAME] = spread[spread._leg1] - spread[spread._leg2]

        spread._df = df.copy()
        spread._df.sort_index()

        return spread

    def plot_legs(self):
        """Plots legs of spread"""
        fig = plt.figure(figsize=(15, 6))
        plt.title(self.get_name())
        plt.plot(self.get_raw_df()[self._leg1])
        plt.plot(self.get_raw_df()[self._leg2])

        if self._frame_start or self._frame_end:
            start = self._frame_start or self._df.index[0]
            end = self._frame_end or self._df.index[-1]
            plt.axvspan(start, end, color='blue', alpha=0.1)

        plt.show()
        plt.close(fig)

    def get_name(self):
        """Returns name of spread

        :returns: Returns name in format Leg1 - Leg2
        """
        return f"{self._leg1} {self._op} {self._leg2}"

    def get_magnitude(self, framed: bool = True, percentile: float = 0.025):
        """Returns magnitude of a spread as high percentile - low percentile

        :returns: Magnitude
        """
        if framed:
            df = self.get_framed_df()
        else:
            df = self.get_raw_df()

        low = np.percentile(df[self.SPREAD_SERIES_NAME], percentile * 100)
        high = np.percentile(df[self.SPREAD_SERIES_NAME], 100 - percentile * 100)
        return high - low

    def get_sensitivity(self, sma: str) -> pd.DataFrame:
        """Returns sensitivity of one leg to another

        :returns: Sensitivity series
        """
        s1, s2 = self.get_symbols()
        df: pd.DataFrame = self.get_framed_df()

        if len(df) <= 1:
            return pd.DataFrame()

        df_interval = pd.DataFrame(df.index).diff().mode().time[0]
        df_interval_secs = df_interval.total_seconds()
        idx = pd.date_range(df.index[0], df.index[-1], freq=df_interval)
        df = df.reindex(idx, method='ffill')

        sma = pd.Timedelta(sma)
        # breakpoint()

        vola = 1 # (self.get_volatility(s1, sma * 2) + self.get_volatility(s2, sma * 2)) / 2

        l1: pd.Series = (df[s1].diff() / df[s1] * 100000 / vola).dropna()
        l2: pd.Series = (df[s2].diff() / df[s2] * 100000 / vola).dropna()

        k: pd.Series = ((l1 + 1000000) / (l2 + 1000000))

        sma_secs = sma.total_seconds()

        k = k.rolling(window=sma, min_periods=int(sma_secs // df_interval_secs)).mean().dropna()

        k = (k - 1) * 100000000

        if k.empty:
            return pd.DataFrame()

        magn = np.percentile(k, 99.9) - np.percentile(k, 0.1)
        k = k / magn * 2
        return k.to_frame(self.get_name())

    def get_legs_correlation(self, framed=True) -> Tuple[float, float]:
        """Calculates correlation of spread and it's legs

        :returns: Two correlations with first and second leg as tuple: (corr_leg1, corr_leg2)"""
        c1 = self.get_correlation(self.SPREAD_SERIES_NAME, self._leg1, framed=framed)
        c2 = self.get_correlation(self.SPREAD_SERIES_NAME, self._leg2, framed=framed)
        return c1, -c2

    def get_closes(self, low=0.1, high=0.9) -> Tuple[List[pd.Timestamp], List[pd.Timestamp]]:
        """Finds low and high crosses that followed by crossing zero line

        :param low: Low cross percentile
        :param high: High cross percentile
        :returns: Low and high crosses followed by zero cross
        """
        crosses_high = self.get_crosses(self.get_percentile(high))
        crosses_center = self.get_crosses(0)
        crosses_low = self.get_crosses(self.get_percentile(low))

        all_crosses = {}
        for cross in crosses_low:
            all_crosses[cross] = 'low'

        for cross in crosses_high:
            all_crosses[cross] = 'high'

        for cross in crosses_center:
            all_crosses[cross] = 'center'

        all_crosses = sorted(list(all_crosses.items()))

        last = 'center'
        closes_low = []
        closes_high = []

        for cross_time, zone in all_crosses:
            if zone != last and zone == 'center':
                if last == 'low':
                    closes_low.append(cross_time)
                if last == 'high':
                    closes_high.append(cross_time)
            last = zone

        return closes_low, closes_high

    def plot(self):
        df = self.get_raw_df()

        s1 = self._leg1
        s2 = self._leg2

        bbh = ta.volatility.bollinger_hband(df.SPREAD, window=20 * 2)
        bbl = ta.volatility.bollinger_lband(df.SPREAD, window=20 * 2)

        with plt.rc_context({'axes.edgecolor': 'white',
                             'xtick.color': 'gray',
                             'ytick.color': 'gray',
                             'figure.facecolor': '#112329'}):
            fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(12, 6),
                                     gridspec_kw={"height_ratios": [2.5, 2.5, 1, 1]})
            ax1, ax12, ax2, ax3 = axes
            plt.tight_layout(pad=1)
            plt.subplots_adjust(hspace=0.001)

            for ax in axes:
                ax.title.set_visible(False)
                ax.set_facecolor('#112329')

            fig.canvas.manager.set_window_title(self.get_name())

            ax1.plot(df.SPREAD, color='#d62e4d', linewidth=1, label=self.get_name())
            ax1.plot(bbl, linestyle='--', linewidth=1, color='grey', label='BB Low')
            ax1.plot(bbh, linestyle='--', linewidth=1, color='grey', label='BB High')
            ax1.fill_between(df.index, bbl, bbh, alpha=0.1, color='grey')

            ax12.plot(df[s1], color='#f7c028', linewidth=1, label=s1)
            ax12.plot(df[s2], color='#3477e3', linewidth=1, label=s2)

            hours = 6

            fast_macd = ta.trend.macd(df.SPREAD, window_slow=26*hours, window_fast=12*hours)
            slow_macd = ta.trend.macd_signal(
                df.SPREAD, window_slow=26*hours, window_fast=12*hours, window_sign=9*hours)
            macd_diff = ta.trend.macd_diff(
                df.SPREAD, window_slow=26 * hours, window_fast=12 * hours,
                window_sign=9 * hours).resample('6h').mean()

            ax2.plot(fast_macd,color='#f7c028', linewidth=1, label='MACD')
            ax2.plot(slow_macd, color='#3477e3', linewidth=1, label='MACD Signal')
            # TODO: mean or sum?
            ax2.bar(macd_diff.index, macd_diff, color='white', label='MACD Diff', alpha=0.6,
                    width=0.2)
            ax2.hlines(0, df.index[0], df.index[-1], colors=['white'], linestyle='--', linewidth=1)

            ax3.set_ylim([-1.5, 1.5])
            sens_period = '7d'
            ax3.plot(self.get_sensitivity(sens_period), color='#34e3a0',
                     linewidth=1, label=f'Sensitivity {sens_period}')
            ax3.hlines([-1, 0, 1], df.index[0], df.index[-1],
                       colors=['gray', 'white', 'gray'], linestyles='dashed', linewidth=1)

            cur = MultiCursor(fig.canvas, axes, c='white', linewidth=1)
            for ax in axes:
                ax.legend(loc='upper left', framealpha=0.6)
                ax.grid(color='white', alpha=0.2, linestyle='dotted')
            plt.show()
            plt.close(fig)
