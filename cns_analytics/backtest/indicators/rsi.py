import numpy as np

from cns_analytics.backtest.indicators.base import BaseIndicator
from cns_analytics.utils import fast_ta


class RSIIndicator(BaseIndicator):
    nan_value = 50

    def __init__(self, low_percentile=20, high_percentile=80):
        super().__init__()
        self.history_low_limit = []
        self.history_high_limit = []
        self.low_limit = 0
        self.high_limit = 0

        self._low_percentile = low_percentile
        self._high_percentile = high_percentile

    def calculate(self, df_close, period):
        rsi = fast_ta.rsi(df_close=df_close, period=period)

        self._post_process(rsi)

        if not self.history_low_limit:
            init_rsi = rsi.copy()
            init_rsi[:] = self.nan_value
            self.history_low_limit = init_rsi.tolist()
            self.history_high_limit = init_rsi.tolist()

        low = 50 - np.nanpercentile(np.abs(rsi), self._low_percentile)
        high = np.nanpercentile(np.abs(rsi), self._high_percentile) - 50

        best = max(low, high)

        self.low_limit = 50 - best
        self.high_limit = 50 + best

        self.history_low_limit.append(self.low_limit)
        self.history_high_limit.append(self.high_limit)

    def buy_strength(self):
        if self.last is None:
            return 0
        return float(self.last <= self.low_limit)

    def sell_strength(self):
        if self.last is None:
            return 0
        return float(self.last >= self.high_limit)
