from cns_analytics.backtest.indicators.base import BaseIndicator
from cns_analytics.utils import fast_ta


class MACDDiffIndicator(BaseIndicator):
    nan_value = 0

    def __init__(self):
        super().__init__()

    def calculate(self, df_close, period_slow=26, period_fast=12, period_signal=9):
        macd = fast_ta.macd(df_close,
                            window_slow=period_slow,
                            window_fast=period_fast,
                            window_sign=period_signal)

        self._post_process(macd[0] - macd[1])

    def buy_strength(self):
        pass

    def sell_strength(self):
        pass
