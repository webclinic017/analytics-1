import pytz
import numpy as np
import pandas as pd

from cns_analytics.entities import Duration
from cns_analytics.timeseries import TimeSeries


class SeriesTrendStability:
    """Works with trend of series"""
    def __init__(self,
                 series: TimeSeries,
                 window_backward: Duration,
                 symbol: str):
        self.series = series
        self.window_backward = pd.Timedelta(window_backward)
        self.pointer = series.get_raw_df().index[0] + self.window_backward
        self.symbol = symbol

    def _is_series_stable(self, series: TimeSeries, frame: Duration, threshold: float):
        trend = series[:frame].get_trend(symbol=self.symbol, size=len(series))
        diff = series.get_raw_df()[self.symbol] - trend
        # print(np.percentile(np.abs(diff), 99) )
        return np.percentile(np.abs(diff), 99) < threshold

    def is_past_stable(self, threshold: float):
        """Returns whether 90'th percentile of difference of trend and past series is smaller
        than threshold"""
        past_series = self.series[self.pointer - self.window_backward: self.pointer]
        if len(past_series) <= 1:
            return False
        return self._is_series_stable(past_series, frame=self.pointer, threshold=threshold)

    def get_future_stability(self, threshold: float):
        """Returns duration while price didn't fall below threshold from running high"""
        data = self.series.get_raw_df()[self.pointer:][self.symbol]
        diff = data.cummax() - data
        exit_time = (diff > threshold).idxmax().astimezone(tz=pytz.UTC)
        work_range = self.series.get_raw_df()[self.pointer: exit_time].index
        dates = work_range.to_series().dt.date.unique()
        last_day = exit_time - exit_time.normalize()
        return pd.Timedelta('1d') * (len(dates) - 1) + last_day, exit_time
