"""Simplifies work with ohlc bars

| This module realizes OHLC class which simplifies work with OHLC bars.
| OHLC stands for Open, High, Low, Close prices inside fixed time period.
| For example:
| - open  1.3
| - high  1.4
| - low   1.1
| - close 1.2
| for 5-minute bar means that
  inside those five minutes the first market deal was at price 1.1, last at 1.2,
  highest price was 1.4 and lowest was 1.1.
| Bars can follow one another in time (bar starts where previous ends)
  or bars can go with some fixed interval one over another
  (for example bars are one hour big, but there is a bar every 1 minute).
| By default first behaviour is used, to change it use "rolling_backwards" when initializing OHLC
"""

import numpy as np
import pandas as pd

from cns_analytics.entities import Direction


class OHLC:
    """Represents a collection of OHLC bars"""
    def __init__(self, df: pd.Series, resolution: str, rolling_backwards=False):
        """Initialises OHLC instance from pandas series

        :param df: Series to initialize from
        :param resolution: Resolution to resample bars
        :param rolling_backwards: Create bar for every point in df, not every 1 resolution
        """

        if isinstance(resolution, str):
            resolution = pd.Timedelta(resolution)

        idx = pd.date_range(df.index[0], df.index[-1], freq='1T')
        df = df.reindex(idx, method='ffill')

        if rolling_backwards:
            self._df = df.shift(periods=1, freq=resolution).to_frame('open')
            self._df['high'] = df.rolling(resolution).max()
            self._df['low'] = df.rolling(resolution).min()
            self._df['close'] = df
        else:
            self._df = df.resample(resolution).ohlc()

        self._df['body'] = self._df['close'] - self._df['open']

        self._df.dropna(inplace=True)

        self._filters = {}
        self.__mask = self._df['open'].astype(np.bool)

    def _apply_filters(self):
        self.__mask = self._df['open'].astype(np.bool)

        for _filter, value in self._filters.items():
            if _filter == 'direction':
                if value is Direction.UP:
                    self.__mask &= self._df['body'] > 0
                else:
                    self.__mask &= self._df['body'] < 0

    def get_filtered_df(self) -> pd.DataFrame:
        """Returns pandas DataFrame after applying filters

        :returns: Filtered DataFrame
        """
        return self._df[self.__mask]

    def filter_direction(self, direction: Direction):
        """Applies direction filter for OHLC bars

        When direction is Direction.UP only rising bars will be selected

        :param direction: Direction to filter by
        """
        self._filters['direction'] = direction
        self._apply_filters()

    def get_body(self) -> pd.Series:
        """Returns series of bodies of bars

        Body is calculated as bar's close - open

        :returns: Body bars series
        """
        return self['body']

    def get_drop(self) -> pd.Series:
        """Returns series of drops of bars

        Drop is calculated as bar's open - low

        :returns: Drop series
        """
        return self['open'] - self['low']

    def __getitem__(self, item):
        """Returns series from DataFrame

        :param item: open, high, low, close
        :returns: Selected series
        """
        return self.get_filtered_df()[item]
