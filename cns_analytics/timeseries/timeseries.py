"""Simplifies work with time series data
"""
import contextlib
import functools
from datetime import datetime, timedelta
from typing import Optional, Union, List, Dict

import dateutil
import numpy as np
import pandas as pd
import pytz
import ta.trend
import matplotlib.pyplot as plt
from dateutil import parser

from scipy.optimize import minimize

from cns_analytics import utils
from cns_analytics.database import DataBase
from cns_analytics.entities import Symbol, DateTime, Duration, Triangle, DropLogic
from cns_analytics.utils import get_ols_regression

import statsmodels.tsa.stattools as ts


class DateTimeIterator:
    """Allows easy iteration over timeseries
    Also can be manually shifted to a custom location with DateTimeIterator.set_pointer,
    even while iterating
    """
    def __init__(self, start: DateTime, end: DateTime, step: Duration):
        self._pointer = pd.Timestamp(start)
        self._end = pd.Timestamp(end)
        self._step = pd.Timedelta(step)

    def set_pointer(self, new_pointer):
        """Manually set pointer"""
        self._pointer = new_pointer

    def get_step(self):
        """Returns step of iterating"""
        return self._step

    def __iter__(self):
        """Returns iterator"""
        return self

    def __next__(self):
        """Iterates, shifting pointer by one step forward"""
        if self._pointer >= self._end:
            raise StopIteration
        result = self._pointer
        self._pointer += self._step
        return result


class TimeSeries:
    """Represents time series with one or multiple symbols

    | Uses concept of frame for backtest.
    | Frame allows to hide "future" information from backtest, or to test stability of assumption
      by building it inside the frame and then testing it outside.
    """

    def __init__(self, *symbols: Union[Symbol, str]):
        """Initializes TimeSeries

        :param symbols: List of symbols to load data from
        """
        symbols = [x if isinstance(x, Symbol) else Symbol(x) for x in symbols]

        self._frame_start = None
        self._frame_end = None
        self.__symbols = symbols
        self._df: pd.DataFrame = pd.DataFrame()
        self._excluded_symbols = list()
        self._figure = None
        self._pointer = None

        from cns_analytics.timeseries.addons.mask import MaskAddon
        self.mask = MaskAddon(self)

        from cns_analytics.timeseries.addons.optimizer import SpreadOptimizerAddon
        self.optimize = SpreadOptimizerAddon(self)

        from cns_analytics.timeseries.addons.fix import FixAddon
        self.fix = FixAddon(self)

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """Creates TimeSeries from pandas DataFrame

        :param df: DataFrame to create TimeSeries from
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()

        series = cls()
        series._df = df.copy()
        series._df.sort_index()
        return series

    async def load(self, start: Optional[str] = None, end: Optional[str] = None, resolution='1h'):
        """Loads prices from database

        :param start: First date to keep after loading
        :param end: Last date to keep after loading
        :param resolution: Can be any of 1m/5m/15m/1h/1d
        """
        dfs = []

        for symbol in self.__symbols:
            dfs.append(await DataBase.get_ohlcs(symbol, resolution=resolution))

        if start:
            start = pd.Timestamp(parser.parse(start, dayfirst=True)).tz_localize(pytz.UTC)
        if end:
            end = pd.Timestamp(parser.parse(end, dayfirst=True)).tz_localize(pytz.UTC)

        self._df = pd.concat(dfs, axis=1, join="inner")
        if start:
            self._df = self._df[start:]
        if end:
            self._df = self._df[: end]

        self._df.sort_index()

    def set_frame(self,
                  start: Optional[Union[str, datetime]] = None,
                  end: Optional[Union[str, datetime]] = None):
        """Changes effective range for most functions

        Used for backtesting, when data is revealed continuously

        :param start: Start of frame
        :param end: End of frame
        """
        if start:
            if isinstance(start, str):
                self._frame_start = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            else:
                self._frame_start = start.astimezone(pytz.UTC)
        else:
            self._frame_start = None

        if end:
            if isinstance(end, str):
                self._frame_end = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            else:
                self._frame_end = end.astimezone(pytz.UTC)
        else:
            self._frame_end = None

    def get_frame(self) -> [Optional[datetime], Optional[datetime]]:
        """Returns currently applied frame

        :returns: Applied frame start and end
        """
        return self._frame_start, self._frame_end

    def shift_frame(self, step: timedelta, keep_start: bool = False):
        """Shifts frame start and end

        :param step: Time to shift frame start and end by
        :param keep_start: Whether to skip shifting start
        """
        if not keep_start:
            self._frame_start += step
            self._frame_start = min(self._df.index[-1], self._frame_start)

        self._frame_end += step
        self._frame_end = min(self._df.index[-1], self._frame_end)

    @contextlib.contextmanager
    def context_frame(self,
                      start: Optional[Union[str, datetime]],
                      end: Optional[Union[str, datetime]]):
        """Saves current frame, sets new one, then restores old one back

        It's a context manager, usage:

        .. python::
            spread.set_frame("2020-10-01", "2020-12-01")
            # here frame is "2020-10-01", "2020-12-01"

            with spread.context_frame("2021-01-01", "2021-02-01"):
                # here frame is "2021-01-01", "2021-02-01"

            # here frame is "2020-10-01", "2020-12-01" again


        :param start: New frame start
        :param end: New frame end
        """
        saved_frame = self.get_frame()
        # set temp frame
        self.set_frame(start, end)
        yield
        # restore initial frame
        self.set_frame(*saved_frame)

    def exclude_symbol(self, symbol):
        """Excludes symbol from time series

        Data is preserved, but most methods will skip excluded symbols

        :param symbol: Symbol to exclude
        """
        if symbol not in self._excluded_symbols:
            self._excluded_symbols.append(symbol)

    def include_symbol(self, symbol):
        """Includes symbol back to time series

        :param symbol: Symbol to include
        """
        if symbol in self._excluded_symbols:
            self._excluded_symbols.remove(symbol)

    def get_symbols(self):
        """Returns all not excluded symbols

        :returns: List of active symbols
        """
        return [symbol for symbol in self._df.columns if symbol not in self._excluded_symbols]

    def scale_mean(self, *symbols: Optional[str], dry_run=False) -> Dict[str, float]:
        """ Scales symbols to match first symbol's mean price inside frame

        | For example take timeseries with two price series:
        | A = [1, 2, 3, 4, 4, 4, 5]
        | B = [5, 7, 7, 9, 10, 10, 10]
        | Mean for A is 3.29
        | Mean for B is 8.29
        | Coefficient for B is [A mean] / [B mean] = 3.29 / 8.29 = 0.397
        | New B = B * 0.397 = [1.98, 2.78, 2.78, 3.57, 3.97, 3.97, 3.97]

        :param symbols: List of symbols to scale
        :param dry_run: Don't scale, only return coefs
        :returns: List of scale coefficient
        """
        df = self.get_framed_df()

        symbols = symbols or self.get_symbols()

        base_symbol = symbols[0]
        base_series_mean = df[base_symbol].mean()

        coefs = {}

        for symbol in symbols:
            coef = base_series_mean / df[symbol].mean()
            coefs[symbol] = coef
            if not dry_run:
                self._df[symbol] *= coef

        return coefs

    def scale_running(self, period='30d'):
        """Scale symbols to match first symbol's running mean inside frame"""

        df = self.get_framed_df()
        base_symbol = self.get_symbols()[0]
        base_series_mean = df[base_symbol].rolling(period).mean()

        for symbol in self.get_symbols():
            running_coef = base_series_mean / df[symbol].rolling(period).mean()
            self._df[symbol] *= running_coef

    def scale_ols(self, *symbols: Optional[str], dry_run=False) -> Dict[str, float]:
        """ Scales all symbols using OLS regression with first symbol inside frame

        :param symbols: List of symbols to scale
        :param dry_run: Don't scale, only return coefs
        :returns: List of scale coefficient
        """
        df = self.get_framed_df()
        symbols = symbols or self.get_symbols()
        base_symbol = symbols[0]
        base_series = df[base_symbol]

        coefs = {}

        for symbol in symbols:
            coef = get_ols_regression(base_series, df[symbol])[1]
            coefs[symbol] = coef
            if not dry_run:
                self._df[symbol] *= coef

        return coefs

    def get_raw_df(self) -> pd.DataFrame:
        """Returns all data disregarding frame

        :returns: Full DataFrame
        """
        return self._df

    def get_framed_df(self) -> pd.DataFrame:
        """Returns framed data

        :returns: Framed DataFrame
        """
        return self._df.loc[self._frame_start: self._frame_end]

    def get_df(self, framed: bool) -> pd.DataFrame:
        """Returns data

        :param framed: Whether to frame return data
        :returns: DataFrame"""
        if framed:
            return self.get_framed_df()
        return self.get_raw_df()

    def get_frame_iterator(self, frame_start: str, frame_end: str):
        """Iterates over time series shifting frame along the way

        Frame is set to current row on every iteration

        :param frame_start: Start of frame to iterate, won't be shifted
        :param frame_end: End of frame to iterate, will shifted every iteration
        """

        self.set_frame(None, None)

        for row in self._df[frame_end:].itertuples():
            self.set_frame(frame_start, frame_end)
            yield row
            frame_end = row.Index

    def get_adf_test(self,
                     symbol: str = None,
                     max_lag: int = 1) -> tuple[float, Dict[str, float], float]:
        """Tests series for stationarity

        :param symbol: Symbol to test
        :param max_lag: Max lag for adf test

        :returns: Stationarity, key points and p-value
        """
        if len(self.get_symbols()) != 1 and symbol is None:
            raise Exception("Select one symbol!")
        elif symbol is None:
            symbol = self.get_symbols()[0]

        stationarity, key_points, p_value, *other = ts.adfuller(
            self.get_framed_df()[symbol], max_lag)

        return stationarity, key_points, p_value

    def get_hurst_exponent(self, symbol: str = None) -> float:
        """Returns the Hurst Exponent of the time series vector ts

        | 0-0.5 Mean Reverting
        | 0.5 Random Work
        | 0.5-1 Trending

        :param symbol: Symbol to get exponent
        :returns: Hurst exponent
        """

        if len(self.get_symbols()) != 1 and symbol is None:
            raise Exception("Select one symbol!")
        elif symbol is None:
            symbol = self.get_symbols()[0]

        data = self.get_framed_df()[symbol].values

        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0

    def get_correlation(self, *symbols: str, framed=True):
        """Returns correlation between two symbols

        | https://www.investopedia.com/terms/c/correlation.asp
        | If there are more symbols in time series, than two must be selected by symbols param

        :param symbols: Optional two symbols to calculate corr. for
        :param framed: Whether to frame underlying data

        :returns: Correlation between two symbols
        """
        symbols = symbols or self.get_symbols()
        if len(symbols) != 2:
            raise Exception("Expected exactly 2 symbols")

        data = self.get_df(framed=framed)

        return np.corrcoef(data[symbols[0]], data[symbols[1]])[0, 1]

    def get_autocorrelation(self, symbol: str = None, nlags: int = 40):
        """Returns autocorrelation for one symbol

        | https://www.investopedia.com/terms/a/autocorrelation.asp
        | If there are more than one symbol in time series,
        | than one must be selected by symbol param

        :param symbol: Optional symbol to calculate corr. for
        :param nlags: Number of lags to calculate

        :returns: Autocorrelation for symbol
        """
        symbols = [symbol] or self.get_symbols()
        if len(symbols) != 1:
            raise Exception("Expected exactly 1 symbol")

        data = self.get_framed_df()[symbols[0]].diff().dropna()

        return ts.acf(data, nlags=nlags, fft=True)

    def get_volatility(self, symbol: str = None, sma: Union[str, pd.Timedelta, timedelta] = None,
                       framed: bool = True):
        """Returns smoothed volatility of a chosen symbol

        If there is only one symbol you can omit "symbol" argument

        :param symbol: Symbol to calculate volatility for
        :param sma: Smoothing window size
        :param framed: Whether to frame underlying data

        :returns: Smoothed volatility pandas series
        """
        symbols = [symbol] or self.get_symbols()
        if len(symbols) != 1:
            raise Exception("Expected exactly 1 symbol")

        df = self.get_df(framed=framed)

        sma_secs = sma.total_seconds()
        df_interval = pd.DataFrame(df.index).diff().mode().time[0]
        df_interval_secs = df_interval.total_seconds()
        min_periods = int(sma_secs // df_interval_secs)

        return df[symbols[0]].rolling(sma or '14d', min_periods=min_periods).std(ddof=0)*(252**0.5)

    def dropna(self):
        """Remove NaN values from dataframe"""
        self._df.dropna(inplace=True)

    def get_crosses(self, value: float,
                    symbol: Union[Symbol, str] = None,
                    framed: bool = True) -> pd.DatetimeIndex:
        """Returns timestamps when specified value was crossed for given symbol

        If there is only one symbol you can omit "symbol" argument

        :param value: Crossing line
        :param symbol: Symbol to find crosses for
        :param framed: Whether to frame underlying data

        :returns: List of timestamps when crossing occurred
        """
        symbol = self.expect_one_symbol(symbol)

        crosses = (self.get_df(framed=framed)[symbol] > value).astype(np.int64).diff().dropna()
        return crosses[crosses != 0].index

    def get_macd_diff(self, window_slow: int, window_fast: int, window_sign: int,
                      symbol: Union[Symbol, str] = None) -> pd.Series:
        """Returns MACD indicator for selected symbol

        :param window_slow: Slow window
        :param window_fast: Fast window
        :param window_sign: Sign window
        :param symbol: Symbol to get MACD
        :returns: MACD indicator series
        """
        symbol = self.expect_one_symbol(symbol)

        data = ta.trend.macd_diff(self.get_df(framed=False)[symbol],
                                  window_slow=window_slow,
                                  window_fast=window_fast,
                                  window_sign=window_sign)

        return data

    def get_percentile(self,
                       value: Union[float, List[float]],
                       symbol: Union[Symbol, str] = None,
                       framed: bool = True):
        """Returns percentile for one symbol in underlying data

        | If value is a list, return type will be also list
        | If there are many symbols in this timeseries, than one must be selected via symbol param

        :param value: Percentile value
        :param symbol: Select one of many symbols or skip if there is only one
        :param framed: Whether to frame underlying data
        :returns: Percentile result as dataframe
        """
        symbol = self.expect_one_symbol(symbol)

        val = self.get_df(framed=framed)[symbol].quantile(value)
        if isinstance(value, list):
            return [float(x[0]) for x in val.values]
        else:
            return float(val)

    def sma(self, window: Duration, symbol: Union[Symbol, str] = None):
        symbol = self.expect_one_symbol(symbol)
        df = self.get_df(framed=False)
        data = df[symbol].rolling(pd.Timedelta(window)).mean()
        return TimeSeries.from_df(data)

    def set_pointer(self, pointer: DateTime):
        """Sets pointer"""
        self._pointer = pd.Timestamp(pointer)

    def get_pointer(self):
        """Returns pointer"""
        return self._pointer

    def get_last_timestamp(self):
        """Returns timestamp of last record"""
        return self._df.index[-1]

    def shift_pointer(self, step: Duration):
        """Shifts pointer"""
        if self._pointer is None:
            raise Exception("Pointer was not set!")
        self._pointer += pd.Timedelta(step)

    def get_before_pointer(self, window: Duration):
        """Returns timeseries before current pointer with specified window"""
        if self._pointer is None:
            raise Exception("Pointer was not set!")
        return TimeSeries.from_df(self._df[self._pointer - pd.Timedelta(window):self._pointer])

    def get_after_pointer(self, window: Duration):
        """Returns timeseries after current pointer with specified window"""
        if self._pointer is None:
            raise Exception("Pointer was not set!")
        return TimeSeries.from_df(self._df[self._pointer: self._pointer + pd.Timedelta(window)])

    def get_around_pointer(self, window_before: Duration, window_after: Duration):
        """Returns timeseries around current pointer with specified windows"""
        if self._pointer is None:
            raise Exception("Pointer was not set!")
        return TimeSeries.from_df(self._df[self._pointer - pd.Timedelta(window_before):
                                           self._pointer + pd.Timedelta(window_after)])

    def get_trend(self, symbol: Union[Symbol, str] = None, size: int = None, framed=False):
        """Finds trend in timeseries and returns it"""
        symbol = self.expect_one_symbol(symbol)
        return utils.get_trend(self.get_df(framed=framed)[symbol], out_size=size)

    def remove_trend(self, symbol: Union[Symbol, str] = None):
        """Finds trend in timeseries and removes it from data"""
        symbol = self.expect_one_symbol(symbol)
        trend = self.get_trend(symbol)
        self._df[symbol] -= trend

    def get_datetime_iterator(self,
                              step: Duration, framed=True,
                              skip_start: Duration = '0m') -> DateTimeIterator:
        """Returns datetime iterator for current dataframe"""
        df = self.get_df(framed=framed)
        return DateTimeIterator(start=df.index[0] + pd.Timedelta(skip_start),
                                end=df.index[-1],
                                step=step)

    def resample(self, dur: Duration, inplace=True):
        """Changes time step of timeseries, irreversible if inplace"""
        df = self._df.resample(pd.Timedelta(dur)).last()
        if inplace:
            self._df = df
        else:
            return type(self).from_df(df)

    def get_triangle(self, symbol: Union[Symbol, str] = None, outside_threshold=0.05) -> Triangle:
        """Returns best triangle for this timeseries
        :param symbol: Symbol to work with
        :param outside_threshold: Max pct of points outside of upper or lower triangle lines
            (each side is counted separately)
        """
        symbol = self.expect_one_symbol(symbol=symbol)
        points = self.get_df(framed=False)[symbol].values
        point_count = points.size
        half_point_count = int(round(point_count / 2))

        xa0 = np.array([np.max(points[:half_point_count]), np.max(points[half_point_count:])])
        xb0 = np.array([np.min(points[:half_point_count]), np.min(points[half_point_count:])])

        minimizer_kwargs = {'disp': False, 'maxiter': 10e3}

        xa2 = minimize(
            functools.partial(Triangle.target_function, direction=-1, h=point_count, points=points,
                              outside_threshold=outside_threshold), xa0,
            method='Nelder-Mead', options=minimizer_kwargs)
        xa3 = minimize(
            functools.partial(Triangle.target_function, direction=1, h=point_count, points=points,
                              outside_threshold=outside_threshold), xb0,
            method='Nelder-Mead', options=minimizer_kwargs)

        a0, a1 = xa2.x
        b0, b1 = xa3.x

        return Triangle(a0=a0, a1=a1, b0=b0, b1=b1, n=point_count)

    def get_drop(self,
                 logic: DropLogic = DropLogic.SIMPLE,
                 window: Optional[Duration] = None,
                 growth_by: Optional[float] = None,
                 growth_during: Optional[Duration] = None,
                 std_period: Duration = '30d',
                 std_limit: Optional[float] = None,
                 symbol: Optional[Union[Symbol, str]] = None) -> pd.Series:
        """Returns drop according to provided logic"""
        symbol = self.expect_one_symbol(symbol=symbol)
        df = self.get_raw_df()[symbol]

        return utils.get_drop(df, logic=logic, window=window,
                              growth_by=growth_by, growth_during=growth_during,
                              std_period=std_period, std_limit=std_limit)

    def plot(self, *symbols: Union[Symbol, str], title: str = ""):
        """Plots specified or all symbols"""
        df = self._df

        for symbol in symbols:
            if isinstance(symbol, Symbol):
                symbol = symbol.name

            plt.plot(df[symbol], label=symbol)

        if not symbols:
            df.plot()
        else:
            plt.legend()

        plt.grid()
        if title:
            plt.title(title)

        plt.show()

    def expect_one_symbol(self, symbol: Optional[Union[Symbol, str]] = None):
        """Raises Exception if no symbol is passed and timeseries has one or more symbol
        otherwise returns passed symbol, or that only symbol, that timeseries has"""
        if isinstance(symbol, Symbol):
            symbol = symbol.name

        symbols = [symbol] if symbol is not None else self.get_symbols()
        if len(symbols) != 1:
            raise Exception("Excepted exactly one symbol")

        return symbols[0]

    def __setitem__(self, key, value):
        """Updates/sets underlying data key by value

        :param key: Key to update/set
        :param value: New value
        """
        if isinstance(value, TimeSeries):
            if len(value.get_symbols()) != 1:
                raise Exception("Can't set more than one symbol!")
            self._df[key] = value._df[value.get_symbols()[0]]
        else:
            self._df[key] = value

    def __getitem__(self, item) -> 'TimeSeries':
        """Returns new timeseries from underling's data item

        :returns: New timeseries
        """
        if isinstance(item, int):
            return self._df.iloc[item]
        return TimeSeries.from_df(self._df[item])

    def __iter__(self):
        """Creates iterator for underlying pandas data frame

        :returns: DataFrame iterator
        """
        return (x for x in self.get_framed_df().itertuples())

    def __bool__(self):
        return not self._df.empty

    def __str__(self):
        """Represents underlying data as string

        :returns: Underlying data as string
        """
        return str(self._df)

    def __len__(self):
        """Length of underlying data

        :returns: Number of points
        """

        return len(self._df)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return self.__truediv__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __gt__(self, other):
        symbol = self.expect_one_symbol(None)
        if isinstance(other, TimeSeries):
            other_symbol = other.expect_one_symbol(None)
            return self.get_raw_df()[symbol] > other.get_raw_df()[other_symbol]
        raise NotImplementedError()

    def __lt__(self, other):
        symbol = self.expect_one_symbol(None)
        if isinstance(other, TimeSeries):
            other_symbol = other.expect_one_symbol(None)
            return self.get_raw_df()[symbol] < other.get_raw_df()[other_symbol]
        raise NotImplementedError()


    def __mul__(self, other):
        """Multiplies timeseries by timeseries, float, numpy array or pandas series/frame

        :param other: Multiplicand
        :returns: Self
        """
        if isinstance(other, TimeSeries):
            self._df *= other.get_raw_df().values
        else:
            self._df *= other
        return self

    def __add__(self, other):
        """Adds timeseries, float, numpy array or pandas series/frame to timeseries

        :param other: Addend
        :returns: Self
        """
        if isinstance(other, TimeSeries):
            self._df += other.get_raw_df().values
        else:
            self._df += other
        return self

    def __sub__(self, other):
        """Subtract timeseries, float, numpy array or pandas series/frame from timeseries

        :param other: Subtrahend
        :returns: Self
        """
        if isinstance(other, TimeSeries):
            self._df -= other.get_raw_df().values
        else:
            self._df -= other
        return self

    def __truediv__(self, other):
        """Divides timeseries by timeseries, float, numpy array or pandas series/frame

        :param other: Divisor
        :returns: Self
        """
        if isinstance(other, TimeSeries):
            self._df /= other.get_raw_df().values
        else:
            self._df /= other
        return self
