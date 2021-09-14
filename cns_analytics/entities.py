"""Defines generic classes and enums"""

import enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Union

import numba
import numpy as np
import pandas as pd

Duration = Union[str, timedelta, pd.Timedelta]
DateTime = Union[str, datetime, pd.Timestamp]


class Side(enum.Enum):
    """ Generic side
     Used in of trades and positions"""
    BUY = enum.auto()
    SELL = enum.auto()


class Direction(enum.Enum):
    """ Generic direction
     Currently used in OHLC to filter rising and falling candles"""
    UP = enum.auto()
    DOWN = enum.auto()


@dataclass
class Position:
    """ Backtest position"""
    side: Side
    pos: Dict[str, float]
    opn_time: datetime = None
    cls_time: datetime = None
    opened_money = 0
    is_closed: bool = False
    fixed_finrez: float = 0
    opn_prices: Dict[str, float] = None

    def get_revaluation(self, md):
        if self.is_closed:
            return self.fixed_finrez

        reval = self.opened_money
        for symbol, qty in self.pos.items():
            reval += qty * getattr(md, symbol)

        return reval


class Resolution(enum.Enum):
    """ Resolutions of market data"""
    """1 Minute"""
    m1 = enum.auto()
    """5 Minutes"""
    m5 = enum.auto()
    """15 Minutes"""
    m15 = enum.auto()
    """30 Minutes"""
    m30 = enum.auto()
    """1 Hour"""
    h1 = enum.auto()
    """1 Day"""
    d1 = enum.auto()


class MDType(enum.Enum):
    """ Types of market data
     Used in market data loaders to identify what data to load"""
    OHLC = enum.auto()
    MARKET_VOLUME = enum.auto()


class Exchange(enum.Enum):
    BinanceFutures = "BinanceFutures"
    BinanceSpot = "BinanceSpot"
    """More of a data source, than exchange"""
    YFinance = "YFinance"
    Bitmex = "Bitmex"
    Barchart = "Barchart"
    Finam = "Finam"


@dataclass
class Symbol:
    name: str
    exchange: Exchange = None

    def __post_init__(self):
        from cns_analytics.database import DataBase

        if self.exchange is None:
            self.exchange = DataBase.get_default_exchange()


@dataclass
class Triangle:
    """Triangle from tech analysis
    https://www.investopedia.com/terms/t/triangle.asp
    Triangle is described by four numbers (a0, a1, b0, b1)
    a0 and a1 represent upper line of the triangle, and are y's of that line.
    x's are first and last point of timeseries.
    """
    a0: float
    a1: float
    b0: float
    b1: float
    # number of data points in underlying timerseries
    n: int

    def get_sides_ratio(self) -> float:
        """ Returns ratio of sides of a trapezoid formed by
        two lines (a and b) and vertical lines at 0 and at self.n"""
        return (self.a0 - self.b0) / (self.a1 - self.b1)

    def is_upper_rising(self, threshold: float = 0):
        return self.a1 - self.a0 > threshold

    def is_lower_rising(self, threshold: float = 0):
        return self.b1 - self.b0 > threshold

    def is_upper_falling(self, threshold: float = 0):
        return self.a0 - self.a1 > threshold

    def is_lower_falling(self, threshold: float = 0):
        return self.b0 - self.b1 > threshold

    @staticmethod
    @numba.njit
    def target_function(y, direction, h, points, outside_threshold):
        line = np.linspace(y[0], y[1], h)

        if direction > 0:
            count_outside = np.sum(points < line)
        else:
            count_outside = np.sum(points > line)

        if count_outside > h * outside_threshold:
            return 1e9 + count_outside

        if direction > 0:
            return -(y[0] + y[1]) / 1e3
        else:
            return (y[0] + y[1]) / 1e3


class DropLogic(enum.Enum):
    """Logic for drop calculation"""
    # current price - last high
    SIMPLE = enum.auto()
    # last high is not updated for some period of time
    SKIP_AFTER_UPDATE = enum.auto()
    # high is calculated in window
    WINDOWED = enum.auto()
