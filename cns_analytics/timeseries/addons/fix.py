from typing import Optional

import numpy as np

from cns_analytics.timeseries import TimeSeries


class FixSetup:
    def __init__(self, fix_addon: 'FixAddon'):
        self.fix_addon = fix_addon
        self._reversed = False
        self._till = "end"
        self._entry_mask = None
        self._loss = None
        self._order_book_spread = 0
        self._fee = 0
        self._return = set()

    def reverse(self, value=True):
        self._reversed = value
        return self

    def till_loss(self):
        self._till = "loss"
        return self

    def till_end(self):
        self._till = "end"
        return self

    def set_entry_mask(self, mask: Optional[np.ndarray]):
        self._entry_mask = mask
        return self

    def add_entry_mask(self, mask: np.ndarray):
        if self._entry_mask is None:
            self._entry_mask = mask
        else:
            self._entry_mask &= mask
        return self

    def loss_by_position(self, qty: int):
        """When position equal to qty, it will be closed"""
        self._loss = "pos", qty
        return self

    def set_order_book_spread(self, spread: float):
        self._order_book_spread = spread
        return self

    def set_fee(self, fee: float):
        self._fee = fee
        return self

    def return_history(self):
        self._return.add('history')
        return self

    def calculate(self, step):
        pass


class FixAddon:
    def __init__(self, ts: TimeSeries):
        self.ts: TimeSeries = ts

    def new_setup(self):
        return FixSetup(self)


