import abc
from collections import defaultdict

from .entities import Side, Position
from .timeseries import TimeSeries


class PositionBacktest(abc.ABC):
    def __init__(self, ts: TimeSeries):
        self.ts = ts
        self.md = None
        self.positions = defaultdict(list)
        self._qty = defaultdict(float)
        self._opened_money = 0
        self.reval_history = []

    async def run(self):
        for row in self.ts:
            await self.tick(row)

    async def tick(self, row):
        self.md = row
        if self.should_open(Side.BUY):
            self._process_position(self.open_position(Side.BUY))

        for pos in self.positions[Side.BUY]:
            if self.should_close(pos):
                self._close_position(pos)
                return

        self.reval_history.append(self.get_revaluation())

        # await asyncio.sleep(0.03)

    @abc.abstractmethod
    def should_open(self, side: Side):
        pass

    @abc.abstractmethod
    def should_close(self, pos: Position):
        pass

    @abc.abstractmethod
    def open_position(self, side) -> Position:
        pass

    def _process_position(self, pos: Position):
        pos.opn_time = self.md.Index

        self.positions[pos.side].append(pos)

        opn_prices = {}

        for symbol, qty in pos.positions.items():
            self._qty[symbol] += qty
            px = getattr(self.md, symbol)
            pos.opened_money -= qty * px
            opn_prices[symbol] = px

        pos.opn_prices = opn_prices

        self._opened_money += pos.opened_money

    def _close_position(self, pos: Position):
        pos.fixed_finrez = pos.get_revaluation(self.md)
        pos.is_closed = True
        pos.cls_time = self.md.Index

        self.positions[pos.side].remove(pos)

        for symbol, qty in pos.positions.items():
            self._qty[symbol] -= qty
            self._opened_money += qty * getattr(self.md, symbol)

    def get_revaluation(self):
        reval = self._opened_money
        for symbol in self._qty:
            reval += self._qty[symbol] * getattr(self.md, symbol)

        return reval
