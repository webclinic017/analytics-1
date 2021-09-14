from collections import defaultdict


class SimpleExchange:
    def __init__(self, fee: float):
        self.qty = defaultdict(int)
        self.prices = {}
        self.money = 0
        self.fee = fee

    def set_price(self, sec: str, price: float):
        self.prices[sec] = price

    def buy(self, sec: str, qty: int):
        assert qty > 0
        self.qty[sec] += qty
        self.money -= self.prices[sec] * qty
        self.money -= self.fee
        print('    exp buy', sec, qty, int(self.prices[sec]), 'pos', self.get_position(sec))

    def sell(self, sec: str, qty: int):
        assert qty > 0
        self.qty[sec] -= qty
        self.money += self.prices[sec] * qty
        self.money -= self.fee
        print('    exp sell', sec, qty, int(self.prices[sec]), 'pos', self.get_position(sec))

    def trade_difference(self, sec: str, expected_qty: int, tolerance: float = 1):
        difference = self.qty[sec] - expected_qty

        if difference > tolerance or (difference == tolerance and tolerance > 0):
            self.sell(sec, difference)
        elif difference < -tolerance or (difference == -tolerance and tolerance > 0):
            self.buy(sec, -difference)

    def get_position(self, sec: str):
        return self.qty[sec]

    def get_revaluation(self):
        return int(round(sum([self.qty[sec] * self.prices[sec] for sec in self.qty]) + self.money))
