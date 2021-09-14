import numpy as np
import abc


class BaseIndicator(abc.ABC):
    nan_value = 0

    def __init__(self):
        # last value of indicator
        self.last = None
        # all last values of indicator
        self.data = np.empty(1)
        # history of last values for all past data points
        self.history = []

    def _post_process(self, data):
        self.last = data[-1]
        self.data = data

        if not self.history:
            init_data = data.copy()
            init_data[np.isnan(init_data)] = self.nan_value
            self.history = init_data.tolist()
        else:
            self.history.append(data[-1])

    @abc.abstractmethod
    def buy_strength(self) -> float:
        pass

    @abc.abstractmethod
    def sell_strength(self) -> float:
        pass
