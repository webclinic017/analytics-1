import enum

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize

from cns_analytics import utils
from cns_analytics.entities import DropLogic
from cns_analytics.timeseries import TimeSeries


class SpreadOptimizerTarget(enum.Enum):
    DROP = "drop"
    DROP_PCT = "drop_pct"


class SpreadOptimizerAddon:
    def __init__(self, ts: TimeSeries):
        self.ts = ts
        self.data = None
        self.x = None

    def _get_base(self, x):
        x[0] = 1
        spread = (self.data * x).ffill().sum(axis=1)
        money = np.sum(np.abs(self.data * x), axis=1)
        drop = utils.get_drop(data=spread,
                              logic=DropLogic.SKIP_AFTER_UPDATE,
                              window='90d',
                              growth_by=500 * 50,
                              growth_during='30d').max()
        return spread, money, drop

    def _target_drop(self, x):
        spread, money, drop = self._get_base(x)

        return drop

    def _target_drop_pct(self, x):
        spread, money, drop = self._get_base(x)
        drop_pct = np.mean(drop / money) * 100

        return drop_pct

    def _optimize(self, target: SpreadOptimizerTarget, steps=500, verbose=False)\
            -> tuple[TimeSeries, np.ndarray]:
        self.data = self.ts.get_framed_df()
        self.x = np.ones(self.data.shape[1]) * 1

        target = getattr(self, f"_target_{target.value}")
        minimizer_kwargs = {'disp': verbose, 'maxiter': steps}
        x1 = minimize(target, self.x, method='Nelder-Mead', options=minimizer_kwargs)
        self.x = x1.x

        spread, money, drop = self._get_base(self.x)

        spread = pd.DataFrame(spread, index=self.ts.get_framed_df().index, columns=['SPREAD'])

        return TimeSeries.from_df(spread), self.x

    def drop(self, steps=100, verbose=False):
        return self._optimize(target=SpreadOptimizerTarget.DROP, steps=steps, verbose=verbose)

    def drop_pct(self, steps=100, verbose=False):
        return self._optimize(target=SpreadOptimizerTarget.DROP_PCT, steps=steps, verbose=verbose)
