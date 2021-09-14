import asyncio
import logging
from datetime import timedelta, datetime
from typing import List, Dict

import pytz
import pandas as pd

from cns_analytics.database import DataBase
from cns_analytics.entities import Resolution, MDType, Symbol, Exchange
from cns_analytics.market_data.base_loader import BaseMDLoader

logger = logging.getLogger(__name__)


class BarchartFileLoader(BaseMDLoader):
    source_id = 3

    async def _load_supported_symbols(self) -> List[Symbol]:
        return []

    @staticmethod
    def get_step_for_resolution(md_type: MDType, resolution: Resolution) -> timedelta:
        return timedelta(days=365)

    async def _fetch(self, symbol: Symbol, md_type: MDType,
                     start: datetime, end: datetime, resolution: Resolution) -> List[Dict]:
        df = pd.read_excel('/Users/kostya/Downloads/si.xlsx', engine='openpyxl', sheet_name=3)
        df = df.iloc[1:]
        df.columns = ['ts', 'symbol', 'close', 'open', 'high', 'low', 'volume']
        df.drop(columns=['symbol'], inplace=True)
        # df.columns = ['ts', 'close', 'open', 'high', 'low', 'volume']
        print('loaded data', df.ts.iloc[0].date(), df.ts.iloc[-1].date())

        data = [
            {
                "ts": row[0].astimezone(tz=pytz.UTC),
                "px_open": float(row[2]),
                "px_high": float(row[3]),
                "px_low": float(row[4]),
                "px_close": float(row[1]),
                "volume": float(row[5])
            }
            for row in df.values
        ]

        return data

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


async def main():
    # await DataBase.drop_symbol(Symbol('TIP', exchange=Exchange.Barchart))
    symbol = await DataBase.create_symbol('SI', exchange=Exchange.Barchart)

    async with BarchartFileLoader() as yfl:
        await yfl.fetch(
            symbol, md_type=MDType.OHLC, duration=timedelta(days=360), resolution=Resolution.m1)


if __name__ == '__main__':
    asyncio.run(main())
