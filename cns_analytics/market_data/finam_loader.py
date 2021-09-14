import asyncio
import logging
from datetime import timedelta, datetime
from typing import List, Dict

import aiohttp
import pytz
import pandas as pd
from finam import Exporter, Market, LookupComparator, Timeframe

from cns_analytics.database import DataBase
from cns_analytics.entities import Resolution, MDType, Symbol, Exchange
from cns_analytics.market_data.base_loader import BaseMDLoader, MDLoaderException


logger = logging.getLogger(__name__)


class FinamLoader(BaseMDLoader):
    source_id = 6

    def __init__(self):
        super().__init__()
        self._session = aiohttp.ClientSession()
        self.exporter = Exporter()
        self._sym_cache = {}

    def find_symbol(self, symbol: Symbol, market: Market):
        if symbol.name in self._sym_cache:
            return self._sym_cache[symbol.name]

        sym = self.exporter.lookup(code=symbol.name, market=market,
                                   name_comparator=LookupComparator.EQUALS)
        assert len(sym) == 1

        self._sym_cache[symbol.name] = sym.index[0]

        return sym.index[0]

    @staticmethod
    def get_step_for_resolution(md_type: MDType, resolution: Resolution) -> timedelta:
        if md_type in {MDType.MARKET_VOLUME, MDType.OHLC}:
            if resolution is Resolution.m1:
                return timedelta(days=120)
            elif resolution is Resolution.h1:
                return timedelta(days=365)
            elif resolution is Resolution.d1:
                return timedelta(days=365 * 5)

        raise NotImplementedError()

    async def _fetch(self, symbol: Symbol, md_type: MDType,
                     start: datetime, end: datetime, resolution: Resolution) -> List[Dict]:
        if resolution is Resolution.m1:
            tf = Timeframe.MINUTES1
        elif resolution is Resolution.h1:
            tf = Timeframe.HOURLY
        elif resolution is Resolution.d1:
            tf = Timeframe.DAILY
        else:
            raise NotImplementedError()

        market = Market.FUTURES

        if md_type is MDType.OHLC:
            df = self.exporter.download(self.find_symbol(symbol, market),
                                        market=market,
                                        timeframe=tf,
                                        start_date=start,
                                        end_date=end)

            df['ts'] = pd.to_datetime(df['<DATE>'].astype(str) + ' ' + df['<TIME>'])
            df.drop(columns=['<DATE>', '<TIME>'], inplace=True)

            data = []

            for opn, high, low, cls, volume, ts in df.values:
                if opn is None:
                    continue
                data.append({
                    "ts": ts.to_pydatetime().astimezone(tz=pytz.UTC),
                    "px_open": float(opn),
                    "px_high": float(high),
                    "px_low": float(low),
                    "px_close": float(cls),
                    "volume": float(volume)
                })

            return data

        raise NotImplementedError("Market Data Type is not implemented:", md_type.name)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.close()


async def main():
    await DataBase.create_exchange('Finam')
    symbol = await DataBase.create_symbol('BR', exchange=Exchange.Finam)

    async with FinamLoader() as fnm:
        await fnm.fetch(
            symbol, md_type=MDType.OHLC, duration=timedelta(days=365*7), resolution=Resolution.m1)


if __name__ == '__main__':
    asyncio.run(main())
