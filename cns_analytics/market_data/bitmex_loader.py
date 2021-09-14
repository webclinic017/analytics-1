import pandas as pd
import asyncio
import logging
from datetime import timedelta, datetime
from typing import List, Dict

import aiohttp
import pytz

from cns_analytics.database import DataBase
from cns_analytics.entities import Resolution, MDType, Symbol, Exchange
from cns_analytics.market_data.base_loader import BaseMDLoader, MDLoaderException


logger = logging.getLogger(__name__)


class BitmexLoader(BaseMDLoader):
    """Returns end timestamp, we expect start, so need to shift"""
    source_id = 5

    def __init__(self):
        super().__init__()
        self._session = aiohttp.ClientSession()
        self.limit = 0
        self.current_usage = 0

    async def _load_supported_symbols(self) -> List[Symbol]:
        r = await self._session.get('https://www.bitmex.com/api/v1/instrument/active')
        data = await r.json()

        return [Symbol(x['symbol'], exchange=Exchange.Bitmex) for x in data]

    @staticmethod
    def get_step_for_resolution(md_type: MDType, resolution: Resolution) -> timedelta:
        if md_type in {MDType.MARKET_VOLUME, MDType.OHLC}:
            if resolution is Resolution.m1:
                return timedelta(hours=16)
            elif resolution is Resolution.h1:
                raise timedelta(days=30)

        raise NotImplementedError()

    async def _rest_request(self, url, params, is_retry=False):
        try:
            r = await self._session.get(url, params=params)
        except aiohttp.ClientOSError:
            if is_retry:
                raise

            await self._session.close()
            self._session = aiohttp.ClientSession()
            self.logger.exception("ClientOSError. Retrying...")
            await asyncio.sleep(1)
            return await self._rest_request(url, params, is_retry=True)

        try:
            data = await r.json()
        except aiohttp.ClientConnectionError:
            data = await r.text()
            breakpoint()
            raise
        except aiohttp.ClientPayloadError:
            breakpoint()
            data = await r.text()
            raise

        if isinstance(data, dict):
            # handle error
            raise MDLoaderException(f"Error loading {params['symbol']}: {data['error']}")

        return data

    async def _fetch(self, symbol: Symbol, md_type: MDType,
                     start: datetime, end: datetime, resolution: Resolution) -> List[Dict]:
        # while self.current_usage > self.limit - 100:
        #     await asyncio.sleep(1)
        #
        # self.current_usage += 10

        res_str = resolution.name
        res_str = res_str[1:] + res_str[:1]

        shift = pd.Timedelta(res_str.replace('m', 'T'))

        if md_type is MDType.OHLC:
            data = await self._rest_request("https://www.bitmex.com/api/v1/trade/bucketed", params={
                "symbol": symbol.name,
                "binSize": res_str,
                "startTime": start.strftime("%Y-%m-%d %H:%M:%S"),
                "endTime": end.strftime("%Y-%m-%d %H:%M:%S"),
                "count": 1000,
            })

            data = [
                {
                    "ts": datetime.strptime(row['timestamp'], "%Y-%m-%dT%H:%M:%S.000Z").astimezone(
                        tz=pytz.UTC) - shift,
                    "px_open": float(row['open']),
                    "px_high": float(row['high']),
                    "px_low": float(row['low']),
                    "px_close": float(row['close']),
                    "volume": float(row['volume'])
                }
                for row in data
                if row['open'] is not None
            ]

            await asyncio.sleep(2)

            return data

        raise NotImplementedError("Market Data Type is not implemented:", md_type.name)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.close()


async def main():
    await DataBase.create_exchange('Bitmex')
    await DataBase.create_symbol('XBTUSD', Exchange.Bitmex)
    await DataBase.create_symbol('UNIUSD', Exchange.Bitmex)
    await DataBase.create_symbol('XRPUSD', Exchange.Bitmex)
    await DataBase.create_symbol('LINKUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('BNBUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('ADAUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('EOSUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('TRXUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('SOLUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('FILUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('AAVEUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('SUSHIUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('BCHUSD', Exchange.Bitmex)
    await DataBase.create_symbol('DOTUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('DEFIMEXUSD', Exchange.Bitmex)
    await DataBase.create_symbol('XLMUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('VETUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('MATICUSDT', Exchange.Bitmex)
    await DataBase.create_symbol('XBTEUR', Exchange.Bitmex)
    await DataBase.create_symbol('ETHUSD', Exchange.Bitmex)
    await DataBase.create_symbol('LTCUSD', Exchange.Bitmex)
    symbols = await DataBase.get_all_symbols(exchange=Exchange.Bitmex)

    async with BitmexLoader() as bfl:
        for symbol in symbols:
            print(symbol.name)
            symbol.exchange = Exchange.Bitmex
            await bfl.fetch(symbol=symbol,
                            md_type=MDType.OHLC,
                            duration=timedelta(days=220),
                            resolution=Resolution.m1)
            await asyncio.sleep(10)


if __name__ == '__main__':
    asyncio.run(main())
