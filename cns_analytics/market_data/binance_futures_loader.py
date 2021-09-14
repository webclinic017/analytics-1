import asyncio
import logging
from datetime import timedelta, datetime
from typing import List, Dict

import aiohttp
import pandas as pd
import pytz

from cns_analytics.database import DataBase
from cns_analytics.entities import Resolution, MDType, Symbol, Exchange
from cns_analytics.market_data.base_loader import BaseMDLoader, MDLoaderException


logger = logging.getLogger(__name__)


class BinanceFuturesLoader(BaseMDLoader):
    """Returns start timestamp"""
    source_id = 1

    def __init__(self):
        super().__init__()
        self._session = aiohttp.ClientSession()
        self.limit = 0
        self.current_usage = 0

    async def _load_supported_symbols(self) -> List[Symbol]:
        r = await self._session.get('https://fapi.binance.com/fapi/v1/exchangeInfo')
        data = await r.json()

        for limit in data['rateLimits']:
            if limit['rateLimitType'] == 'REQUEST_WEIGHT':
                self.limit = limit['limit']

        return [Symbol(x['symbol'], exchange=Exchange.BinanceFutures)
                for x in data['symbols'] if x['status'] != 'BREAK']

    @staticmethod
    def get_step_for_resolution(md_type: MDType, resolution: Resolution) -> timedelta:
        if md_type in {MDType.MARKET_VOLUME, MDType.OHLC}:
            if resolution is Resolution.m1:
                return timedelta(days=1)
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

        self.current_usage = int(r.headers['x-mbx-used-weight-1m'])
        print(f"{r.headers['x-mbx-used-weight-1m']}/{self.limit}")

        if isinstance(data, dict):
            # handle error
            raise MDLoaderException(f"Error loading {params['symbol']}: {data['msg']}")

        return data

    async def _fetch(self, symbol: Symbol, md_type: MDType,
                     start: datetime, end: datetime, resolution: Resolution) -> List[Dict]:
        while self.current_usage > self.limit - 100:
            await asyncio.sleep(1)

        self.current_usage += 10

        res_str = resolution.name
        res_str = res_str[1:] + res_str[:1]

        if md_type is MDType.OHLC:
            data = await self._rest_request("https://fapi.binance.com/fapi/v1/klines", params={
                "symbol": symbol.name,
                "interval": res_str,
                "startTime": int(start.timestamp() * 1e3),
                "endTime": int(end.timestamp() * 1e3),
                "limit": 1500,
            })

            data = [
                {
                    "ts": datetime.fromtimestamp(row[6] / 1e3).astimezone(tz=pytz.UTC),
                    "px_open": float(row[1]),
                    "px_high": float(row[2]),
                    "px_low": float(row[3]),
                    "px_close": float(row[4]),
                    "volume": float(row[5])
                }
                for row in data
            ]

            await asyncio.sleep(0.2)

            return data
        elif md_type is MDType.MARKET_VOLUME:
            data = await self._rest_request("https://fapi.binance.com/fapi/v1/klines", params={
                "symbol": symbol.name,
                "interval": res_str,
                "startTime": int(start.timestamp() * 1e3),
                "endTime": int(end.timestamp() * 1e3),
                "limit": 1500,
            })

            data = [
                {
                    "ts": datetime.fromtimestamp(row[6] / 1e3).astimezone(tz=pytz.UTC),
                    "taker_sell_base_volume": float(row[5]) - float(row[9]),
                    "taker_sell_quote_volume": float(row[7]) - float(row[10]),
                    "taker_buy_base_volume": float(row[9]),
                    "taker_buy_quote_volume": float(row[10]),
                    "number_of_trades": float(row[8]),
                    "source": self.source_id,
                }
                for row in data
            ]
            await asyncio.sleep(0.2)

            return data

        raise NotImplementedError("Market Data Type is not implemented:", md_type.name)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._session.close()


async def check_md_ok():
    symbols = await DataBase.get_all_symbols(exchange=Exchange.BinanceFutures)
    for symbol in symbols:
        df = await DataBase.get_ohlcs(symbol, resolution='1m')
        if df.index[1] - df.index[0] != pd.Timedelta('1 minute'):
            symbol_id = await DataBase.get_symbol_id(symbol)
            print(symbol.name, '---', symbol_id, '---', df.index[1] - df.index[0])


async def main():
    symbols = await DataBase.get_all_symbols(exchange=Exchange.BinanceFutures)

    async with BinanceFuturesLoader() as bfl:
        for symbol in symbols:
            symbol.exchange = Exchange.BinanceFutures
            await bfl.fetch(symbol=symbol,
                            md_type=MDType.OHLC,
                            duration=timedelta(days=90),
                            resolution=Resolution.m1)
            await asyncio.sleep(10)


if __name__ == '__main__':
    asyncio.run(main())
