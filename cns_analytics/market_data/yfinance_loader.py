import asyncio
import logging
from datetime import timedelta, datetime
from typing import List, Dict

import aiohttp
import pytz
import yfinance

from cns_analytics.database import DataBase
from cns_analytics.entities import Resolution, MDType, Symbol, Exchange
from cns_analytics.market_data.base_loader import BaseMDLoader, MDLoaderException


logger = logging.getLogger(__name__)


class YFinanceLoader(BaseMDLoader):
    source_id = 4

    def __init__(self):
        super().__init__()
        self._session = aiohttp.ClientSession()

    # async def _load_supported_symbols(self) -> List[Symbol]:
    #     r = await self._session.get('https://fapi.binance.com/fapi/v1/exchangeInfo')
    #     data = await r.json()
    #
    #     for limit in data['rateLimits']:
    #         if limit['rateLimitType'] == 'REQUEST_WEIGHT':
    #             self.limit = limit['limit']
    #
    #     return [Symbol(x['symbol'], exchange=Exchange.BinanceFutures)
    #             for x in data['symbols'] if x['status'] != 'BREAK']

    @staticmethod
    def get_step_for_resolution(md_type: MDType, resolution: Resolution) -> timedelta:
        if md_type in {MDType.MARKET_VOLUME, MDType.OHLC}:
            if resolution is Resolution.m1:
                return timedelta(days=1)
            elif resolution is Resolution.h1:
                return timedelta(days=60)
            elif resolution is Resolution.d1:
                return timedelta(days=365 * 5)

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

        if data['chart']['error']:
            breakpoint()
            pass
            return [], []

        quote_data = []

        try:
            for key in ['low', 'open', 'volume', 'high', 'close']:
                quote_data.append(data['chart']['result'][0]['indicators']['quote'][0][key])
        except KeyError:
            return [], []

        return data['chart']['result'][0]['timestamp'], \
               list(zip(*quote_data))

    async def _fetch(self, symbol: Symbol, md_type: MDType,
                     start: datetime, end: datetime, resolution: Resolution) -> List[Dict]:
        res_str = resolution.name
        res_str = res_str[1:] + res_str[:1]

        if md_type is MDType.OHLC:
            # df = yfinance.download([symbol.name], start=start, end=end, interval=res_str, threads=False)

            ts_data, ohlc_data = await self._rest_request(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.name}?events=div%7Csplit&useYfid=true&corsDomain=finance.yahoo.com", params={
                "formatted": "true",
                "crumb": "JAddEqUFuOp",
                "lang": "en-US",
                "region": "US",
                "includeAdjustedClose": "true",
                "interval": res_str,
                "period1": int(start.timestamp()),
                "period2": int(end.timestamp()),
                "events": "div",
                "corsDomain": "finance.yahoo.com"
            })

            data = []

            for (ts, (low, opn, volume, high, cls)) in zip(ts_data, ohlc_data):
                if opn is None:
                    continue

                ts = datetime.fromtimestamp(ts).astimezone(tz=pytz.UTC)

                if resolution is Resolution.d1:
                    ts = ts.replace(hour=0, minute=0, second=0, microsecond=0)

                data.append({
                    "ts": ts,
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
    # for symbol in (await DataBase.get_all_symbols(Exchange.YFinance)):
    #     await DataBase.drop_symbol(symbol)

    symbol = await DataBase.create_symbol('SP=F', exchange=Exchange.YFinance)

    async with YFinanceLoader() as yfl:
        await yfl.fetch(
            symbol, md_type=MDType.OHLC, duration=timedelta(days=365 * 25), resolution=Resolution.d1)


if __name__ == '__main__':
    asyncio.run(main())
