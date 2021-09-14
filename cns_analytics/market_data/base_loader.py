import abc
import asyncio
import logging
from datetime import timedelta, datetime
from operator import itemgetter
from typing import Union, List, Dict, Tuple

import pandas as pd
import pytz

from cns_analytics.database import DataBase
from cns_analytics.entities import Resolution, MDType, Symbol


class MDLoaderException(Exception):
    pass


class BaseMDLoader(abc.ABC):
    """ Abstract Base Market Data Loader Class

         Always use in context manager!
         Always return "ts" as CLOSE time!!!

     Generalizes process of fetching market data:
      - define time range
      - find already loaded market data
      - download data inside time range that was not already loaded
      - check data is unique
      - sort data
      - save data
    """

    _supported_symbols = {}

    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)

    @staticmethod
    def _get_database_table_name(md_type: MDType):
        return {
            MDType.OHLC: "ohlc",
            MDType.MARKET_VOLUME: "market_volume",
        }[md_type]

    @staticmethod
    async def _load_supported_symbols() -> List[Symbol]:
        """ Returns list of supported symbols or None"""
        return []

    @staticmethod
    def convert_symbol_to_external(internal_symbol: Symbol) -> str:
        """ Returns name/id of external symbol given internal one"""
        return internal_symbol.name

    @staticmethod
    def convert_symbol_to_internal(external_symbol: str) -> Symbol:
        """ Returns interval symbol given name/id of external"""
        return Symbol(name=external_symbol)

    @staticmethod
    @abc.abstractmethod
    def get_step_for_resolution(md_type: MDType, resolution: Resolution) -> timedelta:
        """ Returns how much of market data can be loaded in one request
         based on resolution and md_type"""
        pass

    @staticmethod
    def _make_data_unique(collected_data: List[Dict]) -> List[Dict]:
        """ Takes data, returns unique data"""
        return list(dict(x) for x in set(tuple(x.items()) for x in collected_data))

    @staticmethod
    def _sort_data(collected_data: List[Dict]) -> List[Dict]:
        """ Takes data, returns sorted data"""
        return sorted(collected_data, key=itemgetter('ts'))

    async def fetch(self,
                    symbol: Symbol,
                    md_type: MDType,
                    duration: Union[timedelta, pd.Timedelta],
                    resolution: Resolution) -> bool:
        """ Fetches new market data and saves it (sorted and unique)"""
        date_cursor = datetime.now().astimezone(tz=pytz.UTC)
        earliest_date = date_cursor - duration

        supported_symbols = await self.get_supported_symbols(md_type)

        if supported_symbols and symbol not in supported_symbols:
            self.logger.warning(f'{symbol.name}: Symbol is not supported')
            return False

        step = self.get_step_for_resolution(md_type, resolution)

        saved_first, saved_last = await self._get_already_saved_range(symbol, md_type)

        pending_tasks = []
        task_to_cursor_map = {}

        while date_cursor > earliest_date:
            should_load = self._should_load(saved_first, saved_last, date_cursor-step, date_cursor)

            if not should_load:
                date_cursor -= step
                continue

            task = asyncio.create_task(
                self._fetch_wrapper(symbol, md_type, date_cursor - step, date_cursor, resolution))
            pending_tasks.append(task)
            task_to_cursor_map[task] = date_cursor

            data = await task
            self.logger.info(f'{symbol.name}: Loaded {len(data)} data points '
                             f'from {(date_cursor - step).strftime("%Y-%m-%d")} '
                             f'to {date_cursor.strftime("%Y-%m-%d")}')
            if not data:
                break

            date_cursor -= step

        collected_data: List[Dict] = []

        if not pending_tasks:
            return True

        while True:
            finished, unfinished = await asyncio.wait(
                pending_tasks, return_when=asyncio.FIRST_COMPLETED)

            should_stop = False

            for task in finished:
                pending_tasks.remove(task)
                data = task.result()

                if data is None:
                    # canceled
                    continue

                cursor = task_to_cursor_map[task]
                # self.logger.info(f'{symbol.name}: Loaded {len(data)} data points '
                #                  f'from {(cursor - step).strftime("%Y-%m-%d")} '
                #                  f'to {cursor.strftime("%Y-%m-%d")}')

                if len(data) == 0:
                    should_stop = True
                    continue

                if not isinstance(data, List):
                    raise MDLoaderException(f"Expected object of type List, got:\n {data}")

                if not isinstance(data[0], Dict):
                    raise MDLoaderException(f"Expected row to be Dict, got:\n {data[0]}")

                if 'ts' not in data[0]:
                    raise MDLoaderException(f"Expected row to have key `ts`, got:\n {data[0]}")

                if not isinstance(data[0]['ts'], datetime):
                    raise MDLoaderException(f"Expected row['ts'] to be of type datetime, "
                                            f"got:\n {data[0]}")

                collected_data.extend(data)

            if len(pending_tasks) == 0:
                if len(unfinished) != 0:
                    breakpoint()
                break

            if should_stop:
                for task in pending_tasks:
                    task.cancel()

        total_loaded_count = len(collected_data)

        if saved_first is not None:
            collected_data = [row for row in collected_data
                              if not (saved_first <= row['ts'] <= saved_last)]
        collected_data = self._sort_data(collected_data)
        collected_data = self._make_data_unique(collected_data)

        self.logger.info(f'{symbol.name}: '
                         f'Got {len(collected_data)} out of {total_loaded_count} '
                         f'data points after filtering')

        await self._save_data(md_type, symbol, collected_data)

        return True

    async def _fetch_wrapper(self, *args, **kwargs):
        try:
            return await self._fetch(*args, **kwargs)
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _should_load(saved_first: datetime, saved_last: datetime,
                     start: datetime, end: datetime):
        if saved_first is None:
            return True

        if (saved_first <= start <= saved_last) and (saved_first <= end <= saved_last):
            return False
        return True

    @abc.abstractmethod
    async def _fetch(self, symbol: Symbol, md_type: MDType,
                     start: datetime, end: datetime, resolution: Resolution) -> List[Dict]:
        """ Define market data loading process here

        Data will be sorted on key "ts", so every row in dict must have that key"""
        pass

    async def _get_already_saved_range(self, symbol: Symbol, md_type: MDType) -> \
            Tuple[datetime, datetime]:
        """ Returns timestamps of first and last occurrence of data for specified symbol"""
        table_name = self._get_database_table_name(md_type)
        symbol_id = await DataBase.get_symbol_id(symbol)
        query = f"SELECT min(ts), max(ts) FROM {table_name} WHERE symbol_id=$1"
        res = await DataBase.get_conn().fetchrow(query,  symbol_id)

        return res['min'], res['max']

    async def _save_data(self, md_type: MDType, symbol: Symbol, collected_data: List[Dict]):
        if not collected_data:
            return

        symbol_id = await DataBase.get_symbol_id(symbol)

        for row in collected_data:
            row['symbol_id'] = symbol_id

        first_row = collected_data[0]
        columns_str = ', '.join(list(first_row.keys()))
        columns_idx = ', '.join(f"${x[0] + 1}" for x in list(enumerate(first_row.keys())))

        table_name = self._get_database_table_name(md_type)

        statement = f"""INSERT INTO {table_name} ({columns_str}) VALUES({columns_idx});"""

        # await DataBase.get_conn().copy_records_to_table(
        #     table_name=table_name, records=[tuple(x.values()) for x in collected_data],
        #     columns=list(first_row.keys()))

        await DataBase.get_conn().executemany(
            statement, [tuple(x.values()) for x in collected_data], timeout=1000)

        self.logger.info(f'{symbol.name}: Successfully saved {len(collected_data)} data points')

    async def get_supported_symbols(self, md_type) -> List[Symbol]:
        loader_type = type(self)

        if md_type not in loader_type._supported_symbols:
            self.logger.info(f'Loading supported symbols')
            loader_type._supported_symbols[md_type] = await self._load_supported_symbols()

        return loader_type._supported_symbols[md_type]

    async def __aenter__(self) -> "BaseMDLoader":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
