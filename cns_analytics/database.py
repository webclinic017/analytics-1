"""Interface for database"""

import logging
import os
import sys
from typing import List

import asyncpg
import pandas as pd
from dotenv import load_dotenv

from .cache import cache_db_request
from .entities import Symbol, Exchange

__root = logging.getLogger()
__root.setLevel(logging.INFO)
__handler = logging.StreamHandler(sys.stdout)
__handler.setLevel(logging.INFO)
__handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
__root.addHandler(__handler)

load_dotenv()


class DataBase:
    _conn = None
    _pk_cache = {}
    _queries = {}
    _db_cache = {}
    _default_exchange = Exchange.BinanceFutures

    @classmethod
    async def start(cls):
        cls._conn = await asyncpg.connect(user=os.getenv('DATABASE_USER'),
                                          password=os.getenv('DATABASE_PASSWORD'),
                                          database=os.getenv('DATABASE_NAME'),
                                          host=os.getenv('DATABASE_HOST'),
                                          port=os.getenv('DATABASE_PORT'),
                                          timeout=1000)

        agg_resolutions = ['5m', '15m', '1h', '1d']

        for agg_res in agg_resolutions:
            cls._queries[f"ohlc_{agg_res}"] = await cls._conn.prepare(f'''
                                SELECT
                                  bucket AS "time",
                                  px_close
                                FROM ohlc_{agg_res}
                                WHERE symbol_id=$1
                                ORDER BY "bucket"''')

        cls._queries['ohlc_1m'] = await cls._conn.prepare('''
                            SELECT
                              ts AS "time",
                              px_close
                            FROM ohlc
                            WHERE symbol_id=$1
                            ORDER BY "ts"''')

        cls._queries['interest_rates'] = await cls._conn.prepare('''
                                    SELECT
                                      ts AS "time",
                                      value
                                    FROM interest_rate
                                    WHERE symbol_id=$1
                                    ORDER BY "ts"''')

        cls._queries['open_interests'] = await cls._conn.prepare('''
                                            SELECT
                                              ts AS "time",
                                              qty,
                                              value
                                            FROM open_interest
                                            WHERE symbol_id=$1
                                            ORDER BY "ts"''')

        cls._queries['market_volume'] = await cls._conn.prepare('''
                                                    SELECT
                                                      ts AS "time",
                                                      taker_sell_base_volume,
                                                      taker_sell_quote_volume,
                                                      taker_buy_base_volume,
                                                      taker_buy_quote_volume,
                                                      number_of_trades
                                                    FROM market_volume
                                                    WHERE symbol_id=$1
                                                    ORDER BY "ts"''')

    @classmethod
    def set_default_exchange(cls, exchange: Exchange):
        cls._default_exchange = exchange

    @classmethod
    def get_default_exchange(cls):
        return cls._default_exchange

    @classmethod
    async def check_conn(cls):
        if cls._conn is None or cls._conn.is_closed():
            await cls.start()

    @classmethod
    async def get_exchange_id(cls, exchange: Exchange) -> int:
        await cls.check_conn()

        if exchange in cls._pk_cache:
            return cls._pk_cache[exchange]

        val = await cls._conn.fetch(
            'SELECT exchange_id FROM exchanges WHERE name=$1', exchange.name)

        try:
            exchange_id = val[0]['exchange_id']
        except IndexError:
            raise Exception(f"Exchange not found: {exchange}")

        cls._pk_cache[exchange] = exchange_id
        return exchange_id

    @classmethod
    async def get_symbol_id(cls, symbol: Symbol) -> int:
        await cls.check_conn()

        cache_key = (symbol.name, symbol.exchange or cls._default_exchange)

        if cache_key in cls._pk_cache:
            return cls._pk_cache[cache_key]

        if symbol.exchange is None and cls._default_exchange is None:
            raise Exception("Selected exchange")

        exchange_id = await cls.get_exchange_id(symbol.exchange or cls._default_exchange)

        val = await cls._conn.fetch('SELECT symbol_id FROM symbol WHERE name=$1 AND exchange_id=$2',
                                    symbol.name, exchange_id)

        try:
            symbol_id = val[0]['symbol_id']
        except IndexError:
            raise Exception(f"Symbol not found: {symbol}")

        cls._pk_cache[cache_key] = symbol_id
        return symbol_id

    @classmethod
    async def get_all_symbols(cls, exchange=None) -> List[Symbol]:
        await cls.check_conn()

        if exchange is None:
            val = await cls._conn.fetch('SELECT name, exchange_id FROM symbol')
            exchanges = await cls._conn.fetch('SELECT name, exchange_id FROM exchanges')

            exchanges = {x['exchange_id']: Exchange(x['name']) for x in exchanges}
            return [Symbol(name=x['name'], exchange=exchanges[x['exchange_id']]) for x in val]
        else:
            exchange_id = await cls.get_exchange_id(exchange)
            val = await cls._conn.fetch('SELECT name, exchange_id FROM symbol WHERE exchange_id=$1',
                                        exchange_id)
            return [Symbol(name=x['name'], exchange=exchange) for x in val]

    @classmethod
    async def create_symbol(cls, name, exchange: Exchange) -> Symbol:
        await cls.check_conn()
        exchange_id = await cls.get_exchange_id(exchange)
        try:
            await cls._conn.execute('INSERT INTO symbol (name, exchange_id) VALUES ($1, $2)',
                                    name, exchange_id)
        except asyncpg.UniqueViolationError:
            pass

        return Symbol(name=name, exchange=exchange)

    @classmethod
    async def create_exchange(cls, name):
        await cls.check_conn()
        try:
            await cls._conn.execute('INSERT INTO exchanges (name) VALUES ($1)', name)
        except asyncpg.UniqueViolationError:
            pass
        return Exchange

    @classmethod
    async def drop_symbol(cls, symbol: Symbol):
        if not isinstance(symbol, Symbol):
            raise Exception(f"Expected Symbol, not {type(symbol)}")
        await cls.check_conn()
        eid = await cls.get_exchange_id(symbol.exchange)
        await cls._conn.execute('DELETE FROM symbol WHERE name=$1 AND exchange_id=$2',
                                symbol.name, eid)

    @classmethod
    @cache_db_request
    async def get_ohlcs(cls, symbol: Symbol, resolution='1h') -> pd.DataFrame:
        return await cls._fetch_by_symbol(f'ohlc_{resolution}', symbol)

    @classmethod
    async def get_interest_rates(cls, symbol) -> pd.DataFrame:
        return await cls._fetch_by_symbol('interest_rates', symbol)

    @classmethod
    async def get_open_interests(cls, symbol) -> pd.DataFrame:
        return await cls._fetch_by_symbol('open_interests', symbol)

    @classmethod
    async def get_market_volume(cls, symbol) -> pd.DataFrame:
        return await cls._fetch_by_symbol('market_volume', symbol)

    @classmethod
    async def _fetch_by_symbol(cls, query_name, symbol: Symbol):
        cache_key = (query_name, tuple(symbol.__dict__.items()))

        value = cls._db_cache.get(cache_key)

        if value is None:
            symbol_id = await cls.get_symbol_id(symbol)

            columns = [a.name for a in cls._queries[query_name].get_attributes()]
            data = await cls._queries[query_name].fetch(symbol_id)

            df = cls._data_to_df(data, columns)
            if len(df.columns) == 1:
                df.columns = [symbol.name]
            if len(df.columns) == 2:
                df.columns = [symbol.name, f'{symbol.name}_volume']#+'_' + (symbol.exchange.name if symbol.exchange is not None else cls._default_exchange.name)]

            value = df
            cls._db_cache[cache_key] = value

        return value

    @staticmethod
    def _data_to_df(data, columns):
        df = pd.DataFrame(data, columns=columns)
        df.set_index('time', inplace=True)
        df = df[~df.index.duplicated(keep='first')]

        return df

    @classmethod
    def dont_cache_ohlc(cls):
        cls._db_cache = None

    @classmethod
    def clear_ohlc_cache(cls):
        if cls._db_cache:
            cls._db_cache.clear()

    @classmethod
    def get_conn(cls) -> asyncpg.Connection:
        return cls._conn
