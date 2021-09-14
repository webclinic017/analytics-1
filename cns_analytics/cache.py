"""Interface for caching"""
import functools
import os

import pandas as pd

from cns_analytics.entities import Symbol, Exchange


CACHE_DIR = "./.cache/"


def _to_key(obj):
    import cns_analytics.database

    if isinstance(obj, Symbol):
        return obj.exchange.name + ';' + obj.name
    elif isinstance(obj, Exchange):
        return obj.name
    elif obj is cns_analytics.database.DataBase:
        return "db"
    return str(obj)


def cache_exists(key):
    return os.path.exists(os.path.join(CACHE_DIR, key))


def read_cache(key) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(CACHE_DIR, key))


def store_cache(key, df: pd.DataFrame):
    return df.to_pickle(os.path.join(CACHE_DIR, key))


def cache_db_request(func) -> pd.DataFrame:
    @functools.wraps(func)
    async def new_func(*args, **kwargs) -> pd.DataFrame:
        key = ""
        for arg in args:
            key += f"{_to_key(arg)};"
        for kwarg, val in kwargs.items():
            key += f"{kwarg}={_to_key(val)};"

        if not cache_exists(key):
            value = await func(*args, **kwargs)
            store_cache(key, value)
        else:
            value = read_cache(key)

        return value

    return new_func
