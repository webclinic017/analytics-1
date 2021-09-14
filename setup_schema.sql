-- CREATE USER cns_user WITH ENCRYPTED PASSWORD ''; --

CREATE DATABASE cns_db;

GRANT ALL PRIVILEGES ON DATABASE cns_db TO cns_user;

CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- \c cns_db

CREATE TABLE IF NOT EXISTS exchanges (
   exchange_id serial PRIMARY KEY,
   name varchar(32),
   CONSTRAINT unique_exchange_name UNIQUE (name)
);

insert into exchanges (name) values ('BinanceFutures');
insert into exchanges (name) values ('BinanceSpot');

CREATE TABLE IF NOT EXISTS symbol (
   symbol_id serial PRIMARY KEY,
   exchange_id int not null,
   name varchar(32),

   CONSTRAINT fk_exchange
     FOREIGN KEY(exchange_id)
	   REFERENCES exchanges(exchange_id)
       ON DELETE CASCADE,
   CONSTRAINT unique_name_exchange_id UNIQUE (name, exchange_id)
);

CREATE TABLE IF NOT EXISTS ohlc (
   px_open double precision,
   px_high double precision,
   px_low double precision,
   px_close double precision,
   volume double precision,
   ts timestamptz,
   symbol_id int not null,
   CONSTRAINT fk_symbol
      FOREIGN KEY(symbol_id)
	    REFERENCES symbol(symbol_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS interest_rate (
   value double precision,
   ts timestamptz,
   symbol_id int not null,
   source int,
   CONSTRAINT fk_symbol
      FOREIGN KEY(symbol_id)
	    REFERENCES symbol(symbol_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS open_interest (
   qty double precision,
   value double precision,
   ts timestamptz,
   symbol_id int not null,
   source int,
   CONSTRAINT fk_symbol
      FOREIGN KEY(symbol_id)
	    REFERENCES symbol(symbol_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS market_volume (
   taker_sell_base_volume double precision,
   taker_sell_quote_volume double precision,
   taker_buy_base_volume double precision,
   taker_buy_quote_volume double precision,
   number_of_trades integer,

   ts timestamptz,
   symbol_id int not null,
   source int,
   CONSTRAINT fk_symbol
      FOREIGN KEY(symbol_id)
	    REFERENCES symbol(symbol_id)
        ON DELETE CASCADE
);


-- SELECT create_hypertable('ohlc', 'ts', partitioning_column => 'symbol_id');
SELECT create_hypertable('ohlc', 'ts');

CREATE INDEX ohlc_symbol_id_ts_idx ON ohlc (symbol_id, ts ASC);
CREATE INDEX interest_rate_symbol_id_ts_idx ON interest_rate (symbol_id, ts ASC);
CREATE INDEX market_volume_symbol_id_ts_idx ON market_volume (symbol_id, ts ASC);
CREATE INDEX open_interest_symbol_id_ts_idx ON open_interest (symbol_id, ts ASC);

--- login as cns_user

CREATE MATERIALIZED VIEW ohlc_5m AS
SELECT symbol_id,
       time_bucket(INTERVAL '5 minutes', ts) AS bucket,
       first(px_open, ts) as px_open,
       MAX(px_high) as px_high,
       MIN(px_low) as px_low,
       last(px_close, ts) as px_close
FROM ohlc
GROUP BY symbol_id, bucket;

CREATE MATERIALIZED VIEW ohlc_15m AS
SELECT symbol_id,
       time_bucket(INTERVAL '15 minutes', ts) AS bucket,
       first(px_open, ts) as px_open,
       MAX(px_high) as px_high,
       MIN(px_low) as px_low,
       last(px_close, ts) as px_close
FROM ohlc
GROUP BY symbol_id, bucket;

CREATE MATERIALIZED VIEW ohlc_1h AS
SELECT symbol_id,
       time_bucket(INTERVAL '1 hour', ts) AS bucket,
       first(px_open, ts) as px_open,
       MAX(px_high) as px_high,
       MIN(px_low) as px_low,
       last(px_close, ts) as px_close
FROM ohlc
GROUP BY symbol_id, bucket;

CREATE MATERIALIZED VIEW ohlc_1d AS
SELECT symbol_id,
       time_bucket(INTERVAL '1 day', ts) AS bucket,
       first(px_open, ts) as px_open,
       MAX(px_high) as px_high,
       MIN(px_low) as px_low,
       last(px_close, ts) as px_close
FROM ohlc
GROUP BY symbol_id, bucket;


REFRESH MATERIALIZED VIEW ohlc_5m;
REFRESH MATERIALIZED VIEW ohlc_15m;
REFRESH MATERIALIZED VIEW ohlc_1h;
REFRESH MATERIALIZED VIEW ohlc_1d;