# multivariate lstm example
import asyncio

from keras.callbacks import LearningRateScheduler
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from cns_analytics import utils
from cns_analytics.entities import Symbol, Exchange
from cns_analytics.timeseries import TimeSeries

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


def split_sequences(sequences, n):
    X, y = list(), list()
    for i in range(len(sequences) - n):
        X.append(sequences[i:i + n, :] / sequences[i:i + n, :][0])
        y.append(sequences[i + n, ][-1] - sequences[i + n - 1, ][-1])
        # y.append(sequences[i + n, ][-1] / sequences[i:i + n, :][0][-1])

    y = np.array(y)
    y /= np.percentile(np.abs(y), 99)

    return np.array(X), y


async def main():
    s1 = Symbol('RTS', Exchange.Finam)
    s2 = Symbol('Si', Exchange.Finam)

    ts = TimeSeries(s1, s2)
    await ts.load(resolution='1m', start='2017-06-01', end='2021-05-01')

    df = ts.get_raw_df()

    spread = df[s1.name] * 0.02 * df[s2.name] / 1000 + df[s2.name] * 5
    # spread -= utils.get_trend(spread)

    df['spread'] = spread / 1000

    df = df.resample('1H').last().dropna()

    data = df.values

    X, y = split_sequences(data, 350)

    print('Total X len:', X.shape[0])

    train_len = int(X.shape[0] * 0.90)

    print('Train X len:', train_len)

    X_train, y_train = X[:train_len], y[:train_len]
    X_test, y_test = X[train_len:], y[train_len:]

    model = Sequential()
    model.add(LSTM(100, activation='relu',
                   input_shape=(X.shape[1], X.shape[2]),
                   return_sequences=True))
    model.add(LSTM(100,
                   activation='relu',))
    model.add(Dense(1))
    opt = 'adam'

    opt = AdamOptimizer(learning_rate=0.003)
    model.compile(optimizer=opt, loss='mse')

    def decay(epoch, lr):
        dec = 0.01 / 5
        return lr * 1 / (1 + dec * epoch)

    model.fit(X_train, y_train, epochs=10, batch_size=1200)
    # model.optimizer.learning_rate.assign(0.0001)

    # yhat = model.predict(X_test, verbose=0).reshape(-1)
    #
    # grown_real = y_test > X_test[:, -1][:, -1]
    # grown_test = yhat > X_test[:, -1][:, -1]
    #
    # print((grown_real == grown_test).mean())

    breakpoint()
    pass


if __name__ == '__main__':
    asyncio.run(main())
