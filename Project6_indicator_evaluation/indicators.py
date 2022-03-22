import datetime as dt
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data


def author():
    return "chuang405"

def sma(df_prices, window = 5):
    sma = df_prices.rolling(window = window, min_periods = window).mean()
    return(sma)

def priceSMA(df_prices, window = 5):
    sma = df_prices.rolling(window = window, min_periods = window).mean()
    price_sma_ratio = df_prices/sma
    return price_sma_ratio

def bollinger(df_prices, window = 5):
    std = df_prices.rolling(window=window, min_periods=window).std()
    sma = df_prices.rolling(window=window, min_periods=window).mean()
    sell = sma + std * 2
    buy = sma - std * 2
    return sell, buy

def momentum(df_prices, window = 10):
    momentum = df_prices/df_prices.shift(window) - 1
    return momentum

def ema(df_prices, window = 5):
    ema = df_prices.ewm(span=window, adjust=False).mean()
    return ema

def MACD(df_prices, window1 = 10, window2 = 30):
    ema_short = df_prices.ewm(span = window1, adjust = False).mean()
    ema_long = df_prices.ewm(span=window2, adjust=False).mean()
    return ema_short, ema_long


def test_code():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    df_prices = get_data([symbol], pd.date_range(sd,ed))
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=False)
    df_prices_symbol = df_prices[symbol]
    df_prices_norm = df_prices_symbol/ df_prices_symbol.iloc[0]

    # plot price SMA ratio, and normalized prices
    plt.figure(1)
    plt.plot(priceSMA(df_prices_symbol))
    plt.plot(sma(df_prices_norm))
    plt.plot(df_prices_norm)
    plt.title('Price SMA Ratio VS. Normalized Price')
    plt.axhline(1.05, linestyle = "--")
    plt.axhline(0.95, linestyle = "--")
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.legend(labels=['price/RMA','SMA','Norm Price'], loc='best')
    plt.savefig('Figure2')

    #plot Bolinger Band
    upper, lower = bollinger(df_prices_norm, window = 10)
    percentage = (df_prices_norm - lower)/(upper - lower)
    plt.figure(2)
    plt.plot(upper)
    plt.plot(lower)
    plt.plot(df_prices_norm)
    plt.plot(percentage)
    plt.axhline(1, linestyle="--")
    plt.axhline(0, linestyle="--")
    plt.title('Bollinger Band VS. Normalized Price')
    plt.xlabel('Date')
    plt.legend(labels=['Bollinger band Upper', 'Bollinger band lower','Norm Price', 'Bollinger Band %'], loc='best')
    plt.savefig('Figure3')

    #plot momentum
    plt.figure(3)
    plt.plot(momentum(df_prices_norm))
    plt.plot(df_prices_norm)
    plt.title('Momentum VS. Normalized Price')
    plt.axhline(0, linestyle = "--")
    plt.xlabel('Date')
    plt.legend(labels=['Momentum', 'Norm Price'], loc='best')
    plt.savefig('Figure4')

    #plot EMA
    ema_line = ema(df_prices_norm)
    ema_ratio = df_prices_norm/ema_line
    plt.figure(4)
    plt.plot(ema_ratio)
    plt.plot(ema_line)
    plt.plot(df_prices_norm)
    plt.title('EMA ratio VS. Normalized Price')
    plt.axhline(1.05, linestyle="--")
    plt.axhline(0.95, linestyle="--")
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.legend(labels=['price/EMA','EMA', 'Norm Price'], loc='best')
    plt.savefig('Figure5')

    #plot MACD
    plt.figure(5)
    ema_short, ema_long = MACD(df_prices_norm)
    plt.plot(ema_short)
    plt.plot(ema_long)
    plt.plot(df_prices_norm, linestyle = "--")
    plt.title('MACD VS. Normalized Price')
    plt.xlabel('Date')
    plt.legend(labels=['Short term EMA', 'Long Term EMA', 'Norm Price'], loc='best')
    plt.savefig('Figure6')











if __name__ == "__main__":
    test_code()

