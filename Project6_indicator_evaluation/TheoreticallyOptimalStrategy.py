import datetime as dt
import os
import numpy as np
import pandas as pd
from util import get_data, plot_data
import matplotlib.pyplot as plt

from marketsimcode import compute_portvals


def author():
    return "chuang405"

def benchmark_trade(symbol = 'AAPL', sd = dt.datetime(2010, 1, 1), ed = dt.datetime(2011, 12, 31), sv = 100000):
    df_price = get_data([symbol], pd.date_range(sd,ed))
    df_price.fillna(method="ffill", inplace=True)
    df_price.fillna(method="bfill", inplace=False)
    dates = [df_price.index[0], df_price.index[-1]]
    df_trades = pd.DataFrame(index=dates, columns=['Symbol', 'Order', 'Shares'])
    df_trades["Symbol"] = [symbol, symbol]
    df_trades['Order'] = ['BUY', 'SELL']
    df_trades['Shares'] = [1000, 1000]
    return df_trades

def testPolicy(symbol = 'AAPL', sd = dt.datetime(2010, 1, 1), ed = dt.datetime(2011, 12, 31), sv = 100000):
    df_price = get_data([symbol], pd.date_range(sd, ed))
    df_price.fillna(method="ffill", inplace=True)
    df_price.fillna(method="bfill", inplace=False)
    date_range = pd.date_range(df_price.index[0], df_price.index[-1])
    date = []
    orders = []
    shares = []
    holding = 0
    for i in range(df_price.shape[0] - 1):
        current_price = df_price.iloc[i, 1]
        next_price = df_price.iloc[i + 1, 1]

        if holding == 0:
            if current_price - next_price > 0:
                date.append(df_price.index[i])
                orders.append('SELL')
                shares.append(1000)
                holding = -1000
            else:
                date.append(df_price.index[i])
                orders.append('BUY')
                shares.append(1000)
                holding = 1000

        elif holding == 1000:
            if current_price - next_price > 0:
                date.append(df_price.index[i])
                orders.append('SELL')
                shares.append(2000)
                holding = -1000
            else:
                date.append(df_price.index[i])
                orders.append('BUY')
                shares.append(0)
                holding = 1000

        elif holding == -1000:
            if current_price - next_price > 0:
                date.append(df_price.index[i])
                orders.append('BUY')
                shares.append(0)
                holding = -1000

            else:
                date.append(df_price.index[i])
                orders.append('BUY')
                shares.append(2000)
                holding = 1000
    df_trades = pd.DataFrame(index=date, columns=['Symbol', 'Order', 'Shares'])
    df_trades["Symbol"] = symbol
    df_trades['Order'] = orders
    df_trades['Shares'] = shares
    df_trades.drop(df_trades[df_trades['Shares'] == 0].index, inplace=True)
    return df_trades

def test_code():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    orders_benchmark = benchmark_trade(symbol, sd, ed, sv)
    benchmark_port_vals = compute_portvals(orders_benchmark, start_val=100000)
    benchmark_port_vals_norm = benchmark_port_vals/ benchmark_port_vals.iloc[0]



    orders_optimal = testPolicy(symbol, sd, ed, sv)
    optimal_port_vals = compute_portvals(orders_optimal, start_val=100000)
    optimal_port_vals_norm = optimal_port_vals / optimal_port_vals.iloc[0]

    #calculate daily returns
    daily_return_benchmark = benchmark_port_vals_norm[1:] / benchmark_port_vals_norm[:-1].values - 1
    daily_return_optimal = optimal_port_vals_norm[1:] / optimal_port_vals_norm[:-1].values - 1

    #calculate cumulative returns
    cum_return_benchmark = (benchmark_port_vals_norm.ix[-1, 0]/benchmark_port_vals_norm.ix[0,0]) - 1
    cum_return_optimal = (optimal_port_vals_norm.ix[-1,0]/optimal_port_vals_norm.ix[0,0]) -1

    #mean daily return
    daily_mean_benchmark = daily_return_benchmark.mean()
    daily_mean_optimal = daily_return_optimal.mean()

    #std daily return
    daily_std_benchmark = daily_return_benchmark.std()
    daily_std_optimal = daily_return_optimal.std()


    plt.figure(1)

    plt.plot(benchmark_port_vals_norm, color = 'green')
    plt.plot(optimal_port_vals_norm, color= 'red')
    plt.title('Port_Val Benchmark VS. Optimal')
    plt.xlabel('Date')
    plt.ylabel('Norm Port_Val')
    plt.legend(labels=['Benchmark', 'Optimal'], loc='best')
    #plt.show()
    plt.savefig('Figure1')

    print('benchmark cumulative return: ', cum_return_benchmark)
    print('Optimal cumulative return: ', cum_return_optimal)
    print('std daily return benchmark: ', daily_std_benchmark)
    print('std daily return optimal: ', daily_std_optimal)
    print('mean daily return benchmark: ', daily_mean_benchmark)
    print('mean daily return optimal: ', daily_mean_optimal)






if __name__ == "__main__":
    test_code()












