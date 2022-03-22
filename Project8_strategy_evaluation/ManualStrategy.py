import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
from marketsimcode import compute_portvals
from indicators import priceSMA, bollinger, momentum

def author():
    return 'chuang405'

class ManualStrategy(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def benchmark_trade(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 31), sv=100000):
        df_price = get_data([symbol], pd.date_range(sd, ed))
        df_price.fillna(method="ffill", inplace=True)
        df_price.fillna(method="bfill", inplace=False)
        dates = [df_price.index[0], df_price.index[-1]]
        df_trades = pd.DataFrame(index=dates, columns=['Symbol', 'Order', 'Shares'])
        df_trades["Symbol"] = [symbol, symbol]
        df_trades['Order'] = ['BUY', 'SELL']
        df_trades['Shares'] = [1000, 1000]
        return df_trades

    def testPolicy(self, symbol='JPM', sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 31), sv=100000):
        df_price = get_data([symbol], pd.date_range(sd, ed))
        df_price.fillna(method="ffill", inplace=True)
        df_price.fillna(method="bfill", inplace=False)
        df_price_stock = df_price[symbol]
        date = []
        orders = []
        shares = []
        get_momentum = momentum(df_price_stock)
        get_SMA = priceSMA(df_price_stock)
        sell, buy = bollinger(df_price_stock)
        bollinger_percentage = (df_price_stock - buy)/(sell - buy)
        holding = 0
        for i in range(df_price.shape[0]-1):
            sign = 0
            if(bollinger_percentage[i] < 0.2 and get_momentum[i] > 0) or ( get_SMA[i] < 0.97 and get_momentum[i] > 0):
                sign = 1
            if(bollinger_percentage[i] > 0.8 and get_momentum[i] < 0) or ( get_SMA[i] > 1.03 and get_momentum[i] < 0):
                sign = -1
            if sign == 0:
                date.append(df_price.index[i])
                orders.append('NA')
                shares.append(0)
            if sign == 1:
                if holding == 0:
                    date.append(df_price.index[i])
                    orders.append('BUY')
                    shares.append(1000)
                    holding = 1000
                elif holding == -1000:
                    date.append(df_price.index[i])
                    orders.append('BUY')
                    shares.append(2000)
                    holding = 1000
                elif holding == 1000:
                    date.append(df_price.index[i])
                    orders.append('NA')
                    shares.append(0)
                    holding = 1000

            if sign == -1:
                if holding == 0:
                    date.append(df_price.index[i])
                    orders.append('SELL')
                    shares.append(1000)
                    holding = -1000
                elif holding == 1000:
                    date.append(df_price.index[i])
                    orders.append('SELL')
                    shares.append(2000)
                    holding = -1000
                elif holding == -1000:
                    date.append(df_price.index[i])
                    orders.append('NA')
                    shares.append(0)
                    holding = -1000

        df_trades = pd.DataFrame(index=date, columns=['Symbol', 'Order', 'Shares'])
        df_trades["Symbol"] = symbol
        df_trades['Order'] = orders
        df_trades['Shares'] = shares
        return df_trades

def manual_strategy():

    # in sample evaluation
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    ms = ManualStrategy()
    orders_benchmark = ms.benchmark_trade(symbol, sd, ed, sv)
    benchmark_port_vals = compute_portvals(orders_benchmark, start_val=100000)
    benchmark_port_vals_norm = benchmark_port_vals / benchmark_port_vals.iloc[0]

    orders_manual = ms.testPolicy(symbol, sd, ed, sv)
    manual_port_vals = compute_portvals(orders_manual, start_val=100000)
    manual_port_vals_norm = manual_port_vals / manual_port_vals.iloc[0]

    orders_manual.to_csv('orders_manual_in_sample.csv')

    # calculate daily returns
    daily_return_benchmark = benchmark_port_vals_norm[1:] / benchmark_port_vals_norm[:-1].values - 1
    daily_return_manual= manual_port_vals_norm[1:] / manual_port_vals_norm[:-1].values - 1

    # calculate cumulative returns
    cum_return_benchmark = (benchmark_port_vals_norm.ix[-1, 0] / benchmark_port_vals_norm.ix[0, 0]) - 1
    cum_return_manual = (manual_port_vals_norm.ix[-1, 0] / manual_port_vals_norm.ix[0, 0]) - 1

    # mean daily return
    daily_mean_benchmark = daily_return_benchmark.mean()
    daily_mean_manual = daily_return_manual.mean()

    # std daily return
    daily_std_benchmark = daily_return_benchmark.std()
    daily_std_manual = daily_return_manual.std()

    print('benchmark cum return: ', cum_return_benchmark)
    print('manual cum return: ', cum_return_manual)
    print('std daily return benchmark: ', daily_std_benchmark)
    print('std daily return manual: ', daily_std_manual)
    print('mean daily return benchmark: ', daily_mean_benchmark)
    print('mean daily return manual: ', daily_mean_manual)
    plt.plot(benchmark_port_vals_norm, color='green')
    plt.plot(manual_port_vals_norm, color = 'red')
    for i in range(0,orders_manual.shape[0]):
        if orders_manual['Order'].iloc[i] == 'SELL':
            plt.axvline(orders_manual.iloc[i].name, color='black')
        elif orders_manual['Order'].iloc[i] == 'BUY':
            plt.axvline(orders_manual.iloc[i].name, color='blue')
    plt.title('Port_Val Benchmark VS. Manual strategy_in sample')
    plt.xlabel('Date')
    plt.ylabel('Norm Port_Val')
    plt.legend(labels=['Benchmark', 'Manual'], loc='best')
    plt.savefig('figure1')
    plt.cla()
    # out of sample evaluation
    symbol = 'JPM'
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000
    orders_benchmark = ms.benchmark_trade(symbol, sd, ed, sv)
    benchmark_port_vals = compute_portvals(orders_benchmark, start_val=100000)
    benchmark_port_vals_norm = benchmark_port_vals / benchmark_port_vals.iloc[0]
    #print('benchmark return: ', benchmark_port_vals_norm.iloc[-1])

    orders_manual = ms.testPolicy(symbol, sd, ed, sv)
    manual_port_vals = compute_portvals(orders_manual, start_val=100000)
    manual_port_vals_norm = manual_port_vals / manual_port_vals.iloc[0]
    #print('manual return: ', manual_port_vals_norm.iloc[-1])

    orders_manual.to_csv('orders_manual_out_of_sample.csv')

    # calculate daily returns
    daily_return_benchmark = benchmark_port_vals_norm[1:] / benchmark_port_vals_norm[:-1].values - 1
    daily_return_manual = manual_port_vals_norm[1:] / manual_port_vals_norm[:-1].values - 1

    # calculate cumulative returns
    cum_return_benchmark = (benchmark_port_vals_norm.ix[-1, 0] / benchmark_port_vals_norm.ix[0, 0]) - 1
    cum_return_manual = (manual_port_vals_norm.ix[-1, 0] / manual_port_vals_norm.ix[0, 0]) - 1

    # mean daily return
    daily_mean_benchmark = daily_return_benchmark.mean()
    daily_mean_manual = daily_return_manual.mean()

    # std daily return
    daily_std_benchmark = daily_return_benchmark.std()
    daily_std_manual = daily_return_manual.std()




    print('benchmark cum return: ', cum_return_benchmark)
    print('manual cum return: ', cum_return_manual)
    print('std daily return benchmark: ', daily_std_benchmark)
    print('std daily return manual: ', daily_std_manual)
    print('mean daily return benchmark: ', daily_mean_benchmark)
    print('mean daily return manual: ', daily_mean_manual)
    plt.plot(benchmark_port_vals_norm, color='green')
    plt.plot(manual_port_vals_norm, color='red')
    for i in range(0, orders_manual.shape[0]):
        if orders_manual['Order'].iloc[i] == 'SELL':
            plt.axvline(orders_manual.iloc[i].name, color='black')
        elif orders_manual['Order'].iloc[i] == 'BUY':
            plt.axvline(orders_manual.iloc[i].name, color='blue')
    plt.title('Port_Val Benchmark VS. Manual strategy Out of Sample')
    plt.xlabel('Date')
    plt.ylabel('Norm Port_Val')
    plt.legend(labels=['Benchmark', 'Manual'], loc='best')

    plt.savefig('figure2')
    

    plt.plot(manual_port_vals_norm, color='red')
    plt.show()

if __name__ == "__main__":
    manual_strategy()