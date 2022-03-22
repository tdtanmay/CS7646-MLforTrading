import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from util import get_data, plot_data
from marketsimcode import compute_portvals
from indicators import priceSMA, bollinger, momentum
from ManualStrategy import *
from StrategyLearner import *
def author():
    return 'chuang405'
def e1():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    sl = StrategyLearner()
    ms = ManualStrategy()
    # bench mark orders
    orders_benchmark = sl.benchmark_trade(symbol, sd, ed, sv)
    benchmark_port_vals = compute_portvals(orders_benchmark, start_val=100000)
    benchmark_port_vals_norm = benchmark_port_vals / benchmark_port_vals.iloc[0]
    print('benchmark cum return:', benchmark_port_vals_norm.iloc[-1])
    # manual orders
    orders_manual = ms.testPolicy(symbol, sd, ed, sv)
    manual_port_vals = compute_portvals(orders_manual, start_val=100000)
    manual_port_vals_norm = manual_port_vals / manual_port_vals.iloc[0]
    orders_manual.to_csv('orders_manual.csv')
    print('manual strategy cum return:', manual_port_vals_norm.iloc[-1])
    #strategy learner orders
    learner = StrategyLearner(impact=0)
    learner.add_evidence(symbol, sd, ed, sv)
    q = learner.testPolicy(symbol, sd, ed, sv)
    q_value = q.values

    date = []
    orders = []
    shares = []
    for i in range(0, q.shape[0] - 1):
        if (q_value[i] == 1000):
            date.append(q.index[i])
            orders.append('BUY')
            shares.append(1000)
        if (q_value[i] == -1000):
            date.append(q.index[i])
            orders.append('SELL')
            shares.append(1000)
        if (q_value[i] == 2000):
            date.append(q.index[i])
            orders.append('BUY')
            shares.append(2000)
        if (q_value[i] == -2000):
            date.append(q.index[i])
            orders.append('SELL')
            shares.append(2000)
    orders_learner = pd.DataFrame(index=date, columns=['Symbol', 'Order', 'Shares'])
    orders_learner["Symbol"] = symbol
    orders_learner['Order'] = orders
    orders_learner['Shares'] = shares

    learner_port_vals = compute_portvals(orders_learner, start_val=100000)
    learner_port_vals_norm = learner_port_vals / learner_port_vals.iloc[0]
    print('learner cum return: ', learner_port_vals_norm.iloc[-1])


    # calculate daily returns
    daily_return_benchmark = benchmark_port_vals_norm[1:] / benchmark_port_vals_norm[:-1].values - 1
    daily_return_manual = manual_port_vals_norm[1:] / manual_port_vals_norm[:-1].values - 1
    daily_return_learner = learner_port_vals_norm[1:]/ learner_port_vals_norm[:-1].values - 1

    # calculate cumulative returns
    cum_return_benchmark = (benchmark_port_vals_norm.ix[-1, 0] / benchmark_port_vals_norm.ix[0, 0]) - 1
    cum_return_manual = (manual_port_vals_norm.ix[-1, 0] / manual_port_vals_norm.ix[0, 0]) - 1
    cum_return_learner = (learner_port_vals_norm.ix[-1, 0] / learner_port_vals_norm.ix[0, 0]) - 1

    # mean daily return
    daily_mean_benchmark = daily_return_benchmark.mean()
    daily_mean_manual = daily_return_manual.mean()
    daily_mean_learner = daily_return_learner.mean()

    # std daily return
    daily_std_benchmark = daily_return_benchmark.std()
    daily_std_manual = daily_return_manual.std()
    daily_std_learner = daily_return_learner.std()

    #plotting
    plt.plot(benchmark_port_vals_norm, color='green')
    plt.plot(manual_port_vals_norm, color='red')
    plt.plot(learner_port_vals_norm, color='blue')
    plt.title('Experiment 1')
    plt.xlabel('Date')
    plt.ylabel('Norm Port_Val')
    plt.legend(labels=['Benchmark', 'Manual', 'Learner'], loc='best')
    plt.savefig('Experiment 1')
    print('daily mean return benchmark:', daily_mean_benchmark)
    print('daily mean return manual:', daily_mean_manual)
    print('daily mean return learner:', daily_mean_learner)
    print('daily std return benchmark:', daily_std_benchmark)
    print('daily std return manual:', daily_std_manual)
    print('daily std return learner:', daily_std_learner)




if __name__ == '__main__':
    pass