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
def e2():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    sl = StrategyLearner()
    impacts = [0, 0.05, 0.1]
    port_values = []
    q_list = []
    for i in impacts:
        learner = StrategyLearner(impact=i)
        learner.add_evidence(symbol, sd, ed, sv)
        q = learner.testPolicy(symbol, sd, ed, sv)
        q_value = q.values
        q_list.append(q_value)
        date = []
        orders = []
        shares = []
        for i in range(0, q.shape[0]):
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
        port_values.append(learner_port_vals)

    low_port_vals = port_values[0]
    medium_port_vals = port_values[1]
    high_port_vals = port_values[2]
    low_port_vals_norm = low_port_vals/low_port_vals.iloc[0]
    medium_port_vals_norm = medium_port_vals / medium_port_vals.iloc[0]
    high_port_vals_norm = high_port_vals / high_port_vals.iloc[0]


    # calculate daily returns
    daily_return_low = low_port_vals_norm[1:] / low_port_vals_norm[:-1].values - 1
    daily_return_medium = medium_port_vals_norm[1:] / medium_port_vals_norm[:-1].values - 1
    daily_return_high = high_port_vals_norm[1:] / high_port_vals_norm[:-1].values - 1

    # calculate cumulative returns
    cum_return_low = (low_port_vals_norm.ix[-1, 0] / low_port_vals_norm.ix[0, 0]) - 1
    cum_return_medium = (medium_port_vals_norm.ix[-1, 0] / medium_port_vals_norm.ix[0, 0]) - 1
    cum_return_high = (high_port_vals_norm.ix[-1, 0] / high_port_vals_norm.ix[0, 0]) - 1

    # mean daily return
    daily_mean_low = daily_return_low.mean()
    daily_mean_medium = daily_return_medium.mean()
    daily_mean_high = daily_return_high.mean()

    # std daily return
    daily_std_low = daily_return_low.mean()
    daily_std_medium = daily_return_medium.mean()
    daily_std_high = daily_return_high.mean()


    print('cum return low:', cum_return_low)
    print('cum return medium:', cum_return_medium)
    print('cum return high:', cum_return_high)
    print('daily mean return low:', daily_mean_low)
    print('daily mean return medium:', daily_mean_medium)
    print('daily mean return high:', daily_mean_high)
    print('daily std return low:', daily_std_low)
    print('daily std return medium:', daily_std_medium)
    print('daily std return high:', daily_std_high)


    print('num of trading, low', len(q_list[0]))
    print('num of trading, medium', len(q_list[1]))
    print('num of trading, high', len(q_list[2]))

    plt.plot(low_port_vals_norm, color='green')
    plt.plot(medium_port_vals_norm, color='red')
    plt.plot(high_port_vals_norm, color='blue')
    plt.title('Experiment 2')
    plt.xlabel('Date')
    plt.ylabel('Norm Port_Val')
    plt.legend(labels=['impact = 0', 'impact = 0.05', 'impact = 0.1'], loc='best')
    plt.savefig('Experiment 2')


if __name__=="__main__":
    pass
