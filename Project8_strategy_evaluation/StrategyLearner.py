""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		   	 		  		  		    	 		 		   		 		  
import random  		  	   		   	 		  		  		    	 		 		   		 		  
import RTLearner as rt
import pandas as pd
import util as ut
from util import get_data, plot_data
import pandas as pd
import BagLearner as bl
from indicators import priceSMA, bollinger, momentum
import math
from marketsimcode import compute_portvals
import numpy as np
import matplotlib.pyplot as plt
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
class StrategyLearner(object):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # constructor  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        self.verbose = verbose  		  	   		   	 		  		  		    	 		 		   		 		  
        self.impact = impact  		  	   		   	 		  		  		    	 		 		   		 		  
        self.commission = commission
        self.learner = bl.BagLearner(kwargs = {'leaf_size': 5}, bags = 15, boost = False, verbose = False)

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

    def author(self):
        return 'chuang405'



  		  	   		   	 		  		  		    	 		 		   		 		  
    # this method should create a QLearner, and train it for trading  		  	   		   	 		  		  		    	 		 		   		 		  
    def add_evidence(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="JPM",
        sd=dt.datetime(2008, 1, 1),  		  	   		   	 		  		  		    	 		 		   		 		  
        ed=dt.datetime(2009, 12, 31),
        sv=100000,
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  

        # add your code to do learning here
        days = 5
        df_price = get_data([symbol], pd.date_range(sd, ed))
        df_price.fillna(method="ffill", inplace=True)
        df_price.fillna(method="bfill", inplace=False)
        df_price_stock = df_price[symbol]
        get_momentum = momentum(df_price_stock)
        get_SMA = priceSMA(df_price_stock)
        sell, buy = bollinger(df_price_stock)
        bollinger_percentage = (df_price_stock - buy) / (sell - buy)
        training_x = pd.concat((get_momentum, get_SMA, bollinger_percentage), axis = 1)
        training_x.fillna(0, inplace = True)
        training_x.columns = ['Momentum','SMA','Bollinger Percentage']
        data_x = training_x[:-days].values

        data_y = np.zeros(data_x.shape[0])
        buy = 0.018 + self.impact
        sell = -0.018 - self.impact
        return_days = df_price_stock.values[days:]/df_price_stock.values[:-days] - 1
        for i in range(0, data_x.shape[0]):
            if return_days[i] > buy:
                data_y[i] = 1
            elif return_days[i] < sell:
                data_y[i] = -1
            else:
                data_y[i] = 0
        self.learner.add_evidence(data_x, data_y)

  		  	   		   	 		  		  		    	 		 		   		 		  
    # this method should use the existing policy and test it against new data  		  	   		   	 		  		  		    	 		 		   		 		  
    def testPolicy(  		  	   		   	 		  		  		    	 		 		   		 		  
        self,  		  	   		   	 		  		  		    	 		 		   		 		  
        symbol="JPM",
        sd=dt.datetime(2010, 1, 1),
        ed=dt.datetime(2011, 12, 31),
        sv=100000,
    ):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # here we build a fake set of trades  		  	   		   	 		  		  		    	 		 		   		 		  
        # your code should return the same sort of data
        df_price = get_data([symbol], pd.date_range(sd, ed))
        df_price.fillna(method="ffill", inplace=True)
        df_price.fillna(method="bfill", inplace=False)
        df_price_stock = df_price[symbol]
        get_momentum = momentum(df_price_stock)
        get_SMA = priceSMA(df_price_stock)
        sell, buy = bollinger(df_price_stock)
        bollinger_percentage = (df_price_stock - buy) / (sell - buy)
        test_x = pd.concat((get_momentum, get_SMA, bollinger_percentage), axis = 1)
        test_x.fillna(0, inplace = True)
        test_x = test_x.values
        test_y = self.learner.query(test_x)
        date = []
        orders = []
        shares = []
        holding = 0
        for i in range(test_y.shape[0] - 2):
            sign = 0
            if test_y[i] > 0:
                sign = 1
            elif test_y[i] < 0:
                sign = -1
            if sign == 0:
                date.append(df_price.index[i])
                orders.append('NA')
                shares.append(0)
            elif sign == 1:
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

            elif sign == -1:
                if holding == 0:
                    date.append(df_price.index[i])
                    orders.append('SELL')
                    shares.append(-1000)
                    holding = -1000
                elif holding == 1000:
                    date.append(df_price.index[i])
                    orders.append('SELL')
                    shares.append(-2000)
                    holding = -1000
                elif holding == -1000:
                    date.append(df_price.index[i])
                    orders.append('NA')
                    shares.append(0)
                    holding = -1000
        if (holding == -1000):
            shares.append(1000)
        elif (holding == 1000):
            shares.append(-1000)
        date.append(df_price.index[-1])
        df_trades = pd.DataFrame(index=date, columns=['Shares'])
        df_trades['Shares'] = shares
        df_trades.drop(df_trades[df_trades['Shares'] == 0].index, inplace=True)



        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(type(df_price_stock))  # it better be a DataFrame!
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(df_price_stock)
        if self.verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(df_price)
        return df_trades

def strategy_learner():

    # in sample evaluation
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    sl = StrategyLearner()
    orders_benchmark = sl.benchmark_trade(symbol, sd, ed, sv)
    benchmark_port_vals = compute_portvals(orders_benchmark, start_val=100000)
    benchmark_port_vals_norm = benchmark_port_vals / benchmark_port_vals.iloc[0]


    learner = StrategyLearner(impact = 0.005)
    learner.add_evidence(symbol, sd, ed, sv)
    q = learner.testPolicy(symbol, sd, ed, sv)
    q_value = q.values

    date = []
    orders = []
    shares = []
    for i in range(0, q.shape[0]):
        if(q_value[i] == 1000):
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
    orders_learner = pd.DataFrame(index = date, columns=['Symbol', 'Order', 'Shares'])
    orders_learner["Symbol"] = symbol
    orders_learner['Order'] = orders
    orders_learner['Shares'] = shares
    print(orders_learner)

    orders_learner.to_csv('orders_learner_in_sample.csv')

    learner_port_vals = compute_portvals(orders_learner, start_val=100000)

    learner_port_vals_norm = learner_port_vals / learner_port_vals.iloc[0]
    print('benchmark return: ', benchmark_port_vals_norm.iloc[-1])
    print('learner return: ', learner_port_vals_norm.iloc[-1])

    plt.plot(benchmark_port_vals_norm, color='green')
    plt.plot(learner_port_vals_norm, color='red')
    for i in range(0, orders_learner.shape[0]):
        if orders_learner['Order'].iloc[i] == 'SELL':
            plt.axvline(orders_learner.iloc[i].name, color='black')
        elif orders_learner['Order'].iloc[i] == 'BUY':
            plt.axvline(orders_learner.iloc[i].name, color='blue')
    plt.title('Port_Val Benchmark VS. Manual strategy_in sample')
    plt.xlabel('Date')
    plt.ylabel('Norm Port_Val')
    plt.legend(labels=['Benchmark', 'Learner'], loc='best')

    # out of sample evaluation
    symbol = 'JPM'
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    sv = 100000
    sl = StrategyLearner()
    orders_benchmark = sl.benchmark_trade(symbol, sd, ed, sv)
    benchmark_port_vals = compute_portvals(orders_benchmark, start_val=100000)
    benchmark_port_vals_norm = benchmark_port_vals / benchmark_port_vals.iloc[0]

    learner = StrategyLearner(impact=0.005)
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
    orders_learner.to_csv('orders_learner_out_of_sample.csv')
    learner_port_vals = compute_portvals(orders_learner, start_val=100000)
    learner_port_vals_norm = learner_port_vals / learner_port_vals.iloc[0]
    print('benchmark return: ', benchmark_port_vals_norm.iloc[-1])
    print('learner return: ', learner_port_vals_norm.iloc[-1])

    plt.plot(benchmark_port_vals_norm, color='green')
    plt.plot(learner_port_vals_norm, color='red')
    for i in range(0, orders_learner.shape[0]):
        if orders_learner['Order'].iloc[i] == 'SELL':
            plt.axvline(orders_learner.iloc[i].name, color='black')
        elif orders_learner['Order'].iloc[i] == 'BUY':
            plt.axvline(orders_learner.iloc[i].name, color='blue')
    plt.title('Port_Val Benchmark VS. Manual strategy_in sample')
    plt.xlabel('Date')
    plt.ylabel('Norm Port_Val')
    plt.legend(labels=['Benchmark', 'Learner'], loc='best')
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    pass
