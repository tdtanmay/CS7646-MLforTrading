""""""
"""MC2-P1: Market simulator.  		  	   		   	 		  		  		    	 		 		   		 		  

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
import os

import numpy as np

import pandas as pd
from util import get_data, plot_data


def author():
    return "chuang405"


def compute_portvals(
        orders,
        start_val=1000000,
        commission=0,
        impact=0,
):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Computes the portfolio values.  		  	   		   	 		  		  		    	 		 		   		 		  

    :param orders_file: Path of the order file or the file object  		  	   		   	 		  		  		    	 		 		   		 		  
    :type orders_file: str or file object  		  	   		   	 		  		  		    	 		 		   		 		  
    :param start_val: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
    :type start_val: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here
    df_orders = orders
    # create date range
    date_range = pd.date_range(df_orders.index[0], df_orders.index[-1])
    # create symbol list
    df_symbols = df_orders['Symbol']
    temp_list = list(df_symbols)
    list_symbols = []
    for i in temp_list:
        if i not in list_symbols:
            list_symbols.append(i)
    # construct Prices data frame
    df_prices = get_data(list_symbols, date_range, addSPY=True, colname='Adj Close')
    df_prices['Cash'] = 1.0
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=False)

    # construct Trades data frame
    df_trades = df_prices.copy()
    df_trades[list_symbols] = 0
    df_trades['Cash'] = 0
    df_trades['SPY'] = 0

    for index, row in df_orders.iterrows():
        if index not in df_prices.index:
            continue
        date = index
        symbol = row['Symbol']
        order = row['Order']
        share = row['Shares']
        price = df_prices.ix[date][symbol]

        if order == 'SELL':
            df_trades.ix[date, 'Cash'] -= commission
            df_trades.ix[date, 'Cash'] += share * price * (1 - impact)
            df_trades.ix[date, symbol] -= share

        if order == 'BUY':
            df_trades.ix[date, 'Cash'] -= commission
            df_trades.ix[date, 'Cash'] -= share * price * (1 + impact)
            df_trades.ix[date, symbol] += share

    # construct holdings data frame
    df_holdings = df_trades.copy()
    df_holdings.ix[0, 'Cash'] = df_holdings.ix[0, 'Cash'] + start_val
    for i in range(1, len(df_holdings)):
        df_holdings.ix[i] += df_holdings.ix[i - 1]

    # construct Values data frame
    portvals = pd.DataFrame(index=df_holdings.index)
    portvals['Total Value'] = 0
    for index, row in df_holdings.iterrows():
        for symbol in df_holdings.columns:
            portvals.ix[index, 'Total Value'] += row[symbol] * df_prices.ix[index, symbol]
    return portvals
