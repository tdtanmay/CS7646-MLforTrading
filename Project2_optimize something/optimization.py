""""""
"""MC1-P2: Optimize a portfolio.  		  	   		   	 		  		  		    	 		 		   		 		  

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

Student Name: Chengqi Huang (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: chaung405 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 903534690 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from util import get_data, plot_data
import scipy.optimize as spo


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality

# filling missing values
def fill_missing_values(df_data):
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=True)
    return df_data


# normallize the stock prices
def normalize_data(df):
    return df / df.ix[0, :]


# filling missing values
def fill_missing_values(df_data):
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=True)
    return df_data


# normallize the stock prices
def normalize_data(df):
    return df / df.ix[0, :]


# calculate combined fortfolio value through the time frame
def combined_values(price_df, alloc):
    price_df = price_df * alloc
    combined_values = price_df.sum(axis=1)
    return combined_values


# calculate cumulative return
def cr(combined_values):
    return (combined_values[-1] / combined_values[0]) - 1


# calculate average daily return
def adr(combined_values):
    daily_returns = combined_values[1:] / combined_values[:-1].values - 1
    adr = daily_returns.mean()
    return adr


# calculate standard deviation of daily return
def sddr(combined_values):
    daily_returns = combined_values[1:] / combined_values[:-1].values - 1
    sddr = daily_returns.std()
    return sddr


# calculate sharpe ratio
def sr(combined_values):
    sr = 15.8745 * adr(combined_values) / sddr(combined_values)
    return sr


# objective function minimize sharp ratio
def objective_min_sr(price_df, allocs):
    port_values = combined_values(price_df, allocs)
    sharpratio = sr(port_values)
    return -sharpratio


def optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 6, 1),
        syms=["IBM", "X", "GLD", "JPM"],
        gen_plot=False,
):
    """
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and
    statistics.

    :param sd: A datetime object that represents the start date, defaults to 1/1/2008
    :type sd: datetime
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009
    :type ed: datetime
    :param syms: A list of symbols that make up the portfolio (note that your code should support any
        symbol in the data directory)
    :type syms: list
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your
        code with gen_plot = False.
    :type gen_plot: bool
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,
        standard deviation of daily returns, and Sharpe ratio
    :rtype: tuple
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all = fill_missing_values(prices_all)
    prices_all = normalize_data(prices_all)

    prices_portfolio = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    num_stocks = len(syms)

    init_allocs = [1.0 / num_stocks] * num_stocks

    init_allocs = np.array(init_allocs)
    # print(init_allocs)

    # combined_values = combined_values(prices_portfolio, init_allocs)
    bounds = [(0, 1)] * num_stocks

    # find the allocations for the optimal portfolio
    # note that the values here ARE NOT meant to be correct for a test case
    constraint_1 = ({'type': 'eq', 'fun': lambda initial_allocs: 1.0 - np.sum(initial_allocs)})

    obj_fun = spo.minimize(objective_min_sr, init_allocs, args=(prices_portfolio), method='SLSQP',
                           bounds=bounds, constraints=constraint_1)

    allocs_result = obj_fun.x

    combined_values_result = combined_values(prices_portfolio, allocs_result)
    cr_result = cr(combined_values_result)
    sr_result = - obj_fun.fun
    adr_result = adr(combined_values_result)
    sddr_result = sddr(combined_values_result)

    # Get daily portfolio value
    port_val = combined_values_result  # add code here to compute daily portfolio values

    # Compare daily portfolio value with SPY using a normalized plot

    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY], keys=["Portfolio", "SPY"], axis=1)
        df_temp.plot()
        plt.xlabel('Date')
        plt.ylabel('Price')

        plt.title('Daily Portfolio Value and SPY')
        #plt.show()
        plt.savefig('figure1')
        pass

    return allocs_result, cr_result, adr_result, sddr_result, sr_result


def test_code():
    """
    This function WILL NOT be called by the auto grader.
    """

    start_date = dt.datetime(2009, 1, 1)
    end_date = dt.datetime(2010, 1, 1)
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM"]

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(
        sd=start_date, ed=end_date, syms=symbols, gen_plot=False
    )

    # Print statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")

def print_chart():
    optimize_portfolio(
        sd=dt.datetime(2008, 1, 1),
        ed=dt.datetime(2009, 6, 1),
        syms=["IBM", "X", "GLD", "JPM"],
        gen_plot=True,
    )



if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called
    test_code()
    print_chart()
