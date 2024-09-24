# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 23:48:22 2024

@author: amude
"""

import time
from CreateLSTM import *


start = time.time()


use_past_n_for_lstm = 10 # 2-weeks inform each prediction
ndays_to_forecast = [1, 2, 3, 4, 5] # up to a 1-week forecast

tickers_of_interest = ['^SPX', 'NVDA', 'BTC-USD', 'ETH-USD']
tickers_from_dad = ['ALGT', 'AAPL', 'LMT', 'PEP', 'YUYM', 'MSFT',
                    'JPM', 'WMT', 'WFC', 'AMZN']

tickers_of_interest = tickers_of_interest + tickers_from_dad

success_history = {}
print('Creating models for the following tickers:\n')
for ticker in tickers_of_interest:
    success_history[ticker] = {}
    print(ticker)
    
    for ndays in ndays_to_forecast:
        success_rate = CreateLSTM(ticker, use_past_n_for_lstm, ndays)
        
        success_history[ticker][ndays] = success_rate
        print(str(ndays) + '-day: ' + str(round(success_rate * 100, 2)) + '%')
        

elapsed_time = time.time() - start
if elapsed_time > 3600:
    print('Run Time: ' + str(round((time.time() - start)/3600, 2)) + ' hours.')
else:
    print('Run Time: ' + str(round((time.time() - start)/60, 2)) + ' minutes.')
    