# -*- coding: utf-8 -*-
"""
Created on Fri Apr 5 22:31:41 2024

@author: amudek
"""

import yfinance

#load in a stock/coin with the Ticker class
btc = yfinance.Ticker('BTC-USD')

#can pull summary info
btc.basic_info
btc.fast_info
btc.info

#can view related news
btc.news

#most importantly, can pull full price histories
full_btc_dat = btc.history('max')
btc_dat_by_dates = btc.history(start = '2019-01-01', end = '2023-12-31', interval = '1d')
btc_dat_date_to_now = btc.history(start = '2010-01-01', end = None, interval = '1mo')
#NOTE: btc_dat_date_to_now has a start date before the BTC data begins. It will automatically
#      start at the earliest data. Setting end to None will provide all data it has up
#      to now.

'''
Unit options for the interval argument:
    
    m  - minute
    h  - hour
    d  - day
    wk - week
    mo - month
    y  - year

NOTE: For some of the indicators, you cannot go any coarser than 1 day step
      sizes. Days are also the standard unit for most of the indicator so going
      with '1d' intervals can be convenient.

'''

#after calling the history function, you can also pull metadata
btc.history_metadata