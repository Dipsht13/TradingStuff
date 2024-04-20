# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:59:31 2024

@author: amudek
"""

import pandas as pd
import _pickle as pkl
import FinancialDataRetriever as fdr

import time
tic = time.time()


tickers_of_interest = ['^SPX', 'BTC-USD']
# tickers_of_interest = ['ALGT', 'AMZN', 'AMGN', 'AAPL', 'BA', 'CCL', 'CI', 'DAL', 
#                        'DIS', 'XOM', 'GOOG', 'GTBIF', 'JPM', 'LMT', 'MCD', 'MSFT', 
#                        'MS', 'NET', 'PEP', 'RCL', 'LUV', 'TSNDF', 'TCNNF', 'WMT', 
#                        'WBD', 'WFC', 'WTFC', 'YUM']

sma_periods = [5, 20, 100]
ema_periods = [5, 20]
atr_periods = [14]

save_data = True
save_folder = 'saved_data/'


ticker_data = fdr.RetrieveData(tickers_of_interest, 
                               start_date = '2015-01-01',
                               yrs_to_keep = 5, 
                               sma_periods = sma_periods, 
                               ema_periods = ema_periods, 
                               atr_periods = atr_periods)


if save_data:
    
    writer = pd.ExcelWriter(save_folder + 'collected_tickers.xlsx')
    
    for ticker in tickers_of_interest:
        ticker_data[ticker].to_excel(writer, sheet_name = ticker)
    
    writer.close()
    pkl.dump(ticker_data, open(save_folder + 'collected_tickers.p', 'wb'))
    
    
print('Run time: ' + str(round((time.time() - tic)/60, 2)) + ' minutes.')