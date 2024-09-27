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


tickers_of_interest = ['^SPX', 'NVDA', 'BTC-USD', 'ETH-USD']
tickers_from_dad = ['ALGT', 'AAPL', 'LMT', 'PEP', 'YUM', 'MSFT',
                    'JPM', 'WMT', 'WFC', 'AMZN', 'CCL']
tickers_from_joey = ['NOC', 'RTX', 'BA']
tickers_from_kyle = ['AMD', 'INTC', 'META', 'GOOG', 'CEG', 'VST', 'MOD', 'LII', 'CARR']

tickers_of_interest = tickers_of_interest + tickers_from_dad + tickers_from_joey + tickers_from_kyle
tickers_of_interest = list(pd.Series(tickers_of_interest).unique()) # remove any duplicates
# tickers_of_interest = ['^SPX', 'NVDA', 'BTC-USD', 'ETH-USD']
# tickers_of_interest = ['ALGT', 'AMZN', 'AMGN', 'AAPL', 'BA', 'CCL', 'CI', 'DAL', 
#                        'DIS', 'XOM', 'GOOG', 'GTBIF', 'JPM', 'LMT', 'MCD', 'MSFT', 
#                        'MS', 'NET', 'PEP', 'RCL', 'LUV', 'TSNDF', 'TCNNF', 'WMT', 
#                        'WBD', 'WFC', 'WTFC', 'YUM']

sma_periods = [20, 50, 200]
ema_periods = [5, 20]
atr_periods = [14]

save_data = False
save_folder = 'saved_data/'


ticker_data = fdr.RetrieveData(tickers_of_interest, 
                               start_date = '2015-01-01',
                               yrs_to_keep = 5, 
                               sma_periods = sma_periods, 
                               ema_periods = ema_periods, 
                               atr_periods = atr_periods)


if save_data:
    
    print('\nExporting data:\n')
    print('Saving Excel file', end = '..')
    writer = pd.ExcelWriter(save_folder + 'collected_tickers.xlsx')
    
    for ticker in tickers_of_interest:
        ticker_data[ticker].to_excel(writer, sheet_name = ticker)
    
    writer.close()
    print('Done.')
    print('Saving Pickle file', end = '..')
    pkl.dump(ticker_data, open(save_folder + 'collected_tickers.p', 'wb'))
    print('Done.\n')
    

run_time = round((time.time() - tic) / 60, 2)
time_per_tkr = round(run_time * 60 / len(tickers_of_interest), 2)

report_str = '\nRun time: ' + str(run_time) + ' minutes'
report_str = report_str + '(~' + str(time_per_tkr) + ' sec/ticker).'
print(report_str)