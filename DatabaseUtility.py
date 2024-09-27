# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:01:50 2024

@author: amude
"""

import sqlite3 as sql


def CreateDB(ticker_list = [], start_date = None, end_date = None, 
             output_db = './you_forgot_to_name_me.db'):
    
    import FinancialDataRetriever as fdr
    import time
    
    start = time.time()
    
    print('Creating database containing the following (' + str(len(ticker_list)) + ') tickers:\n')
    con = sql.connect(output_db)
    for ticker in ticker_list:
        
        print(ticker, end = '..')
        dat = fdr.GetTickerDataAbridged(ticker, start_date = start_date, end_date = end_date)
        
        # dat.reset_index(inplace = True)
        # dat['Date'] = dat['Date'].dt.date # only want the date; no time
        dat['Ticker'] = [ticker] * len(dat)
            
        dat.to_sql(name = 'ticker_history', con = con, if_exists = 'append', index = False)
    
    print(str(round((time.time() - start) / 60, 2)) + ' minutes.')
    con.close()
    return
    
    
def UpdateDB(db = None):
    
    import pandas as pd
    import FinancialDataRetriever as fdr
    import time
    
    start = time.time()    
    
    if ~db:
        return 'Forgetting something there Tex?'
    
    print('Updating database ' + db)
    print('Checking database time bounds and ticker list', end = '...')
    con = sql.connect(db)
    
    # first we need the most recent date in the db
    # we also need a list of tickers in the db
    temp = pd.read_sql('SELECT Date, Ticker from ticker_history', con)
    temp['Date'] = pd.to_datetime(temp['Date'])
    # temp['Date'] = temp['Date'].dt.date
    
    ticker_list = list(temp['Ticker'].unique())
    
    earliest_date = temp['Date'].min()
    latest_date = temp['Date'].max()
    
    # even though we're only looking to update the db, want to pull data
    #   for the whole time span due to some of the indicators needing 10's or
    #   hundreds of preceeding rows
    # we need a start date string for fdr functions
    start_date = str(earliest_date)
    
    print('Done.')
    print('Updating data for each ticker', end = ': ')
    # assuming every ticker is up to the same date in the db
    for ticker in ticker_list:
        
        print(ticker, end = '..')
        temp = fdr.GetTickerDataAbridged(ticker, start_date = start_date)
        temp = temp.loc[temp['Date'] > latest_date]
        
        temp.to_sql(name = 'ticker_history', con = con, if_exists = 'append', index = False)
    
    print(str(round((time.time() - start) / 60, 2)) + ' minutes.')
    con.close()
    return
    
    

    
    