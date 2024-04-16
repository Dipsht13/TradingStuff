# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:44:53 2024

Script to pull stock data

@author: amudek
"""

import pandas as pd
import _pickle as pkl

import yfinance as yf
import Indicators

timespan = 8
tickers = ['^SPX']


def RoundNearestN(val, roundoff):
    """
    Function to round any value to the nearest n (instead of by number of 
    decimal points).

    Parameters
    ----------
    val : Float
        Value to be rounded.
    roundoff : Float
        What number to round to (e.g. roundoff = 0.5 will round val to the
        nearest half)

    Returns
    -------
    rounded_val : Float
        Input value rounded to the nearest roundoff value.

    """
    
    if pd.isnull(val):
        return np.nan
    
    rounded_val = round(val / roundoff) * roundoff
    
    return rounded_val


def NormalizeDfCols(df):
    
    import pandas as pd
    
    df = df.copy()
    #NOTE: Assumes are columns are numeric in the input df.
    for col in df.columns:
        max_abs_val = max(abs(df[col]))
        df[col] = df[col] / max_abs_val
        
    return df


def NormalizeDfColsByGroup(df, groupings):
    
    import pandas as pd
    
    df = df.copy()
    
    all_cols_in_a_group = []
    #first find scalings by group
    for group in groupings:
        max_abs_val = 1
        for col in group:
            all_cols_in_a_group.append(col)
            local_max_abs_val = max(abs(df[col]))
            max_abs_val = max([max_abs_val, local_max_abs_val])
        #now we have the scaling factor for the group; apply it to each col
        for col in group:
            df[col] = df[col] / max_abs_val
            
    
    loner_cols = list(set(df.columns) - set(all_cols_in_a_group))
    df[loner_cols] = NormalizeDfCols(df[loner_cols])    
    
    return df


def GetTickerData(ticker, start_date = '2000-01-01', end_date = None, yrs_to_keep = None, 
                  sma_periods = [20], ema_periods = [20], atr_periods = [14]):
    tkr = yf.Ticker(ticker)
    
    dat = tkr.history(start = '2000-01-01', end = None, interval = '1d')
    
    dt_1d = pd.Timedelta(days = 1); dt_3d = pd.Timedelta(days = 3)
    dt_1w = pd.Timedelta(weeks = 1); dt_4w = pd.Timedelta(weeks = 4)
    
    for ix in dat.index:
        #first handle the 1-day future data
        if ix + dt_1d in dat.index:
            # dat.at[ix, '1-day future'] = dat.at[ix + pd.Timedelta(days = 1), 'Close']
            dat.at[ix, '1-day future % change bin'] = RoundNearestN((dat.at[ix + dt_1d, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
        #if Friday, need to account for the weekend
        elif ix + dt_3d in dat.index:
            # dat.at[ix, '1-day future'] = dat.at[ix + pd.Timedelta(days = 3), 'Close']
            dat.at[ix, '1-day future % change bin'] = RoundNearestN((dat.at[ix + dt_3d, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
        
        #now 1 week (note: won't need to worry about weekends here)
        if ix + dt_1w in dat.index:
            # dat.at[ix, '1-week future'] = dat.at[ix + pd.Timedelta(weeks = 1), 'Close']
            dat.at[ix, '1-week future % change bin'] = RoundNearestN((dat.at[ix + dt_1w, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
            
        #now 4 weeks (note: won't need to worry about weekends here either)
        if ix + dt_4w in dat.index:
            # dat.at[ix, '4-week future'] = dat.at[ix + pd.Timedelta(weeks = 4), 'Close']
            dat.at[ix, '4-week future % change bin'] = RoundNearestN((dat.at[ix + dt_4w, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)


    for period in sma_periods:
        dat[[str(period) + '-Day SMA', str(period) + '-Day Lower Bollinger Band', str(period) + '-Day Upper Bollinger Band']] = Indicators.SMA(dat, 'Close', period, True)
        
    for period in ema_periods:
        dat[str(period) + '-Day EMA'] = Indicators.EMA(dat, 'Close', period)

    dat['RSI'] = Indicators.RSI(dat, 'Close', 14)
    dat['VWAP'] = Indicators.VWAP(dat)
    
    for period in atr_periods:
        dat[str(period) + '-Day ATR'] = Indicators.ATR(dat, period)
        
    dat[['ADX', 'positive_DI', 'negative_DI']] = Indicators.ADX(dat)
    dat[['MACD', 'MACD Signal', 'MACD Histogram']] = Indicators.MACD(dat)
    dat[['AroonUp', 'AroonDown']] = Indicators.Aroon(dat)
    dat[['Stochastic Oscillator', 'Stoch Osc Momentum']] = Indicators.SO(dat)
    
    back_ix = int(yrs_to_keep * 365.25)
    
    dat = dat.iloc[-back_ix:]
    dat.index = dat.index.tz_localize(None)
    
    return dat


data = {}
for ticker in tickers:
    data[ticker] = GetTickerData(ticker, yrs_to_keep = timespan)
    
    data[ticker].to_excel('saved_data/collected_tickers.xlsx', sheet_name = ticker)

pkl.dump(data, open('saved_data/collected_tickers.p', 'wb'))

