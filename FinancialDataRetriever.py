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
                  wks_to_keep = None, sma_periods = [20], ema_periods = [20], 
                  atr_periods = [14]):
    tkr = yf.Ticker(ticker)
    
    dat = tkr.history(start = '2000-01-01', end = None, interval = '1d')
    
    dt_1d = pd.Timedelta(days = 1); dt_3d = pd.Timedelta(days = 3)
    dt_1w = pd.Timedelta(weeks = 1); dt_4w = pd.Timedelta(weeks = 4)
    
    for ix in dat.index:
        #first handle the 1-day future data
        if ix + dt_1d in dat.index:
            dat.at[ix, '1-day future by date'] = dat.at[ix + pd.Timedelta(days = 1), 'Close']
            # dat.at[ix, '1-day future % change bin'] = RoundNearestN((dat.at[ix + dt_1d, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
        #if Friday, need to account for the weekend
        elif ix + dt_3d in dat.index:
            dat.at[ix, '1-day future by date'] = dat.at[ix + pd.Timedelta(days = 3), 'Close']
            # dat.at[ix, '1-day future % change bin'] = RoundNearestN((dat.at[ix + dt_3d, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
        
        #now 1 week (note: won't need to worry about weekends here)
        if ix + dt_1w in dat.index:
            dat.at[ix, '1-week future by date'] = dat.at[ix + pd.Timedelta(weeks = 1), 'Close']
            # dat.at[ix, '1-week future % change bin'] = RoundNearestN((dat.at[ix + dt_1w, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
        
        #now 4 weeks (note: won't need to worry about weekends here either)
        if ix + dt_4w in dat.index:
            dat.at[ix, '4-week future by date'] = dat.at[ix + pd.Timedelta(weeks = 4), 'Close']
            # dat.at[ix, '4-week future % change bin'] = RoundNearestN((dat.at[ix + dt_4w, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
            
    dat['1-day future by index'] = dat['Close'].shift(-1)
    dat['1-week future by index'] = dat['Close'].shift(-5) #only work week for stocks (5 not 7)
    dat['4-week future by index'] = dat['Close'].shift(-20)

    col_order = ['Open', 'High', 'Low', 'Close', 'Volume',
                 '1-day future by date', '1-day future by index',
                 '1-week future by date', '1-week future by index',
                 '4-week future by date', '4-week future by index']
    dat = dat[col_order]

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
    
    if yrs_to_keep:
        back_ix = yrs_to_keep * 52 * 5
    elif wks_to_keep:
        back_ix = wks_to_keep * 5
    
    dat = dat.iloc[-back_ix:]
    dat.index = dat.index.tz_localize(None)
    
    return dat


def RetrieveData(tickers, start_date = '2000-01-01', end_date = None, 
                 yrs_to_keep = None, wks_to_keep = None, sma_periods = [20],
                 ema_periods = [20], atr_periods = [14]):
    """
    Retrieves price history for the provided ticker symbols and returns
    both the price data and some performance indicators in a dictionary of
    dataframes. 

    Parameters
    ----------
    tickers : List
        List of strings containing ticker symbols of interest.
    start_date : String, optional
        YYYY-MM-DD, start date used to query yahoo finance. Should have a 
        healthy margin between how far back you want the data output to be 
        and this start date. The default is '2000-01-01'.
    end_date : String/NoneType, optional
        YYYY-MM-DD, end date for output data. If you want all data up to
        the current date make end_date None. The default is None.
    yrs_to_keep : Integer, optional
        How many years of data to return. Use either this variable or 
        wks_to_keep. If both are provided then yrs_to_keep will be used. 
        The default is None.
    wks_to_keep : TYPE, optional
        How many weeks of data to return. Use either this variable or
        yrs_to_keep. If both are provided then yrs_to_keep will be used. 
        The default is None.
    sma_periods : List, optional
        List of periods (rows) to be used for calculating the Simple Moving 
        Average. If you don't want any, provide a blank list. The default 
        is [20].
    ema_periods : TYPE, optional
        List of periods (rows) to be used for calculating the Exponential 
        Moving Average. If you don't want any, provide a blank list. The 
        default is [20].
    atr_periods : TYPE, optional
        List of periods (rows) to be used for calculating the Average True
        Range. If you don't want any, provide a blank list. The default 
        is [14].

    Returns
    -------
    data : Dictionary
        Dictionary where each key is a ticker symbol and each value is a 
        dataframe containing that ticker's history data.

    """
    
    if yrs_to_keep:
        wks_to_keep = yrs_to_keep * 52

    data = {}
    for ticker in tickers:
        print(ticker, end = '..')
        data[ticker] = GetTickerData(ticker, start_date = start_date, end_date = end_date,
                                     wks_to_keep = wks_to_keep, sma_periods = sma_periods,
                                     ema_periods = ema_periods, atr_periods = atr_periods)
        
    return data

