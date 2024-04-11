# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:11:24 2024

Functions for the various performance/trading indicators.

@author: amudek
"""



def SMA(df, col, n_rows_for_avg, bollinger_bands = False):
    """
    
    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing price history.
    col : String
        Column name to be averaged (e.g. 'high', 'low', 'close').
    n_rows_for_avg : Integer
        Number of rows for the SMA to be applied over (since the time steps may vary).
    bollinger_bands : Boolean
        Whether or not to return the bollinger bands with the SMA information.
        Defaults to False.

    Returns
    -------
    smas : List
        List of float values with as many entries as the df. The first n_rows_for_avg
        will be NaN (since you need trailing data) but every other row in the df will
        have a SMA value. This list can be directly added as a new col to the df.
    lowerbols : List
        List of lower bollinger bands. The first n_rows_for_avg will be NaN.
    upperbols : List
        List of upper bollinger bands. The first n_rows_for_avg will be NaN.

    """
    import numpy as np
    
    
    if len(df) <= n_rows_for_avg:
        return 'What are you doing?'
    
    #initial rows (n_rows_for_avg - 1) will be blank since we need trailing data
    smas = [np.nan] * (n_rows_for_avg - 1)
    lowerbols = [np.nan] * (n_rows_for_avg - 1)
    upperbols = [np.nan] * (n_rows_for_avg - 1)
    
    #now loop through the dataframe and compute SMA as well as the Bollinger Bands
    for ix in range(n_rows_for_avg, len(df) + 1):
        #SMA is the simple mean
        sma = df.iloc[ix - n_rows_for_avg : ix][col].mean()
        smas.append(sma)
        
        #and the Bollinger Bands are +/- 2 standard deviations from the mean
        std = df.iloc[ix - n_rows_for_avg : ix][col].std()
        lbol = sma - (2 * std)
        ubol = sma + (2 * std)
        
        lowerbols.append(lbol)
        upperbols.append(ubol)
        
    if bollinger_bands:
        return smas, lowerbols, upperbols
    
    return smas
	

def EMA(df, col, n_rows_for_avg, weight = 2):
    """

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    n_rows_for_avg : TYPE
        DESCRIPTION.
    weight : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    emas : TYPE
        DESCRIPTION.

    """
    import numpy as np
    
    k = weight / (n_rows_for_avg + 1)
    
    #initial rows (n_rows_for_avg - 1) will be blank since we need trailing data
    emas = [np.nan] * (n_rows_for_avg - 1)
    
    #need a starting point; start with an SMA value
    trailing_ema = df.iloc[:n_rows_for_avg][col].mean()
    emas.append(trailing_ema)
    
    for ix in range(n_rows_for_avg, len(df)):
        ema = k * df[col].iloc[ix] + trailing_ema * (1 - k)
        # ema = k * (df[col].iloc[ix] - previous) + previous
        emas.append(ema)
        
        trailing_ema = ema
     
    return emas


def RSI(df, col, n_rows_for_rsi):
    #NOTE: Col usually is daily close price
    import numpy as np
    
    rsi = [np.nan] * (n_rows_for_rsi - 1)
    for ix in np.arange(n_rows_for_rsi, len(df) + 1):
        dat = df.iloc[ix - n_rows_for_rsi : ix][col]
        
        pos = 0; neg = 0
        pos_count = 0; neg_count = 0
        for jx in np.arange(1, n_rows_for_rsi):
            change = dat.iloc[jx] - dat.iloc[jx-1]
            
            if change > 0:
                pos = pos + change
                pos_count += 1
            elif change < 0:
                neg = neg + abs(change)
                neg_count += 1
            else:
                continue
        
        if pos_count > 0:
            pos_sma = pos / pos_count
        else:
            pos_sma = 0
            
        if neg_count > 0:
            neg_sma = neg / neg_count
        else:
            neg_sma = 0
        
        # k = 2 / 15 #using N=14 for now
        if neg_sma == 0:
            rsi.append(100)
        else:
            rs = pos_sma / neg_sma
            rsi.append(100 - 100 / (1 + rs))
        
    return rsi
        
