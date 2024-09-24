# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:11:24 2024

Functions for the various performance/trading indicators.

List of Functions
-----------------

SMA() or SimpleMovingAverage()
  -Bollinger Bands an optional output of SMA via the bollinger_bands = True/False argument
EMA() or ExponentialMovingAverage()
RSI() or RelativeStrengthIndex()
VWAP() or VolumeWeightedAveragePrice()
ATR() or AverageTrueRange()
ADX() or AverageDirectionalIndex()
MACD() or MovingAverageConvergenceDivergence()
Aroon()
SO() or StochasticOscillator()

@author: amudek
"""

import Indicators
import numpy as np


def SMA200v50(df, col):
    """
    Function to compute the 200-day SMA relative to the 50-day SMA (sma50 - sma200).
    
    
    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing price history.
    col : String
        Column name to be averaged (e.g. 'high', 'low', 'close').

    Returns
    -------
    sma_signal : List
        List of integers that are either -1, 0, or +1. A -1 indicates a sell signal
        (where 50-day SMA crossed from above to below 200-day SMA). A +1 indicates a
        buy signal (where 50-day SMA crossed above the 200-day). A 0 means 0. This 
        list can be directly added as a new col to the df.

    """
    sma200 = Indicators.SMA(df, col, 200)
    sma50 = Indicators.SMA(df, col, 50)
    
    sma200 = np.array(sma200)
    sma50 = np.array(sma50)
    
    sma_diff = list(sma50 - sma200)
    
    sma_signal = [0]
    for ix in range(1, len(sma_diff)):
        preceeding = sma_diff[ix-1]
        current = sma_diff[ix]
        if (preceeding <= 0 and current > 0):
            sma_signal.append(1)
        elif (preceeding >=0 and current < 0):
            sma_signal.append(-1)
        else:
            sma_signal.append(0)
    
    return sma_signal
    
    
def RSISignal(df, col, n_rows):
    """
    Function to find when the RSI is above 70 (overbought) or below 30 (underbought).
    
    
    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing price history.
    col : String
        Column name to be averaged (e.g. 'high', 'low', 'close').
    n_rows : Int
        Duration over which to calculate the RSI. Uses row count instead of 
        time duration because the time steps may vary across different 
        input dfs.

    Returns
    -------
    rsi_signal : List
        List of integers that are either -1, 0, or +1. A -1 indicates a sell signal
        (where RSI is >= 70, a.k.a. "overbought"). A +1 indicates a buy signal 
        (where RSI <= 30, a.k.a. "underbought"). A 0 means 0. This list can be 
        directly added as a new col to the df.

    """
    
    rsi_vals = Indicators.RSI(df, col, n_rows)
    
    rsi_signal = []
    for val in rsi_vals:
        if val >= 70:
            rsi_signal.append(-1)
        elif val <= 30:
            rsi_signal.append(1)
        else:
            rsi_signal.append(0)
            
    return rsi_signal
    
    
    
    