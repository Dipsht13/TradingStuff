# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:11:24 2024

Functions for the various performance/trading indicators.

@author: amudek
"""



def SMA(df, col, n_rows_for_avg):
    """
    
    Parameters
    ----------
    df : DataFrame
        Pandas dataframe containing price history.
    col : String
        Column name to be averaged (e.g. 'high', 'low', 'close').
    n_rows_for_avg : Integer
        Number of rows for the SMA to be applied over (since the time steps may vary).

    Returns
    -------
    smas : List
        List of float values with as many entries as the df. The first n_rows_for_avg
        will be NaN (since you need trailing data) but every other row in the df will
        have a SMA value. This list can be directly added as a new col to the df.

    """
    import numpy as np
    
    
    if len(df) <= n_rows_for_avg:
        return 'What are you doing?'
    
    
    smas = [np.nan] * n_rows_for_avg
    
    for ix in range(n_rows_for_avg, len(df)):
        
        
