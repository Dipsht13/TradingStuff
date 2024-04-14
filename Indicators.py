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


def SMA(df, col, n_rows_for_avg, bollinger_bands = False):
    """
    Function to compute the Simple Moving Average.
    
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
        #return the output as a dataframe with an index matching the input df
        # this way the user can still add the new df columns directly from the
        # function output (e.g. df[['sma', 'lb', 'ub']] = SMA(df, 'Close', 20, True))
        import pandas as pd
        return pd.DataFrame({'sma' : smas, 'lower_bollinger' : lowerbols, 
                             'upper_bollinger' : upperbols}, index = df.index)
    
    return smas

def SimpleMovingAverage(df, col, n_rows_for_avg, bollinger_bands = False):
    """
    See SMA() definition.
    """
    
    if bollinger_bands:
        output_df = SMA(df, col, n_rows_for_avg, True)
        return output_df
    
    smas = SMA(df, col, n_rows_for_avg)
    
    return smas
	

def EMA(df, col, n_rows_for_avg, weight = 2):
    """
    Function to compute the Exponential Moving Average.

    Parameters
    ----------
    df : DataFrame
        Historical price data for the stock/crypto of interest.
    col : String/Integer
        Column of interest in the df.
    n_rows_for_avg : Integer
        Duration over which to calculate the EMA. Uses row count instead of 
        time duration because the time steps may vary across different 
        input dfs.
    weight : Integer, optional
        The weight to use for the EMA weight multiplier. Not sure what the
        physical/mathematical explanation is for what this does. The default
        value is 2 because that's what's used everywhere. I only made it an
        optional input value because I could.

    Returns
    -------
    emas : List
        List of EMA values, one for each row in the df. The first 
        n_rows_for_avg number-1 of rows will be NaN since we need trailing
        data to compute the EMA. All other rows will have a float value.

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

def ExponentialMovingAverage(df, col, n_rows_for_avg, weight = 2):
    """
    See EMA() definition.
    """
    
    emas = EMA(df, col, n_rows_for_avg, weight = 2)
    
    return emas


def RSI(df, col, n_rows_for_rsi):
    """
    Function to compute the Relative Strength Index.

    Parameters
    ----------
    df : DataFrame
        Historical price data for the stock/crypto of interest.
    col : String/Integer
        Column of interest in the df.
    n_rows_for_rsi : Integer
        Duration over which to calculate the RSI. Uses row count instead of 
        time duration because the time steps may vary across different 
        input dfs.

    Returns
    -------
    rsi : TYPE
        DESCRIPTION.

    """
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

def RelativeStrengthIndex(df, col, n_rows_for_rsi):
    """
    See RSI() definition.

    """
    
    rsi = RSI(df, col, n_rows_for_rsi)
    
    return rsi
        

def VWAP(df):
    """
    Function to compute the Volume Weighted Average Price.
    NOTE: This function assumes columns called 'High', 'Low', 'Close', and
          'Volume' appear in the df (consistent with yfinance data). It also
          assumes the df index is comprised of pandas.Timestamp objects (once
          again, consistent with yfinance data). VWAP is always calculated as
          a daily value. Any rows that fall on the same day will report back
          the same VWAP value. Any df that has time intervals >= 1 day per 
          row will always report back the average price over the High, Low, 
          & Close.

    Parameters
    ----------
    df : DataFrame
        Dataframe containing historical price data for a stock/coin. Must
        have columns named 'High', 'Low', 'Close', and 'Volume'. If the data
        that can be found in each column is not obvious by the name, forget
        about stocks and go buy lottery tickets.

    Returns
    -------
    vwaps : List
        List of floats providing the VMAP value for each row in the df.

    """
    
    if not set(['High', 'Low', 'Close', 'Volume']).issubset(set(df.columns)):
        return ['InputError: You need High, Low, Close, and Volume columns there bud.'] * len(df)
    
    #vwap is a daily metric so need to figure out when the days change in the df
    df = df[['High', 'Low', 'Close', 'Volume']].copy()
    df['yr_day'] = [(ix.year, ix.day_of_year) for ix in df.index]
    
    vwaps = []
    for yr_day in df['yr_day'].unique():
        temp = df.loc[df['yr_day'] == yr_day].copy()
        
        temp['Typical Price'] = temp[['High', 'Low', 'Close']].sum(axis = 1) / 3
        temp['PV'] = temp['Typical Price'] * temp['Volume']
        vwap = temp['PV'].sum() / temp['Volume'].sum()
        
        vwaps = vwaps + [vwap] * len(temp) #returning the same value for all rows related to the current day
                
    return vwaps

def VolumeWeightedAveragePrice(df):
    """
    See VWAP() definition.
    """
    
    vwaps = VWAP(df)
    
    return vwaps


def ATR(df, n_rows_for_avg):
    """
    Function to compute the Average True Range.

    Parameters
    ----------
    df : DataFrame
        Price history data. Must include High, Low, and Close data.
    n_rows_for_avg : Integer
        Number of rows to average over. Independent of time; time will depend
        on the time steps used in the price history. Recommended interval for
        n_rows_for_avg is 14 days.

    Returns
    -------
    atrs : List
        List of floats returning the computed ATR values over the input df.

    """
    import numpy as np
    
    #we need n_rows_for_avg + 1 rows of trailing data before ATR vals will be returned
    if len(df) <= n_rows_for_avg + 1:
        return ['InputError: Not enough rows in the df there bud.'] * len(df)
    
    #first find the True Range for each row in the df (except the first, need a trailing Close value)
    trs = [np.nan]
    for ix in range(1, len(df)):
        tr = max([df.iloc[ix]['High'] - df.iloc[ix]['Low'],
                  abs(df.iloc[ix]['High'] - df.iloc[ix-1]['Close']),
                  abs(df.iloc[ix]['Low'] - df.iloc[ix-1]['Close'])])
        trs.append(tr)
        
    #now we have TR values for each entry in the df
    #the ATR is the simple average of the TR values over n_rows_for_avg 
    atrs = [np.nan] * (n_rows_for_avg + 1)
    for ix in range(n_rows_for_avg + 1, len(df)): #offset by 1 b/c TR doesn't start until row 1
        atr = np.array(trs[ix - n_rows_for_avg : ix]).sum() / n_rows_for_avg
        atrs.append(atr)
    
    return atrs

def AverageTrueRange(df, n_rows_for_avg):
    """
    See ATR() definition.
    """
    
    atrs = ATR(df, n_rows_for_avg)
    
    return atrs


def SMAOnList(val_list, average_over_n):
    """
    Utility function because ADX requires a million fucking rounds of SMAs.

    Parameters
    ----------
    val_list : List
        List of numerical values.
    average_over_n : Int
        Number of consecutive values in the list to average over at a time.

    Returns
    -------
    smas : List
        List of numbers containing the thing you asked for.

    """
    import numpy as np
    
    smas = [np.nan] * average_over_n
    
    for ix in range(average_over_n, len(val_list)):
        sma = np.array(val_list[ix - average_over_n : ix]).sum() / average_over_n
        smas.append(sma)
    
    return smas


def ADX(df, n_rows_for_adx = 14):
    """
    Function to compute the Average Directional Index.
    Output is returned as a Pandas DataFrame so that it can be directly
    inserted as columns in the df passed into the function.

    Parameters
    ----------
    df : DataFrame
        Historical price data for a stock/coin. Must contain 'High' and 'Low'
        columns. 
    n_rows_for_adx : TYPE, optional
        Number of rows in the df to average over. The default is 14 since 14
        days is the standard interval for ADX. This only works if the input
        df uses 1-day time steps in the price data. If not, adjust this input 
        accordingly.

    Returns
    -------
    adxs : List
        List of Average Direction Index values for each row in the input df.
    di_pluses : List
        List of positive directional indicator values.
    di_minuses : List
        List of negative directional indicator values.

    """
    #Default is 14 days (ASSUMING 1-DAY STEPS IN PRICE DATA!!)
    
    import numpy as np
    import pandas as pd
    
    #first find the positive and negative directional movement
    dm_pluses = [np.nan]; dm_minuses = [np.nan] #requires 1 row of trailing data
    for ix in range(1, len(df)):
        dm_plus  = max([df.iloc[ix]['High'] - df.iloc[ix-1]['High'], 0]) #no underflow
        dm_minus = max([df.iloc[ix-1]['Low'] - df.iloc[ix]['Low'], 0]) #no overflow
        
        dm_pluses.append(dm_plus)
        dm_minuses.append(dm_minus)
        
    #will need to smooth the directional movement values (SMA over n_rows_for_adx)
    smooth_dm_pluses = SMAOnList(dm_pluses, n_rows_for_adx)
    smooth_dm_minuses = SMAOnList(dm_minuses, n_rows_for_adx)
    
    #will also need the ATR data in the next step
    atrs = ATR(df, n_rows_for_adx) #NOTE: ATR has the same hanging first row that ADX has so the NaN columns should line up
    
    #use the smoothed values to compute the directional indicators
    di_pluses = [np.nan] * (n_rows_for_adx + 1)
    di_minuses = [np.nan] * (n_rows_for_adx + 1)
    #finally use the +/- directional indicators to compute the average direction index
    dxs = [np.nan] * (n_rows_for_adx + 1)
    for ix in range(n_rows_for_adx + 1, len(df)):        
        di_plus = smooth_dm_pluses[ix] / atrs[ix] * 100
        di_minus = smooth_dm_minuses[ix] / atrs[ix] * 100
        
        dx = abs(di_plus - di_minus) / abs(di_plus + di_minus) * 100
        
        di_pluses.append(di_plus)
        di_minuses.append(di_minus)
        dxs.append(dx)
    
    #the final ADX value is the SMA of the dx values
    adxs = SMAOnList(dxs, n_rows_for_adx)
    
    #ADX usually plots 3 lines: the ADX, +DI, and -DI
    #NOTE: Formatting the return as a dataframe so that this function can be
    #      called the same way as every other function in this script:
    #           df[['adx', 'di_plus', 'di_minus']] = ADX(df, 14)
    return pd.DataFrame({'ADX' : adxs, '+DI' : di_pluses, '-DI' : di_minuses}, index = df.index)

def AverageDirectionalIndex(df, n_rows_for_adx = 14):
    """
    See ADX() definition.
    """
    
    output_df = ADX(df, n_rows_for_adx)
    
    return output_df
  

def EMAOnList(val_list, average_over_n, weight = 2):
    """
    Utility function for MACD.

    Parameters
    ----------
    val_list : List
        List of numerical values.
    average_over_n : Integer
        Number of consecutive values in the list to average over at a time.
    weight : Integer
        Weighting value to compute the weighting multiplier, k.

    Returns
    -------
    emas : List
        List of numbers containing the thing you asked for.

    """
    import numpy as np
    
    k = weight / (average_over_n + 1)
    
    #initial rows (average_over_n - 1) will be blank since we need trailing data
    emas = [np.nan] * (average_over_n - 1)
    
    #need a starting point; start with an SMA value
    starting_ix = average_over_n
    trailing_ema = np.mean(val_list[:starting_ix])
    emas.append(trailing_ema)
    
    #need to make sure we have a non-nan starting point
    while np.isnan(trailing_ema):
        starting_ix += 1
        trailing_ema = np.mean(val_list[starting_ix - average_over_n : starting_ix])
        emas.append(trailing_ema)
    
    #now that we have a non-nan starting point, compute the emas
    for ix in range(starting_ix, len(df)):
        ema = k * val_list[ix] + trailing_ema * (1 - k)
        # ema = k * (df[col].iloc[ix] - previous) + previous
        emas.append(ema)
        
        trailing_ema = ema
     
    return emas


def MACD(df, col = 'Close', ema1_period = 12, ema2_period = 26, signal_period = 9):
    """
    Function to compute the Moving Average Convergence Divergence indicator.
    MACD is computed 3 EMAs. The difference between the first 2 EMAs determines
    the MACD line. The signal line is its own EMA which is plotted alongside
    the MACD line. The difference between the MACD line and the Signal line
    determines the histogram that's plotted along with the 2 lines. The
    output is returned as a Pandas DataFrame so that it can be directly
    inserted as columns in the df passed into the function.

    Parameters
    ----------
    df : DataFrame
        Historical price data for a stock/coin.
    col : String, optional
        Which column in the df should be used for the EMAs. Defaults to 'Close'
        because the closing price is most commonly used for the MACD.
    ema1_period : Integer, optional
        Time interval (in non-dim dataframe rows) to be used for the first EMA
        for the MACD. Typically, 12 days is used for this EMA.
    ema2_period : Integer, optional
        Time interval (in non-dim dataframe rows) to be used for the second EMA
        for the MACD. Typically, 26 days is used for this EMA.
    signal_period : Integer, optional
        Time interval (in non-dim dataframe rows) to be used for the Signal line
        EMA. Typically, 9 days is used. 
    Returns
    -------
    macd : List
        List of values for the MACD line.
    signal : List
        List of values for the Signal line.
    histogram : List
        List of values for the histogram in the MACD chart.

    """
    import numpy as np
    import pandas as pd
    
    #need 2 EMAs to compute the MACD line
    ema1 = EMA(df, col, ema1_period)
    ema2 = EMA(df, col, ema2_period)
    
    #now subtract the 26-period EMA from the 12-period to get the MACD line.
    macd = list(np.array(ema1) - np.array(ema2))
    
    #also need the signal line
    signal = EMAOnList(macd, signal_period)
    
    #finally subtract the Signal from the MACD to get the histogram data
    histogram = list(np.array(macd) - np.array(signal))
       
    
    return pd.DataFrame({'macd' : macd, 'signal' : signal, 'histogram' : histogram},
                         index = df.index)

def MovingAverageConvergenceDivergence(df, col = 'Close', ema1_period = 12, ema2_period = 26, signal_period = 9):
    """
    See MACD() definition.
    """
    
    output_df = MACD(df, col, ema1_period, ema2_period, signal_period)
    
    return output_df


def Aroon(df, aroon_period = 25):
    """
    Function to compute the Aroon indicator. The output is returned as a 
    Pandas DataFrame so that it can be directly inserted as columns in the 
    df passed into the function.

    Parameters
    ----------
    df : DataFrame
        Historical price data for a stock/coin. Requires 'High' and 'Low' df
        columns.
    aroon_period : Integer
        Number of rows to look back when finding the Aroon Up and Aroon Down
        values. This function defaults to 25 but it is common to use 20 or 25.

    Returns
    -------
    aroon_ups : List
        Values for the Aroon Up indicator.
    aroon_downs : List
        Values for the Aroon Down indicator.

    """
    import numpy as np
    import pandas as pd
    
    
    if aroon_period >= len(df):
        return ['InputError: Lol what you trying to do there bud?'] * len(df)
    
    aroon_ups   = [np.nan] * aroon_period
    aroon_downs = [np.nan] * aroon_period
    for ix in range(aroon_period, len(df)):
        trailing_periods = df.iloc[ix - aroon_period : ix]
        
        #need to know when the highest high and lowest low occurred
        highest_high_ix = trailing_periods['High'].argmax()
        lowest_low_ix = trailing_periods['Low'].argmin()
        
        #how many time periods (whatever your df time steps are) have passed 
        # since each happened?
        periods_since_high = ix - highest_high_ix
        periods_since_low = ix - lowest_low_ix
        
        #now plug these into Aroon formulae
        aroon_up = (1 - periods_since_high / aroon_period) * 100
        aroon_down = (1 - periods_since_low / aroon_period) * 100
        
        aroon_ups.append(aroon_up)
        aroon_downs.append(aroon_down)
        
    return pd.DataFrame({'aroon_up' : aroon_ups, 'aroon_down' : aroon_downs},
                        index = df.index)


def SO(df, lookback_period = 14, reference_sma_period = 3):
    """
    Function to compute the Stochastic Oscillation. The output is returned as 
    a Pandas DataFrame so that it can be directly inserted as columns in the 
    df passed into the function.
    

    Parameters
    ----------
    df : DataFrame
        Price history of the stock/coin. Must have 'High', 'Low', and 'Close'
        df columns.
    lookback_period : Integer, optional
        Number of rows to include when looking at past data. The default is 14.
    reference_sma_period : Integer, optional
        An SMA is included with stochastic oscillators as a momentum indicator.
        This determines the time interval for that SMA. The default is 3.

    Returns
    -------
    stoch_osc : List
        List of stochastic oscillator values for the input df.
    reference_sma : List
        List of values for the mommentum-indicating SMA that usually is plotted
        alongside the stochastic oscillation data.

    """
    
    import numpy as np
    import pandas as pd
    
    stoch_osc = [np.nan] * lookback_period
    for ix in range(lookback_period, len(df)):
        trailing_periods = df.iloc[ix - lookback_period : ix]
        
        highest_high = max(trailing_periods['High'])
        lowest_low = min(trailing_periods['Low'])
        
        stoch = (trailing_periods.iloc[-1]['Close'] - lowest_low) / (highest_high - lowest_low) * 100
        stoch_osc.append(stoch)
        
    reference_sma = SMA(df, 'Close', 3)
    
    return pd.DataFrame({'stochastic_osc' : stoch_osc, 'momentum' : reference_sma},
                        index = df.index)
    
def StochasticOscillator(df, lookback_period = 14, reference_sma_period = 3):
    """
    See the SO() definition.
    """
    
    output_df = SO(df, lookback_period, reference_sma_period)
    
    return output_df


