# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 19:11:34 2024

This script takes the first nn example from this tutorial playlist:

https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1

And attempts to predict future trends of the S&P500 using that Feedforward
Neural Net (FNN). This script runs in under 10 minutes but its FNN only has
a success rate of ~47%. The loss function gets better with more training 
iterations but pushing the script to an hour run time only improved the 
performance to ~49% and the loss function noticeably plateaus in the final
10 minutes.

Update: Fixed my fuck up in the MACD calc and got ~51% success rate with the
        long run time version of the script.
        
Update: Normalizing the input data gets ~50.5% success rate with the short 
        run time version.

@author: amudek
"""

import torch
from torch import nn
import torch.nn.functional as nnf

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import yfinance as yf
import Indicators

import time
start = time.time()


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


#%% collect data

# fetch dataset
#start with the S&P500 (using the SPX mutual fund)
sp500 = yf.Ticker('^SPX')

dat = sp500.history(start = '2014-01-01', end = None, interval = '1d')
dt_1d = pd.Timedelta(days = 1); dt_3d = pd.Timedelta(days = 3)
dt_1w = pd.Timedelta(weeks = 1); dt_4w = pd.Timedelta(weeks = 4)
for ix in dat.index:
    #first handle the 1-day future data
    if ix + dt_1d in dat.index:
        # dat.at[ix, '1-day future'] = dat.at[ix + pd.Timedelta(days = 1), 'Close']
        dat.at[ix, '1-day future'] = RoundNearestN((dat.at[ix + dt_1d, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
    #if Friday, need to account for the weekend
    elif ix + dt_3d in dat.index:
        # dat.at[ix, '1-day future'] = dat.at[ix + pd.Timedelta(days = 3), 'Close']
        dat.at[ix, '1-day future'] = RoundNearestN((dat.at[ix + dt_3d, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
    
    #now 1 week (note: won't need to worry about weekends here)
    if ix + dt_1w in dat.index:
        # dat.at[ix, '1-week future'] = dat.at[ix + pd.Timedelta(weeks = 1), 'Close']
        dat.at[ix, '1-week future'] = RoundNearestN((dat.at[ix + dt_1w, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)
        
    #now 4 weeks (note: won't need to worry about weekends here either)
    if ix + dt_4w in dat.index:
        # dat.at[ix, '4-week future'] = dat.at[ix + pd.Timedelta(weeks = 4), 'Close']
        dat.at[ix, '4-week future'] = RoundNearestN((dat.at[ix + dt_4w, 'Close'] - dat.at[ix, 'Close']) / dat.at[ix, 'Close'] * 100, 0.5)

dat[['SMA', 'LowerBoll', 'UpperBoll']] = Indicators.SMA(dat, 'Close', 20, True)
dat['EMA'] = Indicators.EMA(dat, 'Close', 20)
dat['RSI'] = Indicators.RSI(dat, 'Close', 14)
dat['VWAP'] = Indicators.VWAP(dat)
dat['ATR'] = Indicators.ATR(dat, 14)
dat[['ADX', '+DI', '-DI']] = Indicators.ADX(dat)
dat[['MACD', 'Signal', 'Hist']] = Indicators.MACD(dat)
dat[['AroonUp', 'AroonDown']] = Indicators.Aroon(dat)
dat[['StochOsc', 'Momentum']] = Indicators.SO(dat)
#17 total indicator cols
# x_dat = dat[['SMA', 'LowerBoll', 'UpperBoll', 'EMA', 'RSI', 'VWAP', 'ATR', 'ADX',
#              '+DI', '-DI', 'MACD', 'Signal', 'Hist', 'AroonUp', 'AroonDown',
#              'StochOsc', 'Momentum']].copy()
# y_dat = dat[['1-day future', '1-week future', '4-week future']].copy()

# x_dat = x_dat.values
# y_dat = y_dat.values

def PercentToEnum(val):
    if val < -5:
        return 0
    elif -5 <= val < -0.5:
        return 1
    elif -0.5 <= val <= 0.5:
        return 2
    elif 0.5 < val <= 5:
        return 3
    elif 5 < val:
        return 4
    else:
        return 'wtf'
  
dat['1-day'] = dat['1-day future'].apply(PercentToEnum)
dat['1-week'] = dat['1-week future'].apply(PercentToEnum)
dat['4-week'] = dat['4-week future'].apply(PercentToEnum)
# y_dat['1-day'] = y_dat['1-day future'].apply(PercentToEnum)
# y_dat['1-week'] = y_dat['1-week future'].apply(PercentToEnum)
# y_dat['4-week'] = y_dat['4-week future'].apply(PercentToEnum)

one_day_dat = dat.loc[~dat['1-day future'].isnull()].copy()
one_week_dat = dat.loc[~dat['1-week future'].isnull()].copy()
four_week_dat = dat.loc[~dat['4-week future'].isnull()].copy()

#let's try the 1 week data first
x_dat = one_week_dat[['SMA', 'LowerBoll', 'UpperBoll', 'EMA', 'RSI', 'VWAP', 'ATR', 'ADX',
                      '+DI', '-DI', 'MACD', 'Signal', 'Hist', 'AroonUp', 'AroonDown',
                      'StochOsc', 'Momentum']].copy()
y_dat = one_week_dat['1-week'].copy()

x_dat = x_dat.iloc[-2100:].copy()
y_dat = y_dat.iloc[-2100:].copy()

groupings_for_normalization = [['SMA', 'LowerBoll', 'UpperBoll', 'EMA'],
                               ['ADX', '+DI', '-DI'],
                               ['MACD', 'Signal'],
                               ['AroonUp', 'AroonDown']]
x_dat = NormalizeDfColsByGroup(x_dat, groupings_for_normalization)

x_dat = x_dat.values
y_dat = y_dat.values


#now have our nn data ready
#make the model
class Model(nn.Module):
    # Input layer (4 features: sepal length, sepal width, petal length, petal width) --> 
    #   Hidden Layer1 (number of neurons) --> 
    #     Hidden Layer 2 (number of neurons) --> 
    #       output (which class of Iris is it?)
    
    def __init__(self, in_features = 17, h1 = 34, h2 = 42, out_features = 5):
        #need to following line to instantiate nn.Module
        super().__init__() #required since we're inheriting from a parent class
        
        #define model structure
        self.fc1 = nn.Linear(in_features, h1) #fully connects (fc) inputs to hl1
        self.fc2 = nn.Linear(h1, h2) #hl1 --> hl2
        self.out = nn.Linear(h2, out_features) #hl2 --> output
        
    #need a method to propogate forward through each layer of the network
    def forward(self, x):
        x = nnf.relu(self.fc1(x)) #bounds numbers to (0, inf]
        x = nnf.relu(self.fc2(x))
        x = self.out(x)
        
        return x

#create an instance of our model
model = Model()

#now we need to train test split (sklearn)
#the following lines break our data up into testing and training data sets
x_train, x_test, y_train, y_test = train_test_split(x_dat, y_dat, test_size = 0.25)
#going to use 20% of the data as testing, 80% of the data for training^

#need to convert numpy arrays out of train_test_split to tensors
x_train = torch.FloatTensor(x_train) #float tensor
x_test = torch.FloatTensor(x_test) #float tensor
y_train = torch.LongTensor(y_train.astype(int)) #long (int) tensor
y_test = torch.LongTensor(y_test.astype(int)) #long (int) tensor

#now we need to set up the training itself
#need to be able to measure the error of the model
criterion = nn.CrossEntropyLoss() #need to look this up
#choose optimizer -- going to use the Adam Optimizer here
#                    lr = learning rate (if error doesn't go down after a bunch of iterations, lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
#NOTE: The lower the learning rate, the longer it will take to train
#NOTE: Can also perform worse if lr is too low. Idk why but good rule of 
#      thumb is don't go lower than 1e-4?


#now get training
#NOTE: Epoch is one run through all training data in our network (one iteration)
epochs = 1000000#0<--adding another zero increases the run time to ~1 hr but only improves the success rate to ~49%
losses = [] #list to track losses with each epoch
loop_start = time.time()
for i in range(epochs):
    #go forward and get a prediction
    y_pred = model.forward(x_train)
    
    #measure the loss/error
    loss = criterion(y_pred, y_train) #predicted values vs. truth values
    losses.append(float(loss.detach())) #.detach().numpy() converts pytorch tensor to number
    
    #print every n epochs to gauge progress
    if i % 1000 == 0:
        
        time_per_iter = (time.time() - loop_start) / (i+1)
        time_remaining = (epochs - i) * time_per_iter / 60
        
        print('\r  ' + str(round(100*i/epochs, 2)) + '%', end = '    ')
        print('Loss: ' + str(float(loss.detach())), end = '    ')
        print(str(round(time_remaining, 2)) + ' min remaining', end = '')
    
    #Use some back propagationto fine tune the weights (take the error rate of
    # forward prop and feed it back into the network)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
print('\n')
    

print('\nTesting...', end = '')
#now test the model
with torch.no_grad(): #turn off backprop
    y_eval = model.forward(x_test)
    loss = criterion(y_eval, y_test)
    
#of the 30 test data points, how many did it get right?
correct = 0; total = len(y_test)
with torch.no_grad():
    for ix, data in enumerate(x_test):
        y_val = model.forward(data)
        result = y_val.argmax().item()
        truth = int(y_test[ix])
        if result == truth:
            correct += 1
        
#         print('ix: ' + str(ix), end = '; ')
#         print('Class: ' + str(result), end = '; ')
#         print('Truth: ' + str(truth), end = '; ')
#         print('Match: ' + str(result == truth), end = '; ')
#         print('Output: ' + str(y_val))
        
print('Got ' + str(correct) + '/' + str(total) + ' (' + str(round(correct/total*100, 2)) + '%) correct.')


from datetime import datetime
now = datetime.now()
out_name = 'fnn_fuckaround-' + str(now.year) + '_' + str(now.month) + '_' +\
                    str(now.day) + '-' + str(now.hour) + '_' + str(now.minute)
torch.save(model.state_dict(), 'saved_nns/' + out_name)


print('Run time: ' + str(round((time.time() - start)/60, 2)) + ' min')