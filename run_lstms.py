# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 23:56:42 2024

@author: amude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd

import time
from CreateLSTM import *


start = time.time()


#%% inputs
use_past_n_for_lstm = 10 # 2-weeks inform each prediction
ndays_to_forecast = [1, 2, 3, 4, 5] # up to a 1-week forecast
outputName = 'lstm_trial1'

workbook = 'saved_data/collected_tickers.xlsx'
cols_with_inputs = ['Range', 'Diff',
                    'SMA 200-Day', 'SMA 200-Day Lower Boll', 'SMA 200-Day Upper Boll',
                    'SMA 50-Day', 'SMA Signal', 'EMA 200-Day', 'EMA 50-Day',
                    'RSI', 'RSI Signal', 'VWAP', 'MACD', 'MACD Signal', 'MACD Histogram']

bin_sizes = {0 : '5+% decrease', 1 : '2%-5% decrease', 2 : '+/-2% (no change)',
             3 : '2%-5% increase', 4 : '5+% increase'}

sequence_length = use_past_n_for_lstm # number of days to use for each input sequence to the model
hidden_size = 256
num_layers = 2
num_classes = 5
learning_rate = 0.001
batch_size = 64
epochs = 1300
input_size = len(cols_with_inputs)

tickers_of_interest = ['^SPX', 'NVDA', 'BTC-USD', 'ETH-USD']
tickers_from_dad = ['ALGT', 'AAPL', 'LMT', 'PEP', 'YUM', 'MSFT',
                    'JPM', 'WMT', 'WFC', 'AMZN']#, 'CCL']
tickers_from_joey = ['NOC', 'RTX', 'BA']
tickers_from_kyle = ['AMD', 'INTC', 'META', 'GOOG', 'CEG', 'VST']

tickers_of_interest = tickers_of_interest + tickers_from_dad +\
                      tickers_from_joey + tickers_from_kyle

predictions = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% function defs
def ComputeBoundsDef(currentClose, projectedBin):
    if projectedBin == 0:
        return [None, .95*currentClose]
    elif projectedBin == 1:
        return [.95*currentClose, .98*currentClose]
    elif projectedBin == 2:
        return [.98*currentClose, 1.02*currentClose]
    elif projectedBin == 3:
        return [1.02*currentClose, 1.05*currentClose]
    elif projectedBin == 4:
        return [1.05*currentClose, None]
    else:
        return ['What the fuck', 'are you doing?']
ComputeBounds = np.frompyfunc(ComputeBoundsDef, 2, 1)

def CheckBoundsDef(newClose, bounds):
    if (bounds[0]) and (bounds[1]): # two bounds were provided
        return bounds[0] <= newClose <= bounds[1]
    elif bounds[0]: # only a lower bound was provided
        return bounds[0] <= newClose
    elif bounds[1]: # only an upper bound was provided
        return newClose <= bounds[1]
    else: # if there were no lower OR upper bounds, wtf are you doing?
        return "What the fuck are you doing?"
CheckBounds = np.frompyfunc(CheckBoundsDef, 2, 1)

#%% run the models
print('Running models:')
n_models = len(tickers_of_interest)
for ix, ticker in enumerate(tickers_of_interest):
    # print progress
    if ix < 9:
        strix = '0' + str(ix+1)
    else:
        strix = str(ix+1)
    print('(' + strix + '/' + str(n_models) + ')', end = ' ')
    print(ticker, end = '..')
    
    # load in financial data    
    data = pd.read_excel(workbook, sheet_name = ticker)
    
    data = data.iloc[-use_past_n_for_lstm:].copy()
    current_date = str(data.iloc[-1].Date).split()[0]
    
    model_input = data[cols_with_inputs].values
    
    # need to convert to tensors
    model_input = torch.tensor(model_input, dtype = torch.float)
    model_input = model_input.to(device = device).unsqueeze(0)
    
    for ndays in ndays_to_forecast:
        print(str(ndays), end = '..')
        
        target_col = str(ndays) + '-day bin'
        model_save_file = 'saved_nns/lstm_' + ticker + '_' + str(ndays) + 'day_forecast.pt'
        
        # initialize a blank LSTM model
        model = LSTM(input_size, hidden_size, num_layers, num_classes, sequence_length).to(device)
        
        # saving/loading models is done through the internal state dict
        model.load_state_dict(torch.load(model_save_file))
        
        # now call the model
        with torch.no_grad():
            y_val = model(model_input)
            predicted = int(torch.max(y_val, 1)[1])
        
        predictions.append({'ticker' : ticker, 'date' : current_date, 
                            'n_days' : ndays, 'prediction_bin' : predicted,
                            'prediction' : bin_sizes[predicted],
                            'current_close' : data.iloc[-1]['Close']})
    print('Done.')
        
        
predictions = pd.DataFrame(predictions)

predictions['temp'] = ComputeBounds(predictions['current_close'].values, predictions['prediction_bin'].values)
predictions[['lower_bound', 'upper_bound']] = pd.DataFrame(predictions['temp'].to_list(), index = predictions.index)
del predictions['temp']


writer = pd.ExcelWriter('saved_data/' + outputName + '.xlsx')
predictions.to_excel(writer, sheet_name = current_date, index = False)
writer.close()

#%% check past predictions

print('Run Time: ' + str(round(time.time() - start, 2)) + ' seconds.')
            
            