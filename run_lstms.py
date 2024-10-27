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
start_date = '2024-09-15'
end_date = '2024-10-25'

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


def CheckBoundsDef(newClose, lowerBound, upperBound):
    if pd.isna(newClose):
        return [None, None]
        
    if lowerBound and upperBound: # two bounds were provided
        isCorrect = lowerBound <= newClose <= upperBound
        correctOrOver = lowerBound <= newClose
    elif lowerBound: # only a lower bound was provided
        isCorrect = lowerBound <= newClose
        correctOrOver = lowerBound <= newClose
    elif upperBound: # only an upper bound was provided
        isCorrect = newCLose <= upperBound
        correctOrOver = True #???is this what we really want???
    else: # if there were no lower OR upper bounds, wtf are you doing?
        return ["What the fuck", " are you doing?"]
    
    return [isCorrect, correctOrOver]
CheckBounds = np.frompyfunc(CheckBoundsDef, 3, 1)


#%% run the models
start_date = pd.Timestamp(start_date)
end_date = pd.Timestamp(end_date)

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
    
    # load in financial data for each n_rows since start_date  
    
    data = pd.read_excel(workbook, sheet_name = ticker)
    data_of_interest = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)].copy()
    
    for ix2 in range(len(data_of_interest)-use_past_n_for_lstm):
        temp = data_of_interest[ix2:ix2+use_past_n_for_lstm]
        
        current_date = temp.iloc[-1]['Date']
        current_close = temp.iloc[-1]['Close']
        current_ix_in_data = temp.iloc[-1].name
        
        model_input = temp[cols_with_inputs].values
        
        # need to convert to tensors
        model_input = torch.tensor(model_input, dtype = torch.float)
        model_input = model_input.to(device = device).unsqueeze(0)
        
        print(current_date, end = '..')
        for ndays in ndays_to_forecast:
            
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
                
            # if enough time has passed to have an answer, store it
            if current_ix_in_data + ndays in data.index:
                actual_price = data.loc[current_ix_in_data + ndays]['Close']
                percent_change = (actual_price - current_close)*100 / current_close
            else:
                actual_price = None
                percent_change = None
            
            predictions.append({'ticker' : ticker, 'date' : current_date, 
                                'current_price': current_close,
                                'n_days' : ndays, 'prediction_bin' : predicted,
                                'prediction' : bin_sizes[predicted],
                                'truth': actual_price, 'percent_change': percent_change})
    print('Done.')
        
        
predictions = pd.DataFrame(predictions)

predictions['temp'] = ComputeBounds(predictions['current_price'].values, predictions['prediction_bin'].values)
predictions[['lower_bound', 'upper_bound']] = pd.DataFrame(predictions['temp'].to_list(), index = predictions.index)
del predictions['temp']

predictions['temp'] = CheckBounds(predictions['truth'].values, predictions['lower_bound'].values, predictions['upper_bound'].values)
predictions[['correct', 'correct_or_over']] = pd.DataFrame(predictions['temp'].to_list(), index = predictions.index)
del predictions['temp']


#%% Summary data
summary = pd.DataFrame(columns = ['Correct', 'Correct or Over', 'Out Of', '% Correct', '% Correct or Over'])
has_results = predictions.loc[predictions['correct'].notnull()].copy()

summary.at['Total', 'Correct'] = len(has_results[has_results['correct']])
summary.at['Total', 'Correct or Over'] = len(has_results[has_results['correct_or_over']])
summary.at['Total', 'Out Of'] = len(has_results)
summary.at['Total', '% Correct'] = summary.at['Total', 'Correct'] / len(has_results)
summary.at['Total', '% Correct or Over'] = summary.at['Total', 'Correct or Over'] / len(has_results)

for ndays in has_results['n_days'].unique():
    temp = has_results.loc[has_results['n_days'] == ndays]
    
    summary.at['Total ' + str(ndays) + '-day', 'Correct'] = len(temp[temp['correct']])
    summary.at['Total ' + str(ndays) + '-day', 'Correct or Over'] = len(temp[temp['correct_or_over']])
    summary.at['Total ' + str(ndays) + '-day', 'Out Of'] = len(temp)
    summary.at['Total ' + str(ndays) + '-day', '% Correct'] = summary.at['Total ' + str(ndays) + '-day', 'Correct'] / len(temp)
    summary.at['Total ' + str(ndays) + '-day', '% Correct or Over'] = summary.at['Total ' + str(ndays) + '-day', 'Correct or Over'] / len(temp)
    

for ticker in has_results['ticker'].unique():
    temp = has_results.loc[has_results['ticker'] == ticker]
    
    summary.at[ticker, 'Correct'] = len(temp[temp['correct']])
    summary.at[ticker, 'Correct or Over'] = len(temp[temp['correct_or_over']])
    summary.at[ticker, 'Out Of'] = len(temp)
    summary.at[ticker, '% Correct'] = len(temp[temp['correct']]) / len(temp)
    summary.at[ticker, '% Correct or Over'] = len(temp[temp['correct_or_over']]) / len(temp)
    
    for ndays in has_results['n_days'].unique():
        temp = has_results.loc[(has_results['ticker'] == ticker) & (has_results['n_days'] == ndays)]
        
        summary.at[ticker + ' ' + str(ndays) + '-day', 'Correct'] = len(temp[temp['correct']])
        summary.at[ticker + ' ' + str(ndays) + '-day', 'Correct or Over'] = len(temp[temp['correct_or_over']])
        summary.at[ticker + ' ' + str(ndays) + '-day', 'Out Of'] = len(temp)
        summary.at[ticker + ' ' + str(ndays) + '-day', '% Correct'] = len(temp[temp['correct']]) / len(temp)
        summary.at[ticker + ' ' + str(ndays) + '-day', '% Correct or Over'] = len(temp[temp['correct_or_over']]) / len(temp)


#%% output

writer = pd.ExcelWriter('saved_data/' + outputName + '.xlsx')
predictions.to_excel(writer, sheet_name = 'data', index = False)
summary.to_excel(writer, sheet_name = 'summary')
# predictions.to_excel(writer, sheet_name = current_date, index = False)
writer.close()



print('Run Time: ' + str(round(time.time() - start, 2)) + ' seconds.')
            
            