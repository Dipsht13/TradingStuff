# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 11:44:35 2024


Following this tutorial:

https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1


Instructions for accessing the training data found here:
    
https://archive.ics.uci.edu/dataset/53/iris


@author: amudek
"""

from ucimlrepo import fetch_ucirepo

import torch
from torch import nn
import torch.nn.functional as nnf

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split


# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
x_dat = iris.data.features 
y_dat = iris.data.targets

#need to have numerical values assigned to each class of Iris
iris_enum = dict(zip(y_dat['class'].unique(), [0, 1, 2]))
enum_to_class = dict(zip(iris_enum.values(), iris_enum.keys()))

y_dat['class_enum'] = y_dat['class'].map(iris_enum)

#want the actual training data to be numpy arrays
x = x_dat.values
y = y_dat['class_enum'].values


#Create model class that inherits nn.Module
class Model(nn.Module):
    # Input layer (4 features: sepal length, sepal width, petal length, petal width) --> 
    #   Hidden Layer1 (number of neurons) --> 
    #     Hidden Layer 2 (number of neurons) --> 
    #       output (which class of Iris is it?)
    
    def __init__(self, in_features = 4, h1 = 8, h2 = 9, out_features = 3):
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
    
#need to set a seed to make sure our randomized values match the tutorial video
torch.manual_seed(41)

#create an instance of our model
model = Model()


#now we need to train test split (sklearn)
#the following lines break our data up into testing and training data sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 41) 
#going to use 20% of the data as testing, 80% of the data for training^

#need to convert numpy arrays out of train_test_split to tensors
x_train = torch.FloatTensor(x_train) #float tensor
x_test = torch.FloatTensor(x_test) #float tensor
y_train = torch.LongTensor(y_train) #long (int) tensor
y_test = torch.LongTensor(y_test) #long (int) tensor

#now we need to set up the training itself
#need to be able to measure the error of the model
criterion = nn.CrossEntropyLoss() #need to look this up
#choose optimizer -- going to use the Adam Optimizer here
#                    lr = learning rate (if error doesn't go down after a bunch of iterations, lower our learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
#NOTE: The lower the learning rate, the longer it will take to train

#now get training
#NOTE: Epoch is one run through all training data in our network (one iteration)
epochs = 100
losses = [] #list to track losses with each epoch
for i in range(epochs):
    #go forward and get a prediction
    y_pred = model.forward(x_train)
    
    #measure the loss/error
    loss = criterion(y_pred, y_train) #predicted values vs. truth values
    losses.append(float(loss.detach())) #.detach().numpy() converts pytorch tensor to number
    
    #print every 10 epochs to gauge progress
    if i % 10 == 0:
        print('Epoch: ' + str(i), end = '; ')
        print('Loss: ' + str(float(loss.detach())))
    
    #Use some back propagationto fine tune the weights (take the error rate of
    # forward prop and feed it back into the network)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

#now test the model
with torch.no_grad(): #turn off backprop
    y_eval = model.forward(x_test)
    loss = criterion(y_eval, y_test)
    
#of the 30 test data points, how many did it get right?
correct = 0
with torch.no_grad():
    for ix, data in enumerate(x_test):
        y_val = model.forward(data)
        result = y_val.argmax().item()
        truth = int(y_test[ix])
        if result == truth:
            correct += 1
        
        print('ix: ' + str(ix), end = '; ')
        print('Class: ' + str(result), end = '; ')
        print('Truth: ' + str(truth), end = '; ')
        print('Match: ' + str(result == truth), end = '; ')
        print('Output: ' + str(y_val))
        
    print('Got ' + str(correct) + '/30 (' + str(round(correct/30*100, 2)) + '%) correct.')















