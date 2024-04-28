# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:19:33 2024


Modifying my cnn script from the end of the tutorial playlist:
    
https://www.youtube.com/playlist?list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1


With the rnn examples in this video:
    
https://www.youtube.com/watch?v=Gl2WXLIMvKA


@author: amude
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import time
start = time.time()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# need to convert MNIST image files into 4-d tensors (# of images, height, width, color)
transform = transforms.ToTensor()

# load data
train_data = datasets.MNIST(root = 'saved_data/cnn_data', train = True,
                            download = True, transform = transform)
test_data = datasets.MNIST(root = 'saved_data/cnn_data', train = False,
                           download = True, transform = transform)

# need to run it with small batches of images (common to see 2-4 at a time; using 10 here)
train_loader = DataLoader(train_data, batch_size = 10, shuffle = True)
test_loader = DataLoader(test_data, batch_size = 10, shuffle = 10)

# start with defining our hyperparameters
input_size = 28 # 28x28 image will be fed in 1 line at a time
sequence_length = 28 #        ^ what that says
num_layers = 2
hidden_size = 256
num_classes = 10 # 10 possible numbers 0-9
learning_rate = 0.001
batch_size = 64
epochs = 2


# Model Class
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        #forward prop
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        
        return out
        

# need to make sure we match the video's seed value
torch.manual_seed(41)

#create an instance of your model
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# before we can train network, need a loss function & optimizer; use the same as before
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# now cue the training montage music

# need to store some shit
train_losses = []
train_correct = []

test_losses = []
test_correct = []

# train network
for epoch in range(epochs):
    
    trn_corr = 0 # running total of correct results in training
    tst_corr = 0 # ^         what he said            ^ testing
    
    for ix, (x_train, y_train) in enumerate(train_loader):
       
        x_train = x_train.to(device = device).squeeze(1) #1xNx28x28 --> Nx28x28        
        y_train = y_train.to(device = device)
        
        # run it forward
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        
        predicted = torch.max(y_pred.data, 1)[1] # add up the number of correct predictions; indexed off the 1st point
        # NOTE: Adding the 1 in the torch.max(y_pred.data, 1) results in more
        #       feedback from the torch.max() function. If you just say 
        #       torch.max(y_pred.data) it will return the max value in the 
        #       tensor. Saying torch.max(y_pred.data, 1) returns both the 
        #       max value AND that value's index. Since we care about which
        #       index contains the max value, we pull the [1] index out of the
        #       output.
        batch_corr = (predicted == y_train).sum() # how many we got correct from this batch
        trn_corr += batch_corr # update the total number of correct in training
        
        # then run it backward
        optimizer.zero_grad()
        loss.backward()
        
        # optimize
        optimizer.step()
        
        # report progress
        if ix%600 == 0:
            print('Epoch: ' + str(epoch), end = '; ')
            print('Batch: ' + str(ix), end = '; ')
            print('Loss: ' + str(loss.item()))
        
        
    train_losses.append(loss)
    train_correct.append(trn_corr)

    # now test it (still in the epoch loop, want to see how this changes over time)
    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
            
            x_test = x_test.to(device = device).squeeze(1)
            y_test = y_test.to(device = device)
            
            y_val = model(x_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()
            
    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)       

print('Training Time: ' + str(round((time.time() - start)/60, 2)) + ' minutes.')
        

# now plot shit
train_losses = [tl.item() for tl in train_losses]

plt.plot(train_losses, label = 'Training Loss')
plt.plot(test_losses, label = 'Validation Loss')
plt.title('Loss at Epoch')
plt.legend()

plt.figure()
plt.plot([t/600 for t in train_correct], label = 'Training Accuracy')
plt.plot([t/100 for t in test_correct], label = 'Validation Accuracy')
plt.title('Accuracy at the End of Each Epoch')
plt.legend()

# load the test data with batch_size == number of images (why? idk)
test_load_everything = DataLoader(test_data, batch_size = 10000, shuffle = False)

with torch.no_grad():
    correct = 0
    for x_test, y_test in test_load_everything:
        
        x_test = x_test.to(device = device).squeeze(1)
        y_test = y_test.to(device = device)
        
        y_val = model(x_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

# how many did it get right?
correct.item()/len(test_data) # 90.18% correct (ouch)


# now try running an image through the model
test_data[4143] # it's a 9
# grab just the data
test_data[4143][0]

# need to reshape it
test_data[4143][0].reshape(28, 28) #originally of shape (1, 28, 28); don't need the first dim

#show the image
plt.figure()
plt.imshow(test_data[4143][0].reshape(28, 28))

# now pass that 9 through our model
model.eval()
with torch.no_grad():
    new_prediction = model(test_data[4143][0].view(1, 28, 28)) # batch size of 1, 1 color channel, 28x28 image
    
rnn_output = torch.max(new_prediction, 1)[1] # it's a 9!!


# save the network for future reference
torch.save(model.state_dict(), 'saved_nns/first_tutorial_rnn.pt')

#to load the model in later start by initializing a blank one
model_reloaded = RNN(input_size, hidden_size, num_layers, num_classes)

#saving/loading models is done through the internal state dict
model_reloaded.load_state_dict(torch.load('saved_nns/first_tutorial_rnn.pt'))

#can verify that everything loaded in correctly with
model_reloaded.eval()
with torch.no_grad():
    newer_prediction = model_reloaded(test_data[4143][0].view(1, 28, 28)) # batch size of 1, 1 color channel, 28x28 image
    
rnn_reloaded_output = torch.max(newer_prediction, 1)[1] # it's still a 9!!

    