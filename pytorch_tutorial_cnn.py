# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:19:33 2024

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

# before we define our full CNN model we'll step through the convolution & pooling
#  layers step-by-step with 1 image to see how the data is handled/changed
# first describe convolutional layers and what they're doing (will have 2 conv layers)
conv1 = nn.Conv2d(1, 6, 3, 1) # 1 image in, 6 filters/outputs, 3x3 filter/kernal/matrix size, step size of 1)
conv2 = nn.Conv2d(6, 16, 3, 1) # 6 outputs from conv1 as inputs; arbitrarily pick 16 outputs; otherwise use the same parameters (can change these later if we want)

# grab 1 MNIST image
for ix, (x_train, y_train) in enumerate(train_data):
    break

'''
x_train.shape # returns torch.Size([1, 28, 28]) meaning 1 image that's 28x28 pixels
'''

# x_train is a single 2d image (total of 3 dim); want a 4-d batch
x = x_train.view(1, 1, 28, 28) # 1 batch, containing 1 image, that's 28x28 pixels

# perform 1st convolution
x = F.relu(conv1(x)) # rectified linear unit for activation function

'''
x.shape # returns torch.Size([1, 6, 26, 26]) meaning 1 single image, 6 filters (as we told it to do), 26x26 pixels
# NOTE: x image is 26x26 pixels instead of 28x28 b/c there's a default padding value of 2 
#       in the Conv2d constructor.
'''

# now pass through pooling layer
x = F.max_pool2d(x, 2, 2) # kernal = 2, stride = 2

'''
# NOTE: Pooling layer takes nxm pixel image and compresses it by some kernal size k
#       to become an n/k x m/k pixel image.
x.shape # now returns torch.Size([1, 6, 13, 13])
'''

# perform 2nd convolution
x = F.relu(conv2(x))

'''
x.shape # now returns torch.Size(1, 16, 11, 11])
# NOTE: Still 1 image. 16 outputs from definition of conv2. Lost 2 more pixels
#       to padding.
'''

# another pooling layer
x = F.max_pool2d(x, 2, 2)

'''
x.shape # now returns torch.Size([1, 16, 5, 5])
# NOTE: Dividing by 2 again but 11/2 = 5.5 and has to be an int. Can't add info 
#       to round up to 6; it will always round down.
'''

# that's it. we've made it through our conv and pool layers. next step is the 
# "neural net" part of the process

# Model Class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__() #initialize parent class
        
        # create convolution layers
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        
        # also need some fully connected layers (google image of CNN structure)
        self.fc1 = nn.Linear(5*5*16, 120)  #look at x.shape from line 82. 5x5 pixels and 16 filters dictate the input to this layer. arbitrarily selecting 120 neurons
        self.fc2 = nn.Linear(120, 84) #arbitrarily picking the number of neurons here
        self.out = nn.Linear(84, 10) #84 inputs have to become 10 outputs to match data set groupings (trying to identify handwritten numbers 0-9)
    
    #need a propagation function
    def forward(self, X):
        # first convolution pass
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2) # 2x2 kernel & stride 2
        # second convolution pass
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        
        # now we need to flatten it out into 1-d array for linear layers
        X = X.view(-1, 16*5*5) # -1 is to vary the batch size, 16*5*5 comes from output of the second pooling call
        
        #  and pass it through the linear layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.out(X) # no relu for output layer
        
        return F.log_softmax(X, dim = 1) 
        # NOTE: Softmax algorithm normalizes the data. Taking the log of the softmax 
        #       helps with stability. Outputting this instead of the raw X value (like
        #       we did for the previous nn example) can perform better in the CNN.
        
# need to make sure we match the video's seed value
torch.manual_seed(41)

#create an instance of your model
model = ConvolutionalNetwork()

# before we can train network, need a loss function & optimizer; use the same as before
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

# now cue the training montage music

# need to store some shit
train_losses = []
train_correct = []

test_losses = []
test_correct = []

# loop through epochs
epochs = 5
for epoch in range(epochs):
    trn_corr = 0 # running total of correct results in training
    tst_corr = 0 # ^         what he said            ^ testing
    
    # train
    for b, (x_train, y_train) in enumerate(train_loader):
        b += 1 #start batches at 1
        
        y_pred = model(x_train) # get predicted values from training set
        loss = criterion(y_pred, y_train) # you know what it is bitch
        
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
        
        # now update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # report progress
        if b%600 == 0:
            print('Epoch: ' + str(epoch), end = '; ')
            print('Batch: ' + str(b), end = '; ')
            print('Loss: ' + str(loss.item()))
            
    train_losses.append(loss)
    train_correct.append(trn_corr)

    # now test it (still in the epoch loop, want to see how this changes over time)
    with torch.no_grad():
        for b, (x_test, y_test) in enumerate(test_loader):
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
        y_val = model(x_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

# how many did it get right?
correct.item()/len(test_data) # 98.72% correct


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
    new_prediction = model(test_data[4143][0].view(1, 1, 28, 28)) # batch size of 1, 1 color channel, 28x28 image
    
cnn_output = torch.max(new_prediction, 1)[1] # it's a 9!!


# save the network for future reference
torch.save(model.state_dict(), 'saved_nns/first_tutorial_cnn.pt')

#to load the model in later start by initializing a blank one
model_reloaded = ConvolutionalNetwork()

#saving/loading models is done through the internal state dict
model_reloaded.load_state_dict(torch.load('saved_nns/first_tutorial_cnn.pt'))

#can verify that everything loaded in correctly with
model_reloaded.eval()
with torch.no_grad():
    newer_prediction = model_reloaded(test_data[4143][0].view(1, 1, 28, 28)) # batch size of 1, 1 color channel, 28x28 image
    
cnn_reloaded_output = torch.max(newer_prediction, 1)[1] # it's still a 9!!

    