# define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 3)
        
        self.maxpool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(512*4*4, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        
        
    def forward(self, x):
        # Define the feedforward behavior of this model
        ## x is the input image
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
#         print(x.shape) 
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.maxpool(F.relu(self.conv4(x)))
        x = self.dropout3(x)
        x = self.maxpool(F.relu(self.conv5(x)))
        x = self.dropout4(x)
#         print(x.shape)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        
        return x
