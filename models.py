#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:01:06 2022

@author: yu
"""


import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class FC_one(nn.Module):
    def __init__(self,H):
        super(FC_one, self).__init__()
        
        self.feature = [[],[],[]]
        self.fc1 = nn.Linear(784,H,bias = False)
        self.fc2 = nn.Linear(H,10,bias = False)

        
    def forward(self,x):
        layer0_output = x.view(x.shape[0],-1)
        layer1_output = F.relu(self.fc1(layer0_output))
        
        output = self.fc2(layer1_output)
        
        self.feature[0] = layer0_output.detach()
        self.feature[1] = layer1_output.detach()
        self.feature[2] = output.detach()
        
        return output
    
class FC_feature(nn.Module):
    def __init__(self,H, d):
        super(FC_feature,self).__init__()
        
        self.feature = [[],[],[],[]]
        self.fc1 = nn.Linear(784,H,bias = False)
        self.fc2 = nn.Linear(H,H,bias = False)
        self.fc3 = nn.Linear(H,10,bias = False)
        self.dropout = nn.Dropout(d)
        
    def forward(self,x):
        layer0_output = x.view(x.shape[0],-1)
        layer1_output = F.relu(self.fc1(layer0_output))
        
        dropout_output = self.dropout(layer1_output)
        
        layer2_output = F.relu(self.fc2(dropout_output))
        output = self.fc3(layer2_output)
        
        self.feature[0] = layer0_output.detach()
        self.feature[1] = layer1_output.detach()
        self.feature[2] = layer2_output.detach()
        self.feature[3] = output.detach()
        
        return output

class FCmulti_feature(nn.Module):
    def __init__(self,H):
        super(FCmulti_feature, self).__init__()
        
        self.feature = [[],[],[],[],[],[]]
        self.fc1 = nn.Linear(784,H,bias = False)
        self.fc2 = nn.Linear(H,H,bias = False)
        self.fc3 = nn.Linear(H,H,bias = False)
        self.fc4 = nn.Linear(H,H,bias = False)
        self.fc5 = nn.Linear(H,10,bias = False)

    def forward(self,x):
        layer0_output = x.view(x.shape[0],-1)
        layer1_output = F.relu(self.fc1(layer0_output))
        layer2_output = F.relu(self.fc2(layer1_output))
        layer3_output = F.relu(self.fc3(layer2_output))
        layer4_output = F.relu(self.fc4(layer3_output))
        output = self.fc5(layer4_output)
        
        self.feature[0] = layer0_output.detach()
        self.feature[1] = layer1_output.detach()
        self.feature[2] = layer2_output.detach()
        self.feature[3] = layer3_output.detach()
        self.feature[4] = layer4_output.detach()
        self.feature[5] = output.detach()
        
        return output
    
class FC(nn.Module):
    def __init__(self,H):
        super(FC,self).__init__()
        self.net = nn.Sequential(
                Flatten(),
                nn.Linear(784,H,bias = True),
                nn.ReLU(),
                nn.Linear(H,H,bias = True),
                nn.ReLU(),
                nn.Linear(H,10,bias = True),
#                nn.Sigmoid()
                )
    def forward(self,x):
        output = self.net(x)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.feature = [[],[],[]]
        self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=16,
                          kernel_size=3,
                          stride=1,
                          padding=2,
                        bias = False),       #output shape(8,28,28)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2) #output shape(8,14,14)
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=16,
                          out_channels=32,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                        bias = False),               #output shape(16,14,14))
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)   #output shape(16,7,7)
                )
        self.out = nn.Linear(32*7*7,10,bias = False)
        
    def forward(self,x):
        self.feature[0] = x
        x = self.conv1(x)
        self.feature[1] = x.detach()
        x = self.conv2(x)
        self.feature[2] = x.detach()
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output