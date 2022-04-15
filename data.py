#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:04:40 2022

@author: yu
"""

import torch
import numpy as np
import torchvision

def sub_set_task(label_list,sample_number):
    
    download_mnist = True
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',    
        train=True,  
        transform = torchvision.transforms.ToTensor(),                                                      
        download = download_mnist,
    )
    test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
    
    
    Train_x = train_data.data.type(torch.FloatTensor).view(-1,28*28)/255.
    Train_y = train_data.targets
    Test_x = test_data.test_data.type(torch.FloatTensor).view(-1,28*28)/255.   # shape from (2000, 28, 28) to (2000, 784), value in range(0,1)
    Test_y = test_data.targets
    
    Train_x,Train_y = shuffle_data_set(Train_x,Train_y)
    train_index = []
#    for k in label_list:
    count = 0
    for i in range(len(Train_x)):
        if Train_y[i] in label_list:
            train_index.append(i)
            count = count+1
        if count >= sample_number*len(label_list):
            break
    
    test_index = []
    for i in range(len(Test_x)):
        if Test_y[i] in label_list:
            test_index.append(i)

    
    
    Train_y = np.array(Train_y)
    train_x = Train_x[np.array(train_index),:]
    train_y = Train_y[np.array(train_index)]
    
    train_x = torch.tensor(train_x).type(torch.FloatTensor)
    train_y = torch.tensor(train_y).type(torch.LongTensor)

    Test_y = np.array(Test_y)
    test_x = Test_x[np.array(test_index),:]
    test_y = Test_y[np.array(test_index)]
    
    test_x = torch.tensor(test_x).type(torch.FloatTensor)
    test_y = torch.tensor(test_y).type(torch.LongTensor)
    
    train_x = train_x.view(-1,1,28,28)
    test_x = test_x.view(-1,1,28,28)
    
    return train_x,train_y,test_x,test_y

def shuffle_data_set(data_x,data_y):
    np.random.seed(1)
    index = np.arange(0,len(data_y))
    index = np.random.permutation(index)
    data_x = data_x[index,:]
    data_y = data_y[index]
    return data_x,data_y

def mislabel(train_x,train_y,mis_label_prob):
    
    index = np.arange(0,len(train_y))
    
    train_y = train_y[index]
    mis_train_x = train_x[index,:]
    
    mis_range = int(len(train_y)*(1-mis_label_prob))
    mis_index = np.arange(mis_range,len(train_y))
    permute_index =mis_index[np.random.permutation(len(mis_index))]
    
    index = np.arange(0,len(train_y))
    index[mis_index] = index[permute_index]
    
    mis_train_y = train_y[index]
    
    for i in mis_index:
        if (mis_train_y[i] == train_y[i]):
            mis_train_y[i] = (mis_train_y[i]+np.random.randint(1,10))%10
    
    identity = np.hstack((np.ones(len(train_y)-mis_range),np.zeros(mis_range)))
    return mis_train_x,mis_train_y,identity
