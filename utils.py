#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:22:15 2022

@author: yu
"""

import torch
import numpy as np
import torch.nn as nn
import copy
import torch.utils.data as Data
import train_model

def transform_matrix_to_array(para_list):
    para_holder = []
    for para in para_list:
        para_holder.append(para.data.clone().numpy().reshape(1,-1))
    para_array = np.hstack(para_holder)
    return para_array

def transform_array_to_matrix(model,layer_index,para_array):
    para_list = []
    start_point = 0
    for i in layer_index:
        weight = list(model.parameters())[i]
        num_weight = np.prod(weight.shape)
        para_matrix = para_array[0][start_point:num_weight+start_point].reshape(weight.shape)
        para_list.append(torch.tensor(para_matrix))
        start_point += num_weight
    return para_list

def replace_weight(model,weight_list):
    paras_dict = model.state_dict()
    paras_key = list(paras_dict.keys())
    for i in range(len(paras_key)):
        key_name = paras_key[i]
        paras_dict[key_name] = weight_list[i]
    model.load_state_dict(paras_dict)
    return model

def init_weight(model,scale_ratio):
    weight_list = list(model.parameters())
    scaled_weight_list = scale_weight_list(weight_list,scale_ratio)
    model = replace_weight(model,scaled_weight_list)
    return model

def scale_weight_list(weight_list,scale_ratio):
    scaled_weight_list = []
    for i in range(len(weight_list)):
        scaled_weight_list.append(weight_list[i]*scale_ratio)
    return scaled_weight_list

def predict_accuracy(model,data_x,data_y):
    
    pred = torch.max(model(data_x),1)[1].data.numpy()
    accuracy = np.mean(pred == data_y.data.numpy())
    
    return accuracy

def cal_loss(model,data_x,data_y):
    
    loss_func = nn.CrossEntropyLoss()
    out_put = model(data_x)
    loss = loss_func(out_put,data_y)
    
    return loss.data.numpy()

def cal_grad_for_given_coordinate(model,components,data_x,data_y,layer_index):
    
    copy_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(copy_model.parameters(),lr=0.1)
    loss_func = nn.CrossEntropyLoss()
    out_put = copy_model(data_x)
    l = loss_func(out_put,data_y)
    optimizer.zero_grad()
    l.backward()
    grads_matrix = [list(copy_model.parameters())[l].grad.detach() for l in layer_index]
    grads = transform_matrix_to_array(grads_matrix)
    new_grads = np.dot(grads,components.T)
    
    return new_grads    


def cal_grad(model,data_x,data_y,layer_index):
    
    copy_model = copy.deepcopy(model)
    optimizer = torch.optim.SGD(copy_model.parameters(),lr=0.1)
    loss_func = nn.CrossEntropyLoss()
    out_put = copy_model(data_x)
    l = loss_func(out_put,data_y)
    optimizer.zero_grad()
    l.backward()
    grads_matrix = [list(copy_model.parameters())[l].grad.detach() for l in layer_index]
    grads = transform_matrix_to_array(grads_matrix)
    
    return grads 

def smooth_curve(sequence,window_size):
    ave_sequence = []
    for i in range(len(sequence)-window_size):
        ave_sequence.append(np.mean(sequence[i:i+window_size]))
    return ave_sequence

def regulization(model,threshold):
    
    r = torch.tensor(0.0)
    for weight in model.parameters():
        r += torch.sum(torch.pow(weight,2))
    R = torch.relu(torch.sqrt(r) - threshold)
    
    return R

def get_net_paras(model):
    model_copy = copy.deepcopy(model)
    weight_list = list(model_copy.parameters())
    grad_list = []
    for i in range(len(weight_list)):
        grad_list.append(weight_list[i].grad)
    return weight_list,grad_list

def cal_L2_norm_weight(para_list):
    w = transform_matrix_to_array(para_list)
    norm = np.linalg.norm(w)
    return norm

def cal_L2_norm_model(model):
    weight_list,_ = get_net_paras(model)
    weight_norm = cal_L2_norm_weight(weight_list)
    return weight_norm

def cal_fisher_information(model,data_x,data_y,layer_index):
    from scipy import linalg
    torch_dataset = Data.TensorDataset(data_x,data_y)
    train_loader = Data.DataLoader(torch_dataset, batch_size = 1, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1)
    sample_grad_holder = []
    for step, (b_x,b_y) in enumerate(train_loader):
        optimizer.zero_grad()
        out_put = model(b_x)
        loss = nn.CrossEntropyLoss()(out_put,b_y)
        loss.backward()
        
        layer_weight_grad = []
        for i in layer_index:
            layer_weight_grad.append(list(model.parameters())[i].grad)
        
        
        sample_grad = transform_matrix_to_array(layer_weight_grad)
        sample_grad_holder.append(sample_grad)
    
    fisher_info_matrix = 0
    for grad in sample_grad_holder:
        fisher_info_matrix = fisher_info_matrix + np.dot(grad.T,grad)
    fisher_info_matrix = fisher_info_matrix/len(sample_grad_holder)
    
    v,w = linalg.eig(fisher_info_matrix)
    
    return fisher_info_matrix,np.real(v),np.real(w).T

    
def cal_hessian(model,data_x,data_y, layer_index):
    model = copy.deepcopy(model)
    loss_func = nn.CrossEntropyLoss()
    layer = layer_index[0]
    parameter = list(model.parameters())[layer]
    out_put = model(data_x)
    loss = loss_func(out_put,data_y)
    grad_1 = torch.autograd.grad(loss,parameter,create_graph=True)[0]
    hessian = []
    for grad in grad_1.view(-1):
        grad_2 = torch.autograd.grad(grad,parameter,create_graph=True)[0].view(-1)
        hessian.append(grad_2.data.numpy())
    h = np.array(hessian)
    eigenvalue,eigenvector = np.linalg.eig(h)
    return hessian,np.real(eigenvalue),np.real(eigenvector).T

def get_weight_num(configs, para_name):
    config = copy.deepcopy(configs)
    config[para_name] = config[para_name][0]
    model = train_model.set_model(config)
    num_weight = np.sum([np.prod(list(model.parameters())[l].shape) for l in config['layer_index']])
    return num_weight

def set_config(hyper_parameter):
    config = {}
    config['alpha'] = 0.1
    config['B'] = 25
    config['max_epoch'] = 1000
    config['test_size'] = 1000
    config['train_size'] = 400
    config['rho'] = 0
    config['layer_index'] = [1]
    config['net_size'] = 30
    config['s'] = 1
    config['d'] = 0
    config['beta'] = 0
    config['stop_loss'] = 1e-3
    config['regulization'] = False
    config['sample_holder'] = [0,1,2,3,4,5,6,7,8,9]
    config['layer_index'] = [1]
    config['iter_time'] = 10
    config['evaluation'] = True
    
    if hyper_parameter == 'alpha':
        config['alpha'] = [0.005,0.01,0.02,0.05, 0.1]
    elif hyper_parameter == 'B':
        config['B'] = [25,100,200,400]
    elif hyper_parameter == 's':
        config['s'] = [4,5,6,7]
    elif hyper_parameter == 'd':
        config['d'] = [0, 0.05, 0.1, 0.2, 0.3]
    elif hyper_parameter == 'beta':
        config['beta'] = [0,5e-3,1e-2,2e-2]
        config['regulization'] = True
    elif hyper_parameter == 'rho':
        config['rho'] = [0,0.091,0.13,0.167,0.2]
    elif hyper_parameter == 'train_size':
        config['train_size'] = [400, 800, 1600, 3200]
    else:
        print('please set your hyper-parameter setting in set_config')
    return config
    

def result(para_name, x_variable, dim, P):
    result = {}
    result['para_name'] = para_name
    result['para'] = x_variable
    result['loss0'] = np.zeros([len(x_variable), dim, P])
    result['loss1'] = np.zeros([len(x_variable), dim, P])
    result['sigma_w'] = np.zeros([len(x_variable), dim, P])
    result['sigma_g'] = np.zeros([len(x_variable), dim, P])
    result['mean_w'] = np.zeros([len(x_variable), dim, P])
    result['mean_g'] = np.zeros([len(x_variable), dim, P])
    result['c'] = np.zeros([len(x_variable), dim, P])
    result['hessian'] = np.zeros([len(x_variable), dim, P]);
    result['real_loss_gap'] = np.zeros([len(x_variable), P])
    result['estimate_loss_gap'] = np.zeros([len(x_variable), P])
    result['error_gap'] = np.zeros([len(x_variable), P])
    result['L2'] = np.zeros([len(x_variable), P])
    return result