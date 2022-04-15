#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:11:28 2022

@author: yu
"""
import torch
import copy
import numpy as np
import utils

def get_paras_for_given_layer(model,layer):
    paras = list(model.named_parameters())
    name,Para = paras[layer]
    para = copy.deepcopy(Para)
    para = torch.squeeze(para.reshape(1,-1))
    para = para.data.numpy()
    return para

def replace_given_weight(weight,model,layer):
    
    model_new = copy.deepcopy(model)
    weight_new = weight
    paras = model_new.state_dict()
    paras_key = list(paras.keys())
    paras[paras_key[layer]] = weight_new
    model_new.load_state_dict(paras)
    return model_new

def amend_weight(model,weight_old_list,component,amend_amplitude,layer_index):
    new_weight_list = []
    component_list = utils.transform_array_to_matrix(model,layer_index,component.reshape(1,-1))
    for weight_old,component in zip(weight_old_list,component_list):
        weight_change = amend_amplitude*component
        new_weight = weight_old + weight_change
        new_weight_list.append(new_weight)
    return new_weight_list

def move_model_in_weight_space(model,direction,amplitude,layer_index):
    model_new = copy.deepcopy(model)
    weight_list_start = [list(model.parameters())[l] for l in layer_index]
    new_weight_list = amend_weight(model,weight_list_start,direction,amend_amplitude = amplitude,layer_index = layer_index)
    for l in range(len(layer_index)):
        new_weight = new_weight_list[l]
        model_new = replace_given_weight(new_weight,model = model_new,layer = layer_index[l])
    return model_new
        
def loss_change_by_changing_weight(model,weight_list_start,direction,amplitude,data_x,data_y,layer_index):
    model_new = copy.deepcopy(model)
    # weight_list_start = [list(model.parameters())[l] for l in layer_index]
    new_weight_list = amend_weight(model,weight_list_start,direction,amend_amplitude = amplitude,layer_index = layer_index)
    for l in range(len(layer_index)):
        new_weight = new_weight_list[l]
        model_new = replace_given_weight(new_weight,model = model_new,layer = layer_index[l])
    amend_loss = utils.cal_loss(model_new,data_x,data_y)

    return amend_loss

def cal_order_derivative(model,direction,data_x,data_y,layer_index,order,precision):
    
    h = precision
    weight_list_start = [list(model.parameters())[l] for l in layer_index] 
    if order == 1:
        y1 = loss_change_by_changing_weight(model,weight_list_start,direction,amplitude = h,data_x = data_x,data_y = data_y,layer_index = layer_index)
        y2 = loss_change_by_changing_weight(model,weight_list_start,direction,amplitude = 0,data_x = data_x,data_y = data_y,layer_index = layer_index)
        
        derivative = (y1-y2)/h
    return derivative

def find_sample_with_same_index(data_x,data_y,index_list):
    index_holder = []
    for i in range(len(data_y)):
        if data_y[i] in index_list:
            index_holder.append(i)
    select_data_x = data_x[np.array(index_holder),:]
    return select_data_x

def get_hidden_output(model,data_x,layer_index):
    _ = model(data_x)
    feature = model.feature[layer_index[0]]
    return feature

def cal_distance_between_train_and_test(model,layer_index,init_train_x,init_test_x,k_near_num = 1):
    
    train_x = init_train_x
    test_x = init_test_x
    distance_map = np.zeros((len(test_x),len(train_x)))
    smallest_distance_sample_holder = []
    close_train_arg_holder = []
    for i in range(len(test_x)):
        for j in range(len(train_x)):
            distance_map[i,j] = torch.norm(train_x[j]-test_x[i]).data.numpy()
        smallest_distance_arg = np.argsort(distance_map[i,:])[0:k_near_num]
        smallest_distance_sample_holder.append(torch.mean(train_x[smallest_distance_arg],0))
        close_train_arg_holder.append(smallest_distance_arg)

    
    hidden_train_x = get_hidden_output(model,train_x,layer_index)
    hidden_test_x = get_hidden_output(model,test_x,layer_index)
    
    close_train_arg_holder = np.squeeze(np.array(close_train_arg_holder))
    delta_x_holder = np.zeros((len(hidden_test_x),np.prod(hidden_test_x[0].shape)))
    x_holder = np.zeros((len(hidden_test_x),np.prod(hidden_test_x[0].shape)))
    
    
    test_num = len(smallest_distance_sample_holder)
    for i in range(test_num):
        delta_x = (hidden_test_x[i] - hidden_train_x[close_train_arg_holder[i]])
        delta_x_holder[i,:] = delta_x.data.numpy().reshape(1,-1)
        x_holder[i,:] = hidden_test_x[i].data.numpy().reshape(1,-1)
                

    return x_holder,delta_x_holder,close_train_arg_holder

def cal_duality_solution(model,input_test_x,close_delta_x,layer_index):
    
    W = list(model.parameters())[layer_index[0]].data.numpy()
    
    duality_solution = np.zeros(W.shape)
    close_train_x = input_test_x - close_delta_x
    
    Z = np.sum(close_train_x**2)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            duality_solution[i,j] = np.sum(W[i,:]*close_delta_x)*close_train_x[j]/Z
    
    return duality_solution

def pertubation_theory(model, train_x, train_y, test_x, test_y, sample_holder, layer_index):
    
    pertubation_strength_holder = []
    weight_grad_holder = []
    for s in sample_holder:
        
        train_data_x = find_sample_with_same_index(train_x,train_y,[s])
        test_data_x = find_sample_with_same_index(test_x,test_y,[s])
        x,close_delta_x,close_train_arg = cal_distance_between_train_and_test(model,layer_index,train_data_x,test_data_x,k_near_num = 1)
        
        data_num = x.shape[0]
        for k in range(data_num):
            input_x = x[k,:]
            delta_x = close_delta_x[k,:]
            induced_weight_change = cal_duality_solution(model,input_x,delta_x,layer_index).reshape(1,-1)
            pertubation_strength = np.linalg.norm(induced_weight_change)
            pertubation_direction = induced_weight_change/np.linalg.norm(induced_weight_change)
            g_k = cal_order_derivative(model,pertubation_direction,train_data_x[close_train_arg[k]:close_train_arg[k]+1,:],
                                       torch.tensor([s]).type(torch.LongTensor),
                                       layer_index,order = 1,
                                       precision = pertubation_strength)
            pertubation_strength_holder.append(pertubation_strength * pertubation_direction)
            weight_grad_holder.append(g_k * pertubation_direction)
            

            
    pertubation_strength_holder = np.squeeze(np.array(pertubation_strength_holder))
    weight_grad_holder = np.squeeze(np.array(weight_grad_holder))

    
    return weight_grad_holder, pertubation_strength_holder

def estimate_loss_gap(model, components, pertubation_strength_holder, weight_grad_holder):
    
    theta_pertubation = np.dot(pertubation_strength_holder,components.T)
    theta_grad = np.dot(weight_grad_holder,components.T)
    
    sigma_theta = np.sqrt(np.var(theta_pertubation,0))
    sigma_grad = np.sqrt(np.var(theta_grad,0))
    mean_theta = np.mean(theta_pertubation,0)
    mean_grad = np.mean(theta_grad,0)
    c_l = np.mean((theta_grad - mean_grad) * (theta_pertubation - mean_theta),0) / sigma_grad / sigma_theta
    for i in range(len(c_l)):
        if np.isnan(c_l[i]):
            c_l[i] = 0
    return c_l, sigma_theta, sigma_grad, mean_theta, mean_grad