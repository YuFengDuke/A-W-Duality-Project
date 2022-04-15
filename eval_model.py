#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 01:45:03 2022

@author: yu
"""

import numpy as np

import utils
import AWD


def eval_model(model, config, correct_train_x, correct_train_y, test_x, test_y, result, index):
    i, k, num_weight = index
    model.eval()
    test_accuracy = utils.predict_accuracy(model,data_x = test_x,data_y = test_y)
    trainLoss = utils.cal_loss(model, correct_train_x, correct_train_y)
    testLoss = utils.cal_loss(model, test_x, test_y)
    l_gap = testLoss - trainLoss
    
    l2 = utils.cal_L2_norm_weight(list(model.parameters())[config['layer_index'][0]])


    matrix,v,components = utils.cal_hessian(model,
                                      data_x = correct_train_x,
                                      data_y = correct_train_y,
                                      layer_index = config['layer_index'])
    
    weight_grad_holder, pertubation_strength_holder = AWD.pertubation_theory(model = model,
                                                                             train_x = correct_train_x,
                                                                             train_y = correct_train_y,
                                                                             test_x = test_x,
                                                                             test_y = test_y,
                                                                             sample_holder = config['sample_holder'],
                                                                             layer_index = config['layer_index'])
    
    c_l, sigma_theta, sigma_grad, mean_theta, mean_grad = AWD.estimate_loss_gap(model = model, 
                                                                                components = components, 
                                                                                pertubation_strength_holder = pertubation_strength_holder, 
                                                                                weight_grad_holder = weight_grad_holder)

    l0 = mean_grad * mean_theta
    l1 = sigma_grad * sigma_theta * c_l
    estim_gap = np.sum(l1) + np.sum(l0)
    
    print(k, 'th', 'Simulation on ', 
          result['para_name'], '=', result['para'][i],
          '| loss gap is ', l_gap, 
          '| loss gap estimated by A-W duality is', estim_gap)
    
    arg = np.argsort(sigma_grad)[::-1]

    
    result['loss0'][i,:num_weight,k] = l0[arg]
    result['loss1'][i,:num_weight,k] = l1[arg]
    result['sigma_w'][i,:num_weight,k] = sigma_theta[arg]
    result['sigma_g'][i,:num_weight,k] = sigma_grad[arg]
    result['mean_w'][i,:,k] = mean_theta[arg]
    result['mean_g'][i,:,k] = mean_grad[arg]
    result['c'][i,:num_weight,k] = c_l[arg]
    result['hessian'][i,:num_weight,k] = v[arg]
    result['real_loss_gap'][i,k] = l_gap
    result['estimate_loss_gap'][i,k] = estim_gap
    result['error_gap'][i,k] = 1 - test_accuracy
    result['L2'][i,k] = l2
    return result