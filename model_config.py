#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:36:47 2022

@author: yu
"""

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