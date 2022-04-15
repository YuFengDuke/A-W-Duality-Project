#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 19:39:32 2020

@author: yf96
"""


from scipy import io
import copy

import utils
import train_model
import eval_model
import model_config



para_name = 'B'
configs = model_config.set_config(para_name)
num_weight = utils.get_weight_num(configs, para_name)
result = utils.result(para_name, configs[para_name], num_weight, configs['iter_time'])



for i,hyper_para in enumerate(configs[para_name]):
    for k in range(configs['iter_time']):
        config = copy.deepcopy(configs)
        config[para_name] = hyper_para
        config['train_size'] = int( config['train_size'] / ( 1 - config['rho'] ) )
        model, correct_train_x, correct_train_y, test_x, test_y = train_model.train(config)
        if config['evaluation'] == True:
            result = eval_model.eval_model(model = model,
                                config = config,
                                correct_train_x = correct_train_x,
                                correct_train_y = correct_train_y,
                                test_x = test_x,
                                test_y = test_y,
                                result = result,
                                index = (i, k, num_weight))
           

pre_text = 'MNIST_' + para_name
io.savemat('./Visualization/{}_result.mat'.format(pre_text),{'{}_result'.format(pre_text):result})

