#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree.
from __future__ import print_function

import argparse
import csv
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from data.datamgr import SetDataManager
import configs
# import models
from methods.baselinetrain import BaselineTrain
from methods.protonet import ProtoNet
# from methods.baselinefinetune import BaselineFinetune
import wrn_mixup_model
from io_utils import model_dict, parse_args, get_resume_file ,get_assigned_file
from os.path import basename

params = parse_args('train')
os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)
use_gpu = torch.cuda.is_available()

def train_meta_learning(base_loader, val_loader, model, start_epoch, stop_epoch, params):    
    SAVE_PATH = params.checkpoint_dir.replace(basename(params.checkpoint_dir), 
                                              f'ProtoNet_from_{params.method}_{params.opt}_lr{params.lr}')
    print(f'Save Path : {SAVE_PATH}')

    if params.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif params.opt == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('Unknown optimization, please define by yourself')
    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        acc = model.test_loop( val_loader)
        if acc > max_acc : #for baseline and baseline++, we don't use validation here so we let acc = -1
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(SAVE_PATH, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
            print(f'Best Acc: {max_acc}')

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(SAVE_PATH, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
            print(f'Current Acc: {acc} Best Acc: {max_acc}')

    return model

if __name__ == '__main__':
    # params is also defined above
    params = parse_args('train')
    print(params)
    params.dataset = 'cifar'
    image_size = 32

    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'
    params.checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    n_query = max(1, int(16* params.test_n_way/params.train_n_way)) 
    
    train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
    base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
    base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        
    test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
    val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
    val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 

    if params.method == 'manifold_mixup':
        model = wrn_mixup_model.wrn28_10(64)
    elif params.method == 'S2M2_R':
        model = ProtoNet(model_dict[params.model], params.train_n_way, params.n_shot)
    elif params.method == 'rotation':
        model = BaselineTrain( model_dict[params.model], 64, loss_type = 'dist')
 
            
    
    if params.method =='S2M2_R':

        if use_gpu:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))  
            model.cuda()

        if params.resume:
            resume_file = get_resume_file(params.checkpoint_dir )        
            print("resume_file" , resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            print("restored epoch is" , tmp['epoch'])
            state = tmp['state']        
            model.load_state_dict(state)        

        else:
            resume_file   = get_resume_file(params.checkpoint_dir)
            # resume_file = './checkpoints/cifar/ProtoNet_from_S2M2_R_SGD_lr0.0001/best_model.tar'
            print("resume_file" , resume_file)
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch']+1
            print("restored epoch is" , tmp['epoch'])
            state = tmp['state']

            for key in list(state.keys()):
                if 'linear' in key:
                    state.pop(key)
                else:
                    newkey = key.replace('module', 'feature')
                    state[newkey] = state.pop(key)

            model.load_state_dict(state)        
    
        model = train_meta_learning(base_loader, val_loader, model, start_epoch, start_epoch+stop_epoch, params)