import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.optim
import json
import torch.utils.data.sampler
import os
import glob
import random
import time

import configs
import backbone
import data.feature_loader as feat_loader
from data.datamgr import SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file, get_best_file , get_assigned_file

import os.path as osp

def feature_evaluation(cl_data_file, model, n_way = 5, n_support = 5, n_query = 15, adaptation = False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list,n_way)
    z_all  = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append( [ np.squeeze( img_feat[perm_ids[i]]) for i in range(n_support+n_query) ] )     # stack each batch

    z_all = torch.from_numpy(np.array(z_all) )
   
    model.n_query = n_query
    if adaptation:
        scores  = model.set_forward_adaptation(z_all, is_feature = True)
    else:
        scores  = model.set_forward(z_all, is_feature = True)
    scores = scores.unsqueeze(0)
    acc = []
    for each_score in scores:
        # each_score: (2975*5)
        pred = each_score.data.cpu().numpy().argmax(axis = 1)
        y = np.repeat(range( n_way ), n_query )
        acc.append(np.mean(pred == y)*100 )
    return acc



if __name__ == '__main__':
    params = parse_args('test')
    print(params)

    acc_all = []

    if params.dataset == 'CUB':
        iter_num = 600
    else:
        iter_num = 10000
    iter_num = 1000
    print(f'The number of epiosdes : {iter_num}')

    few_shot_params = dict(n_way = params.test_n_way , n_support = params.n_shot) 

    model = ProtoNet( model_dict[params.model], **few_shot_params )

    if torch.cuda.is_available():
        model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' %(configs.save_dir, params.dataset, params.model, params.method)

    split = params.split
    if params.save_iter != -1:
        split_str = split + "_" +str(params.save_iter)
    else:
        split_str = split
    novel_file = osp.join( checkpoint_dir.replace("checkpoints","features/S2M2"), split_str +".hdf5")
    if params.save_by_others is not None: 
        novel_file = '_'.join(osp.split(params.save_by_others)).replace('checkpoints', 'features') + '/novel.hdf5'
    cl_data_file = feat_loader.init_loader(novel_file)
        
    acc_all1, acc_all2 , acc_all3 = [],[],[]

    if params.dataset == 'CUB':
        n_query = 15
    else:
        n_query = 600 - params.n_shot
    print(novel_file)
    print("evaluating over %d examples"%(n_query))

    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query = n_query , adaptation = params.adaptation, **few_shot_params)
            
        acc_all1.append(acc[0])
        print("%d steps reached and the mean acc is %g"%(i, np.mean(np.array(acc_all1)) ))
        acc_all  = np.asarray(acc_all)
    acc_mean1 = np.mean(acc_all1)
    acc_std1  = np.std(acc_all1)

    if 'moco' in params.method:
        is_adapt = {True:'(adapt)', False:''}
        filename = f'results/(proto){is_adapt[params.adaptation]}' + osp.dirname(novel_file).split('/')[-1] + '_log.txt'
    else:
        filename = f'results/{params.dataset}_{params.method}_log.txt'
    print(f'Save as {filename}')
    f = open(filename, 'w')
    print(params, file=f)
    print(novel_file, file=f)
    print(f'The number of episodes : {iter_num}', file=f)
    print('%d Test Acc at 100= %4.2f%% +- %4.2f%%' %(iter_num, acc_mean1, 1.96* acc_std1/np.sqrt(iter_num)), file=f)
    f.close()
