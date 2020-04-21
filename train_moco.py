import numpy as np
import torch
import torch.optim
import os

import configs
import backbone
from data.datamgr import SimpleDataManager, SetDataManager
from methods.baselinetrain import BaselineTrain
from methods.baselinefinetune import BaselineFinetune
from methods.protonet import ProtoNet
from io_utils import model_dict, parse_args, get_resume_file  

from os.path import join, dirname
import moco
import torchvision.models as models

from datetime import datetime

params = parse_args('train')
os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)
use_gpu = torch.cuda.is_available()

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif optimization == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, 
                                    momentum=0.9, weight_decay=1e-4)
    else:
       raise ValueError('Unknown optimization, please define by yourself')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    max_acc = 0       
    for epoch in range(start_epoch,stop_epoch):
        model.train()
        msg = model.train_loop(epoch, base_loader,  optimizer, lr_scheduler=None) #model are called by reference, no need to return 
        model.eval()

        acc = model.test_loop( val_loader)
        with open(params.logfile, 'a') as f:
            print(msg, file=f)
        
        if acc > max_acc : #for baseline and baseline++, we don't use validation here so we let acc = -1
            print("best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

        if (epoch % params.save_freq==0) or (epoch==stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)

    return model

if __name__=='__main__':
    # params is also defined above
    params = parse_args('train')

    if params.dataset in ['miniImagenet', 'CUB']:
        image_size = 224
    elif params.dataset in ['cifar']:
        image_size = 32
    else:
        raise ValueError('Wrong Dataset!')

    base_file = configs.data_dir[params.dataset] + 'base.json'
    val_file = configs.data_dir[params.dataset] + 'val.json'
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    optimization = params.opt

    if params.stop_epoch == -1: 
        if params.method in ['baseline', 'baseline++'] :
            if params.dataset in ['CUB', 'cifar']:
                params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet']:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 400 #default
        else: #meta-learning methods
            if params.n_shot == 1:
                params.stop_epoch = 600
            elif params.n_shot == 5:
                params.stop_epoch = 400
            else:
                params.stop_epoch = 600 #default
    if 'localhost' not in params.dist_url:
       params.dist_url = f'tcp://localhost:1{int(params.dist_url):04d}' 
    torch.distributed.init_process_group(backend='nccl', init_method=params.dist_url, 
                                         world_size=1, rank=0)

    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = params.batch_size)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug, dataset_type='moco')
        val_datamgr     = SimpleDataManager(image_size, batch_size = params.batch_size)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False, dataset_type='moco')
        
        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            model = moco.BaselineForMoCo(models.__dict__[params.model], dim=params.dim, K=16384, mlp=True, 
                                         num_class=200, loss_type='dist')

    elif params.method in ['protonet']:
        n_query = max(1, int(16* params.test_n_way/params.train_n_way)) #if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
 
        train_few_shot_params    = dict(n_way = params.train_n_way, n_support = params.n_shot) 
        base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
        base_loader             = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
         
        test_few_shot_params     = dict(n_way = params.test_n_way, n_support = params.n_shot) 
        val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
        val_loader              = val_datamgr.get_data_loader( val_file, aug = False) 
        #a batch for SetDataManager: a [n_way, n_support + n_query, dim, w, h] tensor        

        if params.method == 'protonet':
            model           = ProtoNet( encoder, **train_few_shot_params )
    else:
       raise ValueError('Unknown method')

    torch.cuda.set_device(0)
    model.cuda(0)

    params.checkpoint_dir = (f'./checkpoints/moco/moco_{params.method}_{params.opt}lr{params.lr}_dim{params.dim}')
    print(f'Save checkpoint at {params.checkpoint_dir}')

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # log file
    filename = os.path.join(params.checkpoint_dir, 'log.txt')
    params.logfile = filename
    with open(params.logfile, 'w') as f:
        for key in params.__dict__.keys():
            print(f'{key} : {params.__dict__[key]}', file=f)
        print(f'Start time : {datetime.now():%Y-%m-%d} {datetime.now():%H:%M:%S}', file=f)
    print(params)

    model = train(base_loader, val_loader,  model, optimization, 
                  params.start_epoch, params.stop_epoch, params)
                  
    with open(params.logfile, 'a') as f:
        print(f'End time : {datetime.now():%Y-%m-%d} {datetime.now():%H:%M:%S}', file=f)              
