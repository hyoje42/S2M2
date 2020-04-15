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
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.maml import MAML
from io_utils import model_dict, parse_args, get_resume_file  

from os.path import join, dirname
import moco
import torchvision.models as models

params = parse_args('train')
os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu)
use_gpu = torch.cuda.is_available()
image_size = 80

def train(base_loader, val_loader, model, optimization, start_epoch, stop_epoch, params):    
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    elif optimization == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr, 
                                    momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
       raise ValueError('Unknown optimization, please define by yourself')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    max_acc = 0       

    for epoch in range(start_epoch,stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader,  optimizer ) #model are called by reference, no need to return 
        model.eval()

        if not os.path.isdir(params.checkpoint_dir):
            os.makedirs(params.checkpoint_dir)

        acc = model.test_loop( val_loader)
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
    print(params)

    if params.dataset in ['miniImagenet', 'CUB']:
        image_size = 84
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
            if params.dataset in ['omniglot', 'cross_char']:
                params.stop_epoch = 5
            elif params.dataset in ['CUB', 'cifar']:
                params.stop_epoch = 200 # This is different as stated in the open-review paper. However, using 400 epoch in baseline actually lead to over-fitting
            elif params.dataset in ['miniImagenet', 'cross']:
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
    
    model_moco = moco.MoCo(models.__dict__[params.model], 128, 16384)
    
    if params.save_by_others is not None:
        if torch.cuda.is_available():
            model_moco.cuda()
        tmp = torch.load(params.save_by_others)
        state = tmp['state_dict']
        for key in list(state.keys()):
            if 'module.' in key:
                newkey = key.replace('module.', '')
                state[newkey] = state.pop(key)
            else:
                print(key)
        model_moco.load_state_dict(state, strict=False)
        # get bottom of ResNet
        encoder = moco.ResNetBottom(model_moco.encoder_q)

    if params.method in ['baseline', 'baseline++'] :
        base_datamgr    = SimpleDataManager(image_size, batch_size = params.batch_size)
        base_loader     = base_datamgr.get_data_loader( base_file , aug = params.train_aug )
        val_datamgr     = SimpleDataManager(image_size, batch_size = params.batch_size)
        val_loader      = val_datamgr.get_data_loader( val_file, aug = False)
        
        if params.method == 'baseline':
            model           = BaselineTrain( model_dict[params.model], params.num_classes)
        elif params.method == 'baseline++':
            # model           = BaselineTrain( model_dict[params.model], params.num_classes, loss_type = 'dist')
            model           = BaselineTrain( encoder, params.num_classes, loss_type = 'dist')

    elif params.method in ['protonet','matchingnet','relationnet', 'relationnet_softmax', 'maml', 'maml_approx']:
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
        elif params.method == 'matchingnet':
            model           = MatchingNet( model_dict[params.model], **train_few_shot_params )
        elif params.method in ['relationnet', 'relationnet_softmax']:
            if params.model == 'Conv4': 
                feature_model = backbone.Conv4NP
            elif params.model == 'Conv6': 
                feature_model = backbone.Conv6NP
            elif params.model == 'Conv4S': 
                feature_model = backbone.Conv4SNP
            else:
                feature_model = lambda: model_dict[params.model]( flatten = False )
            loss_type = 'mse' if params.method == 'relationnet' else 'softmax'

            model           = RelationNet( feature_model, loss_type = loss_type , **train_few_shot_params )
        elif params.method in ['maml' , 'maml_approx']:
            backbone.ConvBlock.maml = True
            backbone.SimpleBlock.maml = True
            backbone.BottleneckBlock.maml = True
            backbone.ResNet.maml = True
            model           = MAML(  model_dict[params.model], approx = (params.method == 'maml_approx') , **train_few_shot_params )
            if params.dataset in ['omniglot', 'cross_char']: #maml use different parameter in omniglot
                model.n_task     = 32
                model.task_update_num = 1
                model.train_lr = 0.1
    else:
       raise ValueError('Unknown method')

    model.cuda()

    params.checkpoint_dir = f'./checkpoints/moco/{params.method}' + dirname(params.save_by_others).split('/')[-1]
    print(f'Save checkpoint at {params.checkpoint_dir}')

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.method == 'maml' or params.method == 'maml_approx' :
        stop_epoch = params.stop_epoch * model.n_task #maml use multiple tasks in one update 
    
    

    model = train(base_loader, val_loader,  model, optimization, start_epoch, stop_epoch, params)