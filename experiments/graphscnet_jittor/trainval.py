import argparse
from cmath import isnan
import time
import os
import os.path as osp
import torch.optim as optim
import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor import linalg
import numpy as np
import argparse
# isort: split
from config import make_cfg
from dataset import train_valid_data_loader
from loss import EvalFunction, LossFunction
from model import create_model
jt.flags.use_cuda = 1
#jt.flags.use_cuda_managed_allocator = 1
#jt.flags.lazy_execution=0
jt.cudnn.set_max_workspace_ratio(0.0)


def train(config,model, loss_func,optimizer,scheduler,epoch, train_loader):
    network_label='GraspSCNet'
    save_filename = ('%s_net_%s.pkl' % (epoch, network_label))
    save_path = os.path.join(config.exp.snapshot_dir, save_filename)
    model.train()
    jt.sync_all(True)
    print(len(train_loader))
    for i, data_dict in enumerate(train_loader):
        src_corr_points = data_dict["src_corr_points"].reshape(-1,3)   # (C,)
        num_correspondences = src_corr_points.shape[0]
        print('Epoch'+str(epoch)+'/'+str(i),num_correspondences)
        
        
        output_dict = model(data_dict)
        result_dict = loss_func(output_dict, data_dict)
        print(result_dict)
        if jt.isnan(result_dict['loss']):
            break
        optimizer.step(result_dict['loss'])
        jt.gc()
        if i%100==0:
            model.save(save_path)
    model.save(save_path)
    jt.sync_all(True)
    jt.gc()
    

def main(args):
    print('main')
    config = make_cfg()
    train_loader1,train_loader2 = train_valid_data_loader(config)
    
    model = create_model(config)
    optimizer = jt.optim.Adam(model.parameters(),
                               lr=config.optimizer.lr ,
                               weight_decay=config.optimizer.weight_decay)
    loss_func = LossFunction(config)
    scheduler = jt.lr_scheduler.ExponentialLR(optimizer, config.optimizer.lr)
    network_label='GraspSCNet'
    if args.epochs<1:
        save_filename = ('%s_net_%s.pkl' % (0, network_label))
        save_path = os.path.join(config.exp.snapshot_dir, save_filename)
    else:
        save_filename = ('%s_net_%s.pkl' % (args.epochs-1, network_label))
        save_path = os.path.join(config.exp.snapshot_dir, save_filename)
        print('load_pretrain_model:',save_path)
        model.load(save_path)
    for i in range(int(args.epochs)):
        scheduler.step()
    if args.epochs%2==0:
        train_loader=train_loader1
    else:
        train_loader=train_loader2
    train(config,model, loss_func,optimizer,scheduler,args.epochs, train_loader)

        
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    args = parser.parse_args()
    main(args)
