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
from config import make_cfg
from dataset_jittor import train_valid_data_loader
from model_jittor import create_model
from loss_jittor import OverallLoss, Evaluator

jt.flags.use_cuda = 1
#jt.flags.use_cuda_managed_allocator = 1
#jt.flags.lazy_execution=0
jt.cudnn.set_max_workspace_ratio(0.0)


def train(config,model, loss_func,optimizer,scheduler,epoch, train_loader):
    network_label='Geotransformer'
    save_filename = ('%s_net_%s.pkl' % (epoch, network_label))
    save_path = os.path.join(config.snapshot_dir, save_filename)
    model.train()
    jt.sync_all(True)
    for i, data_dict in enumerate(train_loader):
        print('Epoch'+str(epoch)+'/'+str(i))
        data_dict['features'] = data_dict['features'].squeeze(0)
        data_dict['transform'] = data_dict['transform'].squeeze(0)
        for ii in range(len(data_dict['lengths'])):
            data_dict['lengths'][ii]=data_dict['lengths'][ii].squeeze(0)
            data_dict['points'][ii]=data_dict['points'][ii].squeeze(0)
            data_dict['neighbors'][ii]=data_dict['neighbors'][ii].squeeze(0)
            data_dict['subsampling'][ii]=data_dict['subsampling'][ii].squeeze(0)
            data_dict['upsampling'][ii]=data_dict['upsampling'][ii].squeeze(0)
        output_dict = model(data_dict)
        result_dict = loss_func(output_dict, data_dict)
        print(result_dict)
        loss=result_dict['loss']
        optimizer.step(loss)
        jt.gc()
        """ if i%1000==0:
            model.save(save_path) """
    

def main(args):
    print('main')
    config = make_cfg()
    train_loader0,train_loader1 = train_valid_data_loader(config)
    
    model = create_model(config)
    optimizer = jt.optim.Adam(model.parameters(),
                               lr=config.optim.lr ,
                               weight_decay=config.optim.weight_decay)
    loss_func = OverallLoss(config)
    scheduler = jt.lr_scheduler.ExponentialLR(optimizer, config.optim.lr_decay )
    network_label='Geotransformer'
    if args.epochs<1:
        save_filename = ('%s_net_%s.pkl' % (0, network_label))
        save_path = os.path.join(config.snapshot_dir, save_filename)
    else:
        save_filename = ('%s_net_%s.pkl' % (args.epochs-1, network_label))
        save_path = os.path.join(config.snapshot_dir, save_filename)
    model.load(save_path)
    for i in range(int(args.epochs/2)):
        scheduler.step()
    if args.epochs%2==0:
        train(config,model, loss_func,optimizer,scheduler,args.epochs, train_loader0)
    else:
        train(config,model, loss_func,optimizer,scheduler,args.epochs, train_loader1)    
        
        





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of episode to train ')
    args = parser.parse_args()
    main(args)
