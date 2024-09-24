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
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss#, Evaluator

jt.flags.use_cuda = 1
#jt.flags.use_cuda_managed_allocator = 1
#jt.flags.lazy_execution=0
jt.cudnn.set_max_workspace_ratio(0.0)


def train(config,model, loss_func,optimizer,scheduler,epoch, train_loader):
    network_label='MIRETR'
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
        
        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        if len(ref_points_c)>3000:
            jt.gc()
            continue
        if len(src_points_f)<=64:
            print(len(src_points_f))
            print(len(src_points_c))
            jt.gc()
            continue
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
    train_loader = train_valid_data_loader(config)
    
    model = create_model(config)
    optimizer = jt.optim.Adam(model.parameters(),
                               lr=config.optim.lr ,
                               weight_decay=config.optim.weight_decay)
    loss_func = OverallLoss(config)
    scheduler = jt.lr_scheduler.ExponentialLR(optimizer, config.optim.lr_decay )
    network_label='MIRETR'
    if args.epochs<1:
        save_filename = ('%s_net_%s.pkl' % (0, network_label))
        save_path = os.path.join(config.snapshot_dir, save_filename)
    else:
        save_filename = ('%s_net_%s.pkl' % (args.epochs-1, network_label))
        save_path = os.path.join(config.snapshot_dir, save_filename)
        model.load(save_path)
    for i in range(int(args.epochs)):
        scheduler.step()
    train(config,model, loss_func,optimizer,scheduler,args.epochs, train_loader)

        
        





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    args = parser.parse_args()
    main(args)

