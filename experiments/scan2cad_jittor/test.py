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
from dataset import test_data_loader
from model import create_model
from loss import OverallLoss, Evaluator

jt.flags.use_cuda = 1
#jt.flags.use_cuda_managed_allocator = 1
#jt.flags.lazy_execution=0
#jt.cudnn.set_max_workspace_ratio(0.0)

def test(config,model,evaluator, loss_func, train_loader):
    model.eval()
    mean_pre=0
    mean_recall=0
    count=0
    print(len(train_loader))
    for i, data_dict in enumerate(train_loader):
        print(str(i))
        data_dict['features'] = data_dict['features'].squeeze(0)
        data_dict['transform'] = data_dict['transform'].squeeze(0)
        for ii in range(len(data_dict['lengths'])):
            data_dict['lengths'][ii]=data_dict['lengths'][ii].squeeze(0)
            data_dict['points'][ii]=data_dict['points'][ii].squeeze(0)
            data_dict['neighbors'][ii]=data_dict['neighbors'][ii].squeeze(0)
            data_dict['subsampling'][ii]=data_dict['subsampling'][ii].squeeze(0)
            data_dict['upsampling'][ii]=data_dict['upsampling'][ii].squeeze(0)
        """ ref_length_c = data_dict['lengths'][-1][0].item()
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
            continue """
        try:
            start_time=time.time() 
            output_dict = model(data_dict)
            loading_time = time.time() - start_time
            result_dict=evaluator(output_dict, data_dict)

            mean_pre+=result_dict['precision']
            mean_recall+=result_dict['recall']
            print('precision',100*mean_pre/(count+1))
            print('recall',100*mean_recall/(count+1))    
            print('time',loading_time)
            count+=1
            jt.gc()
        except Exception as inst:
            jt.gc()  
            print(inst)
        

    

def main(args):
    print('main')
    config = make_cfg()
    test_loader = test_data_loader(config)
    model = create_model(config)
    evaluator=Evaluator(config)
    network_label='MIRETR'
    save_filename = ('%s_net_%s.pkl' % (args.epochs, network_label))
    save_path = os.path.join(config.snapshot_dir, save_filename)
    loss_func = OverallLoss(config)
    model.load(save_path)
    test(config,model, evaluator,loss_func,test_loader)

        
        





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--epochs', type=int, default=44, metavar='N',
                        help='number of episode to train ')
    args = parser.parse_args()
    main(args)
