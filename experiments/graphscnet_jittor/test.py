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
from dataset import test_data_loader
from loss import EvalFunction, LossFunction
from model import create_model
jt.flags.use_cuda = 1
#jt.flags.use_cuda_managed_allocator = 1
#jt.flags.lazy_execution=0
jt.cudnn.set_max_workspace_ratio(0.0)


def test(config,model, loss_func,eval_func,epoch, train_loader):
    network_label='GraspSCNet'
    save_filename = ('%s_net_%s.pkl' % (epoch, network_label))
    save_path = os.path.join(config.exp.snapshot_dir, save_filename)
    model.eval()
    jt.sync_all(True)
    print(len(train_loader))
    precision=0
    recall=0
    hit_ratio=0
    NFMR=0
    coverage=0
    EPE = 0
    AccS = 0
    AccR = 0
    OR = 0
    
    for i, data_dict in enumerate(train_loader):
        src_corr_points = data_dict["src_corr_points"].reshape(-1,3)   # (C,)
        num_correspondences = src_corr_points.shape[0]
        data_dict["registration"]=True
        output_dict = model(data_dict)
        result_dict = eval_func(data_dict, output_dict)
        precision+=result_dict['precision']
        recall+=result_dict['recall']
        hit_ratio+=result_dict['hit_ratio']
        coverage+=result_dict['coverage']
        NFMR+=result_dict['NFMR']
        EPE+=result_dict['EPE']
        AccS+=result_dict['AccS']
        AccR+=result_dict['AccR']
        OR+=result_dict['OR']
        print('precision', precision*100/(i+1), 'recall', recall*100/(i+1), 
              'AccS', AccS*100/(i+1),
                'AccR', AccR*100/(i+1), 
                'OR', OR*100/(i+1))
        
        #precision jt.Var([80.923874], dtype=float32) recall jt.Var([100.], dtype=float32) hit_ratio jt.Var([99.98959], dtype=float32) NFMR jt.Var([80.165695], dtype=float32) coverage jt.Var([81.73629], dtype=float32)
        #precision jt.Var([82.865776], dtype=float32) recall jt.Var([100.], dtype=float32) hit_ratio jt.Var([99.9903], dtype=float32) NFMR jt.Var([82.443726], dtype=float32) coverage jt.Var([83.88303], dtype=float32)
    jt.sync_all(True)
    jt.gc()
    

def main(args):
    print('main')
    benchmark='4DMatch-F'
    config = make_cfg()
    test_loader1 = test_data_loader(config,benchmark)
    
    model = create_model(config)
    loss_func = LossFunction(config)
    evaluation_func=EvalFunction(config)
    network_label='GraspSCNet'

    save_filename = ('%s_net_%s.pkl' % (args.epochs, network_label))
    save_path = os.path.join(config.exp.snapshot_dir, save_filename)
    print('load_pretrain_model:',save_path)
    model.load(save_path)
  
 
    test(config,model, loss_func,evaluation_func,args.epochs, test_loader1)

        
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--epochs', type=int, default=59, metavar='N',
                        help='number of episode to train ')
    args = parser.parse_args()
    main(args)
