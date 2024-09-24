import argparse
from cmath import isnan
import time
import os
import os.path as osp
from unittest import result
import torch.optim as optim
import jittor as jt
from jittor import nn
from jittor.contrib import concat
from jittor import linalg
import numpy as np
import argparse
from config import make_cfg
from dataset_jittor import test_data_loader
from model_jittor import create_model
from loss_jittor import OverallLoss, Evaluator
import pandas as pd
jt.flags.use_cuda = 1
#jt.flags.use_cuda_managed_allocator = 1
#jt.flags.lazy_execution=0
#jt.cudnn.set_max_workspace_ratio(0.0)


def test(config,model,evaluator, loss_func, train_loader):
    model.eval()
    PIR=0
    IR=0
    RRE=0
    RTE=0
    RMSE=0
    RR=0
    mean_time=0
    jt.sync_all(True)
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
        with jt.profile_scope() as report:
            output_dict = model(data_dict)
        df = pd.DataFrame(report, columns=['Name', 'FileName', 'Count', 'TotalTime', 'AvgTime', 'MinTime', 'MaxTime', 'Input', 'Output', 'InOut', 'Compute'])
        df.to_excel("report.xlsx")
        break
        """ with jt.no_grad():
            for j in range(30):
                output_dict = model(data_dict)
                jt.sync_all()
            jt.sync_all(True)
            start_time=time.time()
            for j in range(30):
                output_dict = model(data_dict)
                jt.sync_all()
            jt.sync_all(True)
            loading_time = time.time() - start_time
            print('time',loading_time/30)
        break """
        """ result_dict=evaluator(output_dict, data_dict)
        loss=loss_func(output_dict, data_dict)

        PIR+=result_dict['PIR']
        IR+=result_dict['IR']
        RRE+=result_dict['RRE']
        RTE+=result_dict['RTE']
        RMSE+=result_dict['RMSE']
        RR+=result_dict['RR']
        mean_time+=loading_time
        print('PIR',PIR/(i+1))
        print('IR',IR/(i+1))
        print('RRE',RRE/(i+1))
        print('RTE',RTE/(i+1))
        print('RMSE',RMSE/(i+1))
        print('RR',RR/(i+1)) """
        
        """ jt.gc()
        jt.sync_all(True) """
        

    

def main(args):
    print('main')
    config = make_cfg()
    test_loader = test_data_loader(config,'3DLoMatch')
    model = create_model(config)
    evaluator=Evaluator(config)
    network_label='Geotransformer'
    save_filename = ('%s_net_%s.pkl' % (args.epochs, network_label))
    save_path = os.path.join(config.snapshot_dir, save_filename)
    loss_func = OverallLoss(config)
    model.load(save_path)
    test(config,model, evaluator,loss_func,test_loader)

        
        





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--epochs', type=int, default=79, metavar='N',
                        help='number of episode to train ')
    args = parser.parse_args()
    main(args)
