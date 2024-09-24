#!/bin/bash

for((i=0;i<80;i++))
do
echo $i;
/home/yzy/anaconda3/envs/p37-10/bin/python /home/yzy/Desktop/GeoTransformer-main/experiments/3dmatch/trainval_jittor.py --epochs $i;
done


