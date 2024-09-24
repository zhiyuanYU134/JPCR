#!/bin/bash

for((i=22;i<60;i++))
do
echo $i;
/home/yuzhiyuan/.conda/envs/miretr/bin/python /home/yuzhiyuan/GeoTransformer-main/experiments/graphscnet_jittor/trainval.py --epochs $i;
done


