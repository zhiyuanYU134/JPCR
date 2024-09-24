#!/bin/bash

for((i=0;i<60;i++))
do
echo $i;
/home/yuzhiyuan/.conda/envs/miretr/bin/python /home/yuzhiyuan/GeoTransformer-main/experiments/scan2cad_jittor/trainval.py --epochs $i;
done


