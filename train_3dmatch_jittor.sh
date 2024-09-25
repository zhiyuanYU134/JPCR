#!/bin/bash

for((i=0;i<80;i++))
do
echo $i;
python experiments/3dmatch/trainval_jittor.py --epochs $i;
done


