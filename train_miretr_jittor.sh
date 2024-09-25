#!/bin/bash

for((i=0;i<60;i++))
do
echo $i;
python experiments/scan2cad_jittor/trainval.py --epochs $i;
done


