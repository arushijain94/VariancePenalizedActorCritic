#!/bin/bash
lr_pol=1e-3
lr_J=0.05
temp=1
tradeoff=0.01
lr_M=0.005
run=100
eps=6000
path_name="./"
file_name="VAAC.py"

python $path_name$file_name --temperature $temp --lr_pol $lr_pol --lr_J $lr_J --lr_M $lr_M --tradeoff $tradeoff --nruns $run --nepisodes $eps --seed 50
