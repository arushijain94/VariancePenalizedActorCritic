#!/bin/bash
lr_pol=1e-2
lr_J=0.5
temp=1.0
tradeoff=0.01
lr_M=0.5
run=100
eps=1000
seed=10
path_name="./"
file_name="VAAC_TD.py"

python $path_name$file_name --temperature $temp --lr_P $lr_pol --lr_J $lr_J --lr_M $lr_M --mu $tradeoff --nruns $run --nepisodes $eps --seed 10
