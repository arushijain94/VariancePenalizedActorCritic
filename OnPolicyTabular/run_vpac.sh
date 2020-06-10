#!/bin/bash
lr_p=0.01
lr_c=0.5
lr_sigma=0.5
lam=1.0
temp=1.0
Psi=(0.0 0.015)
seed=10
run=100
eps=1000
path_name="./"
file_name="VPAC.py"

for psi in "${Psi[@]}"; do
  nohup python $path_name$file_name --temperature $temp --lr_theta $lr_p --lr_critic $lr_c --lr_sigma $lr_sigma --lmbda $lam --nepisodes $eps --nruns $run --psi $psi --seed $seed &
done
