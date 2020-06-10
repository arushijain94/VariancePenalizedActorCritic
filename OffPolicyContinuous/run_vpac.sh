#!/bin/bash
Lr_p=(0.1)
Lr_c=(0.5)
Lr_sigma=(0.25)
lam=1.0
temp=50
Psi=(0.0 0.001 0.005)
eps=4000
path_name="./"
file_name="VPAC.py"

for lr_p in "${Lr_p[@]}"; do
  for lr_c in "${Lr_c[@]}"; do
    for psi in "${Psi[@]}"; do
      for lr_sigma in "${Lr_sigma[@]}"; do
        for run in {1..50}; do
          nohup python $path_name$file_name --temperature $temp  --lr_theta $lr_p --lr_critic $lr_c --lr_sigma $lr_sigma --lmbda $lam --nepisodes $eps --nruns $run --psi $psi --seed $run &
        done
      done
    done
  done
done

