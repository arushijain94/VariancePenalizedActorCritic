#!/bin/bash
Lr_p=(0.005)
Lr_c=(0.01)
Temp=(100)
Psi=(0.0015)
eps=2500
path_name="./"
file_name="VAAC.py"


for lr_p in "${Lr_p[@]}"; do
  for lr_c in "${Lr_c[@]}"; do
    for psi in "${Psi[@]}"; do
      for temp in "${Temp[@]}"; do
        for run in {1..100}; do
          nohup python $path_name$file_name --temperature $temp --lr_pol $lr_p --lr_Q $lr_c --lr_M $lr_c --nepisodes $eps --nruns $run --tradeoff $psi --seed $run &
        done
      done
    done
  done
done

