#!/bin/bash
Lr_p=(0.005)
Lr_q=(0.05)
temp=50
Psi=(0.0015)
eps=4000
path_name="./"
file_name="VAAC.py"

for lr_p in "${Lr_p[@]}"; do
  for lr_q in "${Lr_q[@]}"; do
    for psi in "${Psi[@]}"; do
      for run in {1..50}; do
        nohup python $path_name$file_name --temperature $temp --lr_pol $lr_p --lr_Q $lr_q --lr_M $lr_q --nepisodes $eps --nruns $run --tradeoff $psi --seed $run &
      done
    done
  done
done

