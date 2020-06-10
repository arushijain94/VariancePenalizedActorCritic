#!/bin/bash
Lr_p=(0.05)
Lr_c=(0.5)
Lr_sigma=(0.25)
temp=100.0
Psi=(0.002)
eps=2500
path_name="./"
file_name="VPAC.py"

for run in {1..100}; do
	for lr_p in "${Lr_p[@]}"; do
		for lr_c in "${Lr_c[@]}"; do
			for psi in "${Psi[@]}"; do
        for lr_sigma in "${Lr_sigma[@]}"; do
          nohup python $path_name$file_name --temperature $temp --lr_theta $lr_p --lr_critic $lr_c --lr_sigma $lr_sigma --nepisodes $eps --nruns $run --psi $psi --seed $run &
        done
			done
		done
	done
done
