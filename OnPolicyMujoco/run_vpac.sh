#!/bin/bash
# 'HalfCheetah-v2' 'Hopper-v2' 'Walker2d-v2'
# Run for all 20 seeds in 20 parallel loops. Later combine results to get average performance.

# Env ,              Iterations,  Psi
# 'HalfCheetah-v2',  1500,        0.15
# 'Hopper-v2',       1500,        0.2
# 'Walker2d-v2',     2500,        0.15

envs=('HalfCheetah-v2')
iters=1500
psis=(0.15)
path_name="./"
file_name="train_ppo.py"

for env in "${envs[@]}"
do
	for run in {1..20}
	do
		for psi in "${psis[@]}"
		do
		  python $path_name$file_name --ENV=$env --name=$run --max_iterations=$iters --psi=$psi
		done
	done
done
