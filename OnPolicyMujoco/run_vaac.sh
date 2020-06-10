#!/bin/bash
# 'HalfCheetah-v2' 'Hopper-v2' 'Walker2d-v2'
# Run for all 20 seeds in 20 parallel loops. Later combine results to get average performance.

# Env ,              Iterations
# 'HalfCheetah-v2',  1500
# 'Hopper-v2',       1500
# 'Walker2d-v2',     2500

envs=('HalfCheetah-v2')
iters=1500
mus=(0.15)
path_name="./"
file_name="train_tamar.py"

for env in "${envs[@]}"
do
	for run in {1..20}
	do
		for mu in "${mus[@]}"
		do
		  python $path_name$file_name --ENV=$env --name=$run --max_iterations=$iters --mu=$mu
		done
	done
done
