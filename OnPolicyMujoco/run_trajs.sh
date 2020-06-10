#!/bin/bash

# 'HalfCheetah-v2' 'Hopper-v2' 'Walker2d-v2'
# Running all 20 seeds for one type of environment

envs=('HalfCheetah-v2')
iters=1500
psis=(0.15)
gamma=1.0
path_name="./"
file_name="generate_trajectories.py"

for env in "${envs[@]}"
do
	for run in {1..20}
	do
		for psi in "${psis[@]}"
		do
		  python $path_name$file_name --ENV=$env --name=$run --psi=$psi --mu=$psi --max_iterations=$iters --discount=$gamma
		done
	done
done
