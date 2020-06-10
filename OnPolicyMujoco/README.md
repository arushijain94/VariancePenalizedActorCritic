# On-policy continuous state-action MuJoCo Experiments

## Prerequisites
The following file will install all the dependencies. Suggestion: create a virtual environment with python3.7.4 and then
 follow the below installation instructions.
```
Requires Mujoco (can use free student trial as well)
run: echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path_mujoco_binaries>/mujoco200/bin' >> ~/.bashrc
Add Mujoco license file in bin folder above. 
pip install -r requirements.txt
```
 
 ## How to run code?
 We generate results on 20 random seeds. The following code would launch 20 different codes. 
 To aggregate the results follow the directions in next section below.
 
 ### VPAC Algorithm
```
bash run_vpac.sh
```
 To run on more random seeds, increase "run" in above bash file. Details about how many iterations or used mean-variance tradeoff parameter is provided in above bash file. Further hyper-parameters are provided in "hyper_params_vpac.py".
 
 ### PPO Baseline
```
bash run_vaac.sh
```
 To run on more random seeds, increase "run" in above bash file. Details about how many iterations is added in above bash file.
  Further hyper-parameters are provided in "hyper_params.py". Set "mus" variable to 0 in order to run PPO.

### VAAC Baseline
```
bash run_vaac.sh
```
 To run on more random seeds, increase "run" in above bash file. Details about how many iterations is added in above bash file.
  Further hyper-parameters are provided in "hyper_params.py". We set "mus" variable to 0.15 in order to run VAAC. 
  The "mus" variable provide mean-variance tradeoff.
 
## To aggregate and visualize the performance across multiple runs
 
 * VPAC results :  Set dir="VPACResults/PPOResults", save_dir="VPACResults/Performance"
 * PPO results :  Set dir="BaselineResults/PPOResults", save_dir="PPOBaseline/Performance"
 
 PPO results are stored in BaselineResults with $\psi=0$. On the other hand, VAAC baseline results are stored in
  "BaselineResults" with $\psi!=0$. The below python file will aggregate the return from 20 random seeds in one .npy file. Now run the below command:
```
python parser.py
```
To visualize the performance, after aggregating the results from multiple runs, use "RewardPerformance.ipynb" jupyter-notebook to plot the learning curves.
  
  
## For analysis of mean-variance performance  

For each converged policy (20 random seeds leads to 20 policies for each algorithm), we rolled out
 100 trajectories to calculate the mean and variance in return. Use the following command to generate the sampled trajectories:
 
 * VPAC
```
bash run_trajs.sh
```
 * Baseline: PPO and VAAC
```
bash run_trajs_baseline.sh
```
Change envs=("Hopper-v2") to generate trajectories for Hopper environment. Similarly, for Walker environment, set envs=("Walker2d-v2") and iters=2500 in the above bash file.

Followed by the above command, run the below file to aggregate the results and generate a pickle file.
```
python aggregate_return_dist.py
```
Change main_dir = "BaselineResults" for creating pickle of return distribution for baselines - PPO and VAAC. We used the
 above pickle file to generate the box plots for the variance distribution across the multiple runs.


