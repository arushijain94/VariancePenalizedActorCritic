# VariancePenalizedActorCritic (VPAC)
Here we run VPAC algorithm and baselines VAAC, VAAC_TD and vanilla AC. We use the following environments to run both on- and off-policy algorithm. See the given directories to run algorithms.

* **OnPolicyTabular**: Tabular four-rooms experiments for on-policy VPAC
* **OffPolicyTabular**: Tabular puddle-world experiments for off-policy VPAC
* **OffPolicyContinuous**: Continuous state puddle-world experiments for off-policy VPAC
* **OnPolicyMujoco**: Check this folder to see the code and readme file for running Mujoco experiments for on-policy VPAC.

## Prerequisites
Things required for four-rooms and puddle-world experiments.
```
gym
python3.7
seaborn
```

## How to run code?
The best hyper-parameters are already mentioned in the bash files. Use them for the best results.

### On-Policy Tabular (four-rooms)
* **VPAC algorithm**
```
bash run_vpac.sh
```
* **AC algorithm**
```
bash run_vpac.sh
```
Set Psis=(0.0) to get results for conventional objective for maximizing just the mean performance.
* **VAAC algorithm** (Tamar et al. 2013 baseline with MC critic)
```
bash run_vaac.sh
```
* **VAAC_TD algorithm**  (Tamar et al. 2013 baseline with TD critic)
```
bash run_vaac_td.sh
```
For results visualization see: **OnPolicyTabular.ipynb**

### Off-Policy Tabular (puddle-world)
* **VPAC algorithm**
```
bash run_vpac.sh
```
* **AC algorithm**
```
bash run_vpac.sh
```
in above bash file, set Psis=(0.0) to get results for AC.
* **VAAC algorithm** (added importance sampling in Tamar et al. 2013 with MC critic)
```
bash run_vaac.sh
```
For results visualization see: **OffPolicyTabular.ipynb**

### Off-Policy Continuous (puddle-world)
* **VPAC algorithm**
```
bash run_vpac.sh
```
* **AC algorithm**
```
bash run_vpac.sh
```
in above bash file, set Psis=(0.0) to get results for AC.
* **VAAC algorithm** (added importance sampling in Tamar et al. 2013 with MC critic)
```
bash run_vaac.sh
```
For results visualization see: **OffPolicyContinuous.ipynb**


