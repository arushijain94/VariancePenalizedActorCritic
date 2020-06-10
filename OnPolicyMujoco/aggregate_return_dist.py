import os
import numpy as np
import pickle

def aggregate_return_results(dir, save_dir, name_read_file):
    env_folder = os.listdir(dir)
    name_return_dict = {}
    for env in env_folder:
        return_val = []
        for run in os.listdir(os.path.join(dir,env)):
            dist = np.load(os.path.join(dir, env, run,name_read_file))
            return_val.append(dist)

        name_return_dict[env] = return_val
    with open(os.path.join(save_dir, name_read_file), 'wb') as handle:
        pickle.dump(name_return_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


main_dir = "VPACResults"
# main_dir = "BaselineResults"
dir = os.path.join(main_dir, "ReturnDistFixedSeed")
save_dir = os.path.join(main_dir, "ReturnDistAggregatedFixedSeed")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

aggregate_return_results(dir, save_dir, "return.npy")
