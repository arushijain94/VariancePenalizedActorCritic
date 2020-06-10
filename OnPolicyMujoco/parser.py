import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default='../VPACResults/PPOResults', type=str, help='Directory name')
parser.add_argument("--maxiter", default=5000, type=int, help='Max iterations to process')

save_dir = "../VPACResults/Performance"
os.makedirs(save_dir, exist_ok=True)

args = parser.parse_args()
print('Directory being parsed:', args.dir)

experiments = os.listdir(args.dir)
for exp in experiments:
    runs = os.path.join(args.dir, exp)
    results = []
    for run in os.listdir(runs):
        print('Run being processed:', run)
        r = os.path.join(runs, run)
        files = os.listdir(r)
        ind = 0
        if len(files) > 1:
            for i, f in enumerate(files):
                print(i, f)
            ind = int(input('More than one eventlog file available, process which one?'))
        f = os.path.join(r, files[ind])

        rewards = np.zeros((args.maxiter))
        for e in tf.train.summary_iterator(f):
            for v in e.summary.value:
                if v.tag == 'total_reward':
                    if e.step < args.maxiter:
                        rewards[e.step] = v.simple_value

        results.append(rewards)
    results = np.array(results)
    np.save(os.path.join(save_dir, exp + '.npy'), results)
