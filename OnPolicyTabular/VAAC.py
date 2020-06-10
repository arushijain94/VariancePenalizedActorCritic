import gym
import argparse
import numpy as np
from fourrooms import Fourrooms
import math
import os
import datetime
import threading
import csv


class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state, ])

    def __len__(self):
        return self.nstates


class RandomPolicy:
    def __init__(self, nactions):
        self.nactions = nactions

    def sample(self):
        return int(np.random.randint(self.nactions))

    def pmf(self):
        prob_actions = [1. / self.nactions] * self.nactions
        return prob_actions


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.random.rand(nfeatures, nactions)  # positive weight initialization
        self.nactions = nactions
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi) / self.temp
        b = v.max()
        new_values = v - b
        y = np.exp(new_values)
        return (y / y.sum())

    def sample(self, phi):
        prob = self.pmf(phi)
        if prob.sum() > 1:
            ind_max = np.argmax(prob)
            prob = np.zeros(self.nactions)
            prob[ind_max] = 1.
        else:
            prob[-1] = 1 - np.sum(prob[:-1])
        return int(self.rng.choice(self.nactions, p=prob))


class OutputInformation:
    def __init__(self):
        # storage the weights of the trained model
        self.weight_policy = []
        self.history = []
        self.sum_weight_J0 = []
        self.sum_weight_J = []
        self.sum_weight_M = []


def GetFrozenStates():
    layout = """\
wwwwwwwwwwwww
w     w     w
w   ffwff   w
w  fffffff  w
w   ffwff   w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""

    num_elem = 13
    frozen_states = []
    state_num = 0
    for line in layout.splitlines():
        for i in range(num_elem):
            if line[i] == "f":
                frozen_states.append(state_num)
            if line[i] != "w":
                state_num += 1
    return frozen_states


def save_params(args, dir_name):
    f = os.path.join(dir_name, "Params.txt")
    with open(f, "w") as f_w:
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False


def save_csv(args, file_name, mean_return, std_return, mean_step):
    csvData = []
    style = 'a'
    if not os.path.exists(file_name):
        style = 'w'
        csvHeader = ['runs', 'episodes', 'temp', 'lr_p', 'lr_j', 'lr_m', 'gamma', 'tradeoff', 'mean', 'std', 'm_step']
        csvData.append(csvHeader)
    data_row = [args.nruns, args.nepisodes, args.temperature, args.lr_pol, args.lr_J, args.lr_M,
                args.gamma, args.tradeoff, mean_return, std_return, mean_step]
    csvData.append(data_row)
    with open(file_name, style) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


def run_agent(outputinfo, features, nepisodes,
              frozen_states, nfeatures, nactions, num_states,
              temperature, gamma, lr_J, lr_M, lr_pol, tradeoff, rng):
    history = np.zeros((nepisodes, 3),
                       dtype=np.float32)  # 1. Steps, 2. Return 3. Sum of Reward
    # storage the weights of the trained model
    weight_policy_store = np.zeros((nepisodes, num_states, nactions),
                                   dtype=np.float32)
    sum_weight_J0 = np.zeros(nepisodes)
    sum_weight_M = np.zeros(nepisodes)
    sum_weight_J = np.zeros(nepisodes)

    # Using Softmax Policy
    policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)

    # Here, we refer to Q matrix as J. It is state-action value function.
    weight_J = np.random.rand(nfeatures, nactions)
    # Weights of M (state-action reward square function)
    weight_M = np.random.rand(nfeatures, nactions)
    # J0 = V(S_0) learns value of initial state distribution
    J0 = np.zeros(nfeatures)

    env = gym.make('Fourrooms-v0')
    for episode in range(nepisodes):
        # Roll out the trajectory
        phi_list = []
        action_list = []
        reward_list = []
        gamma_list = []

        observation = env.reset()
        done = False
        current_gamma = 1.
        while not done:
            old_observation = observation
            phi = features(observation)
            action = policy.sample(phi)
            phi_list.append(phi)
            action_list.append(action)
            observation, reward, done, _ = env.step(action)
            if old_observation in frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)
            reward_list.append(reward)  # discounted reward
            gamma_list.append(current_gamma)
            current_gamma *= gamma
        phi_list.append(features(observation))  # last observation
        return_val = np.dot(np.array(reward_list), np.array(gamma_list))

        # updates of the weight matrix
        episode_length = len(reward_list)
        temp_weight_J = np.zeros_like(weight_J)
        temp_weight_M = np.zeros_like(weight_M)
        temp_weight_policy = np.zeros_like(policy.weights)

        J0[phi_list[0]] += lr_J * (return_val - J0[phi_list[0]])  # initial J(x_0) estimate
        for t in range(episode_length):
            new_reward = np.array(reward_list[t:])
            reward_j = np.dot(new_reward,
                              np.array(
                                  gamma_list[0:new_reward.shape[0]]))  # r_{t} + gamma * r_{t+1} + gamma^2 * r_{t+2}...
            reward_m = reward_j ** 2.0  # (r_{t} + gamma * r_{t+1} + gamma^2 * r_{t+2}...)^2
            temp_weight_J[phi_list[t], action_list[t]] += (lr_J * reward_j) - (
                    lr_J * weight_J[phi_list[t], action_list[t]])  # w_J^{episode}
            if tradeoff > 0.0:
                temp_weight_M[phi_list[t], action_list[t]] += lr_M * reward_m - (
                        lr_M * weight_M[phi_list[t], action_list[t]])  # w_M^{episode}

        for t in range(episode_length):
            # update for policy parameter
            actions_pmf = policy.pmf(phi_list[t])
            new_tradeoff = lr_pol * tradeoff
            if t > 0:
                new_reward = np.array(reward_list[0:t - 1])
                new_gamma = np.array(gamma_list[0:new_reward.shape[0]]) ** 2.0
                new_reward = np.dot(new_reward, new_gamma)
            else:
                new_reward = 0.
            policy_gradient_update = (pow(gamma, t) * lr_pol * weight_J[phi_list[t], action_list[t]]) - (
                        pow(gamma, 2 * t) * new_tradeoff * weight_M[
                    phi_list[t], action_list[t]]) - (2 * pow(gamma, t + 1) * new_tradeoff * new_reward * weight_J[
                phi_list[t], action_list[t]]) + (
                                             pow(gamma, t) * new_tradeoff * 2 * J0[phi_list[0]] * weight_J[
                                         phi_list[t], action_list[t]])
            temp_weight_policy[phi_list[t], :] -= policy_gradient_update * actions_pmf
            temp_weight_policy[phi_list[t], action_list[t]] += policy_gradient_update
        policy.weights += temp_weight_policy

        weight_J += temp_weight_J
        weight_M += temp_weight_M

        history[episode, 0] = episode_length
        history[episode, 1] = return_val
        history[episode, 2] = sum(reward_list)
        weight_policy_store[episode] = policy.weights
        sum_weight_J0[episode] = np.sum(np.absolute(J0))
        sum_weight_M[episode] = np.sum(np.absolute(weight_M))
        sum_weight_J[episode] = np.sum(np.absolute(weight_J))

    outputinfo.weight_policy.append(weight_policy_store)
    outputinfo.sum_weight_J0.append(sum_weight_J0)
    outputinfo.sum_weight_J.append(sum_weight_J)
    outputinfo.sum_weight_M.append(sum_weight_M)
    outputinfo.history.append(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_J', help="Learning rate for J value", type=float, default=0.05)
    parser.add_argument('--lr_M', help="Learning rate for M value", type=float, default=0.05)
    parser.add_argument('--lr_pol', help="Learning rate for policy", type=float, default=0.001)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=1.0)
    parser.add_argument('--tradeoff', help="Tradeoff parameter for mean-variance obj", type=float, default=0.0)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=500)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=20)

    args = parser.parse_args()

    env = gym.make('Fourrooms-v0')
    outer_dir = "../../Results/Results_FR"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "FourRoomVarianceAdjustedAC")
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_mu" + str(args.tradeoff) + \
               "_LRJ" + str(args.lr_J) + "_LRM" + str(args.lr_M) + "_LRP" + str(args.lr_pol) + \
               "_gamma" + str(args.gamma) + "_temp" + str(args.temperature) + "_seed" + str(args.seed)

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)

    num_states = env.observation_space.n
    nactions = env.action_space.n
    frozen_states = GetFrozenStates()

    threads = []
    features = Tabular(num_states)
    nfeatures = len(features)
    outputinfo = OutputInformation()
    for i in range(args.nruns):
        t = threading.Thread(target=run_agent, args=(outputinfo, features,
                                                     args.nepisodes,
                                                     frozen_states, nfeatures, nactions, num_states, args.temperature,
                                                     args.gamma, args.lr_J,
                                                     args.lr_M, args.lr_pol, args.tradeoff,
                                                     np.random.RandomState(args.seed + i),))
        threads.append(t)
        t.start()

    for x in threads:
        x.join()

    hist = np.asarray(outputinfo.history)
    last_meanreturn = np.round(np.mean(hist[:, :-20, 1]), 2)  # Last 100 episodes mean value of the return
    last_stdreturn = np.round(np.std(hist[:, :-20, 1]), 2)  # Last 100 episodes std. dev value of the return
    last_meanstep = np.round(np.mean(hist[:, :-20, 0]), 2)  # Last 100 episodes Steps mean

    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'Weights_J0.npy'), np.asarray(outputinfo.sum_weight_J0))
    np.save(os.path.join(dir_name, 'Weights_J.npy'), np.asarray(outputinfo.sum_weight_J))
    np.save(os.path.join(dir_name, 'Weights_M.npy'), np.asarray(outputinfo.sum_weight_M))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
    save_csv(args, os.path.join(outer_dir, "ParamtersDone.csv"), last_meanreturn, last_stdreturn, last_meanstep)
