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


class EgreedyPolicy:
    def __init__(self, rng, nfeatures, nactions, epsilon):
        self.rng = rng
        self.epsilon = epsilon
        self.nactions = nactions
        self.weights = 0.5 * np.ones((nfeatures, nactions))  # positive weight initialization

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi, curr_epsilon):
        if self.rng.uniform() < curr_epsilon:
            return int(self.rng.randint(self.nactions))
        return int(np.argmax(self.value(phi)))


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


class StateActionLearning:
    def __init__(self, gamma, lr, weights, variance):
        self.lr = lr
        self.gamma = gamma
        self.weights = weights
        self.variance = variance  # binary value (0: its is Q value, 1: sigma(s,a) value)

    def start(self, phi, action):
        self.last_phi = phi
        self.last_action = action
        self.last_value = self.value(phi, action)

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def update(self, phi, action, reward, done):
        update_target = reward
        if not done:
            current_value = self.value(phi, action)
            update_target += self.gamma * current_value
        # Weight gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_action] += self.lr * tderror
        if not done:
            self.last_value = current_value
        self.last_action = action
        self.last_phi = phi
        return tderror


class PolicyGradient:
    def __init__(self, policy, lr, psi, nactions):
        self.lr = lr
        self.policy = policy
        self.psi = psi

    def update(self, phi, action, critic, sigma, current_psi, I_Q, I_sigma):
        actions_pmf = self.policy.pmf(phi)
        if self.psi != 0.0:  # variance as regularization factor to optimization criterion
            psi = current_psi
            regularization = -self.lr * psi * I_sigma * sigma
            self.policy.weights[phi, :] -= regularization * actions_pmf
            self.policy.weights[phi, action] += regularization

        Q_val = self.lr * I_Q * critic
        self.policy.weights[phi, :] -= Q_val * actions_pmf
        self.policy.weights[phi, action] += Q_val


class OutputInformation:
    def __init__(self):
        # storage the weights of the trained model
        self.weight_policy = []
        self.history = []
        self.var = []  # stores variance in return [Var(G_t)]
        self.ret = []  # stores return [G_t]


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


def save_csv(args, file_name, mean_return, std_return, steps):
    csvData = []
    style = 'a'
    if not os.path.exists(file_name):
        style = 'w'
        csvHeader = ['runs', 'episodes', 'temp', 'lr_p', 'lr_c', 'lr_var', 'psi', 'lambda',
                     'mean', 'std', 'step']
        csvData.append(csvHeader)
    data_row = [args.nruns, args.nepisodes, args.temperature, args.lr_theta, args.lr_critic, args.lr_sigma,
                args.psi, args.lmbda, mean_return, std_return, steps]
    csvData.append(data_row)
    with open(file_name, style) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


def var_cal(policy_weight, gamma, num_rollouts, frozen_states):
    env = gym.make('Fourrooms-v0')
    goal_state = 62
    max_time = 300
    disc_return_list = []
    for rollout_ind in range(num_rollouts):
        start = env.reset()
        if start == goal_state:
            continue
        curr_state = start
        curr_time = 0
        d = False
        current_dicounting_factor = 1.
        discounted_reward = 0
        while (curr_state != goal_state and curr_time < max_time and d != True):
            action = np.argmax(policy_weight[curr_state, :])
            next_state, r, d, _ = env.step(action)
            if curr_state in frozen_states:
                r = np.random.normal(0, 8.0)
            discounted_reward += current_dicounting_factor * r
            curr_state = next_state
            current_dicounting_factor *= gamma
            curr_time += 1
        disc_return_list.append(discounted_reward)
    return np.var(disc_return_list)


def run_agent(outputinfo, features, nepisodes, frozen_states, nfeatures, nactions, num_states, temperature, gamma_Q,
              gamma_var, lr_critic, lr_sigma, lr_theta, psi, rng):
    history = np.zeros((nepisodes, 3),
                       dtype=np.float32)  # 1. Return 2. Steps 3. TD error 1 norm
    # storage the weights of the trained model
    weight_policy = np.zeros((nepisodes, num_states, nactions),
                             dtype=np.float32)

    # details for variance calculations
    num_rollouts = 800
    get_variance_after_episode = 50
    list_eps = list(np.arange(0, nepisodes, get_variance_after_episode))
    list_eps.append(nepisodes - 1)
    var_return_list = []

    # Using Softmax Policy
    policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)

    # Action_critic is Q value of state-action pair
    weights_QVal = np.random.rand(nfeatures, nactions)
    action_critic = StateActionLearning(gamma_Q, lr_critic, weights_QVal, 0)

    # Variance is sigma of state-action pair
    weights_var = np.random.rand(nfeatures, nactions)
    sigma = StateActionLearning(gamma_var, lr_sigma, weights_var, 1)
    current_psi = psi

    # Policy gradient improvement step
    policy_improvement = PolicyGradient(policy, lr_theta, psi, nactions)
    env = gym.make('Fourrooms-v0')
    for episode in range(nepisodes):
        return_per_episode = 0
        observation = env.reset()
        phi = features(observation)
        action = policy.sample(phi)
        action_critic.start(phi, action)
        sigma.start(phi, action)

        step = 0
        done = False
        sum_td_error = 0.0
        I_Q = 1.
        I_sigma = 1.
        while done != True and step < 500:
            old_observation = observation
            old_phi = phi
            old_action = action
            observation, reward, done, _ = env.step(action)
            if old_observation in frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)

            phi = features(observation)
            return_per_episode += I_Q * reward
            action = policy.sample(phi)

            # Critic update
            tderror = action_critic.update(phi, action, reward, done)
            sum_td_error += abs(tderror)
            if psi != 0.0:
                try:
                    td_square = pow(tderror, 2.0)
                except OverflowError:
                    td_square = tderror * 2
                sigma.update(phi, action, td_square, done)
                sigma_val = sigma.value(old_phi, old_action)
            else:
                sigma_val = 0.0

            critic_val = action_critic.value(old_phi, old_action)
            policy_improvement.update(old_phi, old_action, critic_val, sigma_val, current_psi, I_Q, I_sigma)

            step += 1
            I_Q *= gamma_Q
            I_sigma *= gamma_var

        if episode in list_eps:
            # calculate variance by rolling out policy
            var_return_list.append(var_cal(policy.weights, gamma_Q, num_rollouts, frozen_states))

        history[episode, 0] = step
        history[episode, 1] = return_per_episode
        history[episode, 2] = sum_td_error
        weight_policy[episode, :, :] = policy.weights

    outputinfo.weight_policy.append(weight_policy)
    outputinfo.history.append(history)
    outputinfo.var.append(np.array(var_return_list))
    outputinfo.ret.append(history[:, 1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lmbda', help='Bias-variance parameter', type=float, default=1.0)
    parser.add_argument('--lr_critic', help="Learning rate for Q value", type=float, default=0.5)
    parser.add_argument('--lr_theta', help="Learning rate for policy parameterization theta", type=float, default=0.01)
    parser.add_argument('--lr_sigma', help="Learning rate for sigma variance of return", type=float, default=0.02)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=0.5)
    parser.add_argument('--psi', help="Psi regularizer for Variance in return", type=float, default=0.0)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=1000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=1)

    args = parser.parse_args()
    env = gym.make('Fourrooms-v0')
    outer_dir = "../../Results/Results_FR"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "FourRoom")
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Psi" + str(args.psi) + \
               "_LRC" + str(args.lr_critic) + "_LRTheta" + str(args.lr_theta) + "_LRV" + str(args.lr_sigma) + \
               "_temp" + str(args.temperature) + "_lambda" + str(args.lmbda)

    dir_name += "_seed" + str(args.seed)
    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)

    num_states = env.observation_space.n
    nactions = env.action_space.n
    gamma_var = pow(args.gamma * args.lmbda, 2.0)
    frozen_states = GetFrozenStates()

    threads = []
    features = Tabular(num_states)
    nfeatures = len(features)
    outputinfo = OutputInformation()

    for i in range(args.nruns):
        t = threading.Thread(target=run_agent, args=(outputinfo, features,
                                                     args.nepisodes,
                                                     frozen_states, nfeatures, nactions, num_states, args.temperature,
                                                     args.gamma, gamma_var, args.lr_critic,
                                                     args.lr_sigma, args.lr_theta, args.psi,
                                                     np.random.RandomState(args.seed + i),))
        threads.append(t)
        t.start()

    for x in threads:
        x.join()

    hist = np.asarray(outputinfo.history)
    last_meanreturn = np.round(np.mean(hist[:, :-20, 1]), 2)  # Last 100 episodes mean value of the return
    last_stdreturn = np.round(np.std(hist[:, :-20, 1]), 2)  # Last 100 episodes std. dev value of the return
    last_steps = np.round(np.mean(hist[:, :-20, 0]), 2)  # Last 100 episodes mean value of the steps
    var_return_array = np.asarray(outputinfo.var)  # run X episode
    return_array = np.asarray(outputinfo.ret)  # run X episode

    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
    np.save(os.path.join(dir_name, 'Var.npy'), var_return_array)
    np.save(os.path.join(dir_name, 'RewardDistributionVarMean.npy'), np.mean(var_return_array, axis=0))
    np.save(os.path.join(dir_name, 'RewardDistributionVarStd.npy'), np.std(var_return_array, axis=0))
    np.save(os.path.join(dir_name, 'RewardDistributionMeanMean.npy'), np.mean(return_array, axis=0))
    np.save(os.path.join(dir_name, 'RewardDistributionMeanStd.npy'), np.std(return_array, axis=0))
    save_csv(args, os.path.join(outer_dir, "ParamtersCorrected.csv"), last_meanreturn, last_stdreturn, last_steps)
