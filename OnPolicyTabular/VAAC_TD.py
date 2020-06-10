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


class StateActionJLearning:
    def __init__(self, gamma, lr, weights):
        self.lr = lr
        self.gamma = gamma
        self.weights = weights

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


class StateActionMLearning:
    def __init__(self, gamma, lr, weights):
        self.lr = lr
        self.gamma = gamma
        self.weights = weights

    def start(self, phi, action):
        self.last_phi = phi
        self.last_action = action
        self.last_value = self.value(phi, action)

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def update(self, phi, action, reward, critic_J, done):
        update_target = pow(reward, 2.)
        if not done:
            current_value = self.value(phi, action)
            update_target += (pow(self.gamma, 2) * current_value) + (2 * self.gamma * reward * critic_J)

        # Weight gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_action] += self.lr * tderror
        if not done:
            self.last_value = current_value
        self.last_action = action
        self.last_phi = phi


class PolicyGradient:
    def __init__(self, policy, lr, mu, gamma):
        self.lr = lr
        self.policy = policy
        self.mu = mu
        self.gamma = gamma

    def update(self, phi, action, critic_J, critic_M, G, J_0, I_J, I_M):
        actions_pmf = self.policy.pmf(phi)
        if self.mu != 0.0:  # variance as regularization factor to optimization criterion
            regularization = -self.lr * self.mu * (
                    (I_M * critic_M) + (2 * self.gamma * I_J * G * critic_J) - (
                    2 * I_J * J_0 * critic_J))
            self.policy.weights[phi, :] -= regularization * actions_pmf
            self.policy.weights[phi, action] += regularization

        Q_val = self.lr * I_J * critic_J
        self.policy.weights[phi, :] -= Q_val * actions_pmf
        self.policy.weights[phi, action] += Q_val


class OutputInformation:
    def __init__(self):
        # storage the weights of the trained model
        self.weight_policy = []
        self.weight_Q = []
        self.weight_var = []
        self.history = []


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
        csvHeader = ['runs', 'episodes', 'temp', 'lrP', 'lrJ', 'lrV', 'mu',
                     'mean', 'std', 'step']
        csvData.append(csvHeader)
    data_row = [args.nruns, args.nepisodes, args.temperature, args.lr_P, args.lr_J, args.lr_M,
                args.mu, mean_return, std_return, steps]
    csvData.append(data_row)
    with open(file_name, style) as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(csvData)
    csvFile.close()


def run_agent(outputinfo, features, nepisodes, frozen_states, nfeatures, nactions, num_states, temperature, gamma, lr_J, lr_M, lr_theta, mu, rng):
    history = np.zeros((nepisodes, 3), dtype=np.float32)  # 1. Return 2. Steps 3. TD error 1 norm
    # storage the weights of the trained model
    weight_policy = np.zeros((nepisodes, num_states, nactions),
                             dtype=np.float32)

    # Using Softmax Policy
    policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)

    # Action_critic is Q value of state-action pair
    weights_J = np.random.rand(nfeatures, nactions)
    action_critic_J = StateActionJLearning(gamma, lr_J, weights_J)

    # Variance is sigma of state-action pair
    weights_M = np.random.rand(nfeatures, nactions)
    action_critic_M = StateActionMLearning(gamma, lr_M, weights_M)

    # Policy gradient improvement step
    policy_improvement = PolicyGradient(policy, lr_theta, mu, gamma)
    env = gym.make('Fourrooms-v0')
    for episode in range(args.nepisodes):
        return_per_episode = 0
        observation = env.reset()
        phi = features(observation)
        action = policy.sample(phi)
        action_critic_J.start(phi, action)
        action_critic_M.start(phi, action)

        step = 0
        done = False
        sum_td_error = 0.0
        I_J = 1.
        I_M = 1.
        G = 0.
        initial_state = phi
        while done != True:
            old_observation = observation
            old_phi = phi
            old_action = action
            observation, reward, done, _ = env.step(action)

            if old_observation in frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)

            phi = features(observation)
            return_per_episode += pow(args.gamma, step) * reward
            action = policy.sample(phi)

            # Critic update
            action_critic_J.update(phi, action, reward, done)
            critic_J_value = action_critic_J.value(phi, action)
            if mu != 0.0:
                action_critic_M.update(phi, action, reward, critic_J_value, done)
                action_critic_M_value = action_critic_M.value(old_phi, old_action)
                initial_state_J_value = np.dot(policy.pmf(initial_state), action_critic_J.value(initial_state))
            else:
                action_critic_M_value = 0.
                initial_state_J_value = 0.
            policy_improvement.update(old_phi, old_action, action_critic_J.value(old_phi, old_action),
                                      action_critic_M_value, G,
                                      initial_state_J_value, I_J, I_M)

            step += 1
            I_J *= gamma
            I_M *= pow(gamma, 2.)
            G += I_M * reward

        history[episode, 0] = step
        history[episode, 1] = return_per_episode
        history[episode, 2] = sum_td_error
        weight_policy[episode, :, :] = policy.weights

    outputinfo.weight_policy.append(weight_policy)
    outputinfo.history.append(history)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_J', help="Learning rate for J", type=float, default=0.05)
    parser.add_argument('--lr_P', help="Learning rate for Policy", type=float, default=0.001)
    parser.add_argument('--lr_M', help="Learning rate for M", type=float, default=0.02)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=0.05)
    parser.add_argument('--mu', help="Mu regularizer for Variance in return", type=float, default=0.0)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=1000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=100)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=10)

    args = parser.parse_args()
    now_time = datetime.datetime.now()

    env = gym.make('Fourrooms-v0')
    outer_dir = "../../Results/Results_FR"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "FourRoomVarianceTamarTD")
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Psi" + str(args.mu) + \
               "_LRJ" + str(args.lr_J) + "_LRP" + str(args.lr_P) + "_LRM" + str(args.lr_M) + \
               "_temp" + str(args.temperature)

    dir_name += "_seed" + str(args.seed)
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
                                                     args.lr_M, args.lr_P, args.mu,
                                                     np.random.RandomState(args.seed + i),))
        threads.append(t)
        t.start()

    for x in threads:
        x.join()

    hist = np.asarray(outputinfo.history)
    last_meanreturn = np.round(np.mean(hist[:, :-10, 1]), 2)  # Last 100 episodes mean value of the return
    last_stdreturn = np.round(np.std(hist[:, :-10, 1]), 2)  # Last 100 episodes std. dev value of the return
    last_steps = np.round(np.std(hist[:, :-10, 0]), 2)  # Last 100 episodes std. dev value of the steps

    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), np.asarray(outputinfo.weight_policy))
    np.save(os.path.join(dir_name, 'History.npy'), np.asarray(outputinfo.history))
    save_csv(args, os.path.join(outer_dir, "ParamtersDone.csv"), last_meanreturn, last_stdreturn, last_steps)
