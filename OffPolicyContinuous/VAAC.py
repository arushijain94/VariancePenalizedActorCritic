import gym
import argparse
import numpy as np
from puddlesimple import PuddleSimpleEnv
import os
from tiles3 import *


class TileFeature:
    def __init__(self, ntiles, nbins, discrete_states, features_range):
        self.ntiles = ntiles
        self.nbins = nbins
        self.max_discrete_states = discrete_states
        self.iht = IHT(discrete_states)
        self.features_range = features_range
        self.scaling = nbins / features_range

    def __call__(self, input_observation):
        return tiles(self.iht, self.ntiles, input_observation * self.scaling)

    def __len__(self):
        return self.max_discrete_states


class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state, ])

    def __len__(self):
        return self.nstates


class RandomPolicy:
    def __init__(self, nactions, rng):
        self.nactions = nactions
        self.rng = rng
        self.prob_actions = [1. / self.nactions] * self.nactions

    def sample(self):
        return int(self.rng.randint(self.nactions))

    def pmf(self):
        return self.prob_actions


class GreedyPolicy:
    def __init__(self, nactions, weights):
        self.nactions = nactions
        self.weights = weights

    def value(self, phi):
        return np.sum(self.weights[phi, :], axis=0)

    def sample(self, phi):
        return int(np.argmax(self.value(phi)))


class SoftmaxPolicy:
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        self.rng = rng
        self.weights = np.random.rand(nfeatures, nactions)  # positive weight initialization
        self.nactions = nactions
        self.temp = temp

    def value(self, phi):
        return np.sum(self.weights[phi, :], axis=0)

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


def GetFrozenStates():
    layout = """\
wwwwwwwwwwww
w          w
w          w
w          w
w   ffff   w
w   ffff   w
w   ffff   w
w   ffff   w
w          w
w          w
w          w
wwwwwwwwwwww
"""
    num_elem = 12
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


# get importance sampling correction factor
def get_rho(target_policy, behavior_policy, phi, action):
    return min(1, (target_policy.pmf(phi)[int(action)] / behavior_policy.pmf()[int(action)]))


def get_off_policy_corrected_return(reward_list, rho_list, gamma):
    trajectory_length = len(reward_list)
    discounted_return = np.zeros(trajectory_length)
    discounted_return[-1] = rho_list[-1] * reward_list[-1]
    for t in range(trajectory_length - 2, -1, -1):
        discounted_return[t] = gamma * discounted_return[t + 1] + rho_list[t] * reward_list[t]
    return discounted_return


def get_G_bar(reward_list, rho_list, gamma):
    trajectory_length = len(reward_list)
    new_gamma = gamma ** 2
    G_bar = np.zeros(trajectory_length)
    current_gamma = 1.
    for t in range(trajectory_length - 1):
        G_bar[t + 1] = G_bar[t] + current_gamma * rho_list[t] * reward_list[t]
        current_gamma *= new_gamma
    return G_bar


# Generates a randlom reward drawn from normal distribution
def tweak_reward_near_puddle(reward):
    noise_mean = 0.0
    noise_sigma = 8.0
    noise = np.random.normal(noise_mean, noise_sigma)
    return reward + noise


# Checks whether the agent has entered puddle zone
def check_if_agent_near_puddle(observation):
    if (observation[0] <= 0.7 and observation[0] >= 0.3):
        if (observation[1] <= 0.7 and observation[1] >= 0.3):
            return True
    return False


# Get return for target policy : For evaluating the target policy performance
def ReturnTargetPolicy(policy, gamma, features, env):
    observation = env.reset()
    return_value = 0.0
    phi = features(observation)
    done = False
    current_gamma = 1.0
    step = 0.
    while done != True:
        old_observation = observation
        action = policy.sample(phi)
        observation, reward, done, _ = env.step(action)
        if check_if_agent_near_puddle(old_observation):
            reward = tweak_reward_near_puddle(reward)
        return_value += current_gamma * reward
        current_gamma *= gamma
        phi = features(observation)
        step += 1
    return return_value, step


# get variance in return for target policy
def VarReturnTargetPolicy(policy, gamma, features, env):
    num_rollouts = 800
    return_dist = []
    for rollout in range(num_rollouts):
        observation = env.reset()
        return_value = 0.0
        phi = features(observation)
        done = False
        current_gamma = 1.0
        while done != True:
            old_observation = observation
            action = policy.sample(phi)
            observation, reward, done, _ = env.step(action)
            if check_if_agent_near_puddle(old_observation):
                reward = tweak_reward_near_puddle(reward)
            return_value += current_gamma * reward
            current_gamma *= gamma
            phi = features(observation)
        return_dist.append(return_value)
    return np.var(return_dist)


def get_state_action_value(weight, phi, action=None):
    if action is None:
        return np.sum(weight[phi, :], axis=0)
    return np.sum(weight[phi, action], axis=0)


def run_agent(nepisodes, temperature, gamma, lr_Q, lr_M, lr_pol, tradeoff, rng, maxDiscreteStates, ntiles):
    env = gym.make('PuddleEnv-v0')
    env._max_episode_steps = 5000
    features_range = env.observation_space.high - env.observation_space.low
    features = TileFeature(ntiles, 5, maxDiscreteStates, features_range)  # 5X5 tiles over joint space
    nfeatures = int(maxDiscreteStates)
    nactions = env.action_space.n

    history = np.zeros((nepisodes, 2),
                       dtype=np.float32)  # 1. Steps, 2. Return for target policy

    # Random Behavioral policy
    behavior_policy = RandomPolicy(nactions, rng)

    # Target Softmax Policy
    target_policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)

    # Weights of J (state-action value function)
    weight_Q = np.zeros((nfeatures, nactions))
    # Weights of M (state-action reward square function)
    weight_M = np.zeros((nfeatures, nactions))
    V0 = np.zeros(nfeatures)

    get_variance_after_episode = 100
    list_eps = list(np.arange(0, nepisodes, get_variance_after_episode))
    list_eps.append(nepisodes - 1)
    var_return_list = []
    save_index = 0
    weight_policy_store = np.zeros((len(list_eps), nfeatures, nactions),
                                   dtype=np.float32)
    for episode in range(nepisodes):
        # Roll out the trajectory
        phi_list = []
        action_list = []
        reward_list = []
        rho_list = []

        observation = env.reset()
        done = False
        current_rho = 1.
        while not done:
            old_observation = observation
            phi = features(observation)
            action = behavior_policy.sample()
            phi_list.append(phi)
            action_list.append(action)
            current_rho *= get_rho(target_policy, behavior_policy, phi, action)
            rho_list.append(current_rho)
            observation, reward, done, _ = env.step(action)
            if check_if_agent_near_puddle(old_observation):
                reward = tweak_reward_near_puddle(reward)
            reward_list.append(reward)  # discounted reward
        phi_list.append(features(observation))  # last observation
        G = get_off_policy_corrected_return(reward_list, rho_list, gamma)
        G_bar = get_G_bar(reward_list, rho_list, gamma)

        # updates of the weight matrix
        episode_length = len(reward_list)
        temp_weight_Q = np.zeros_like(weight_Q)
        temp_weight_M = np.zeros_like(weight_M)
        temp_weight_policy = np.zeros_like(target_policy.weights)

        # V0[phi_list[0]] += lr_Q * (G[0] - V0[phi_list[0]])  # initial V(x_0) estimate
        V0 = np.dot(target_policy.pmf(phi_list[0]),
                    get_state_action_value(weight_Q, phi_list[0]))  # \sum_a pi(a|s),Q(s,a)
        for t in range(episode_length):
            temp_weight_Q[phi_list[t], action_list[t]] += (lr_Q * G[t]) - (
                    lr_Q * weight_Q[phi_list[t], action_list[t]])  # w_J^{episode}
            if tradeoff > 0.0:
                temp_weight_M[phi_list[t], action_list[t]] += (lr_M * (G[t] ** 2)) - (
                        lr_M * weight_M[phi_list[t], action_list[t]])  # w_M^{episode}

        I_Q = 1.
        I_M = 1.
        new_gamma = gamma ** 2
        for t in range(episode_length):
            # update for policy parameter
            actions_pmf = target_policy.pmf(phi_list[t])
            Q_val = get_state_action_value(weight_Q, phi_list[t], action_list[t])
            M_val = get_state_action_value(weight_M, phi_list[t], action_list[t])
            new_tradeoff = lr_pol * tradeoff
            policy_gradient_update = lr_pol * I_Q * rho_list[t] * Q_val - new_tradeoff * (
                    I_M * rho_list[t] * M_val + 2 * gamma * G_bar[t] * I_Q * rho_list[t] * Q_val - 2 * V0 * I_Q *
                    rho_list[t] * Q_val)
            temp_weight_policy[phi_list[t], :] -= policy_gradient_update * actions_pmf
            temp_weight_policy[phi_list[t], action_list[t]] += policy_gradient_update
            I_Q *= gamma
            I_M *= new_gamma
        target_policy.weights += temp_weight_policy
        weight_Q += temp_weight_Q
        weight_M += temp_weight_M

        return_target, step_target = ReturnTargetPolicy(target_policy, gamma, features, env)
        if episode in list_eps:
            weight_policy_store[save_index] = target_policy.weights
            var_return_list.append(
                VarReturnTargetPolicy(target_policy, gamma, features, env))
            save_index += 1

        history[episode, 0] = return_target
        history[episode, 1] = step_target

    return history, np.asarray(var_return_list), weight_policy_store


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lr_Q', help="Learning rate for Q value", type=float, default=0.5)
    parser.add_argument('--lr_M', help="Learning rate for M value", type=float, default=0.5)
    parser.add_argument('--lr_pol', help="Learning rate for policy", type=float, default=0.05)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=100.0)
    parser.add_argument('--tradeoff', help="Tradeoff parameter for mean-variance obj", type=float, default=0.0)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=1000)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=2)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=1)
    parser.add_argument('--maxDiscreteStates', help="discrete states", type=int, default=1024)
    parser.add_argument('--ntiles', help="tiles", type=int, default=10)

    args = parser.parse_args()
    outer_dir = "../../Results/Results_PuddleCont"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "PuddleContTamar", "LRP" + str(args.lr_pol) + "_LRC" + str(args.lr_Q),
                             "Psi" + str(args.tradeoff))
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_mu" + str(args.tradeoff) + \
               "_LRQ" + str(args.lr_Q) + "_LRM" + str(args.lr_M) + "_LRP" + str(args.lr_pol) + \
               "_gamma" + str(args.gamma) + "_temp" + str(args.temperature) + "_seed" + str(args.seed)

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)
    history, var_return, weight_policy = run_agent(args.nepisodes, args.temperature,
                                                   args.gamma, args.lr_Q,
                                                   args.lr_M, args.lr_pol, args.tradeoff,
                                                   np.random.RandomState(args.seed),
                                                   args.maxDiscreteStates, args.ntiles)

    np.save(os.path.join(dir_name, 'History.npy'), history)
    np.save(os.path.join(dir_name, 'Mean.npy'), history[:, 0])
    np.save(os.path.join(dir_name, 'Step.npy'), history[:, 1])
    np.save(os.path.join(dir_name, 'Var.npy'), var_return)
    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), weight_policy)
