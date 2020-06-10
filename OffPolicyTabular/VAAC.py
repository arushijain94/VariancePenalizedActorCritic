import gym
import argparse
import numpy as np
from PuddleDiscrete import PuddleD
import os


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


# Get return for target policy
def ReturnTargetPolicy(weights, gamma_Q, frozen_states, features, nactions):
    env = gym.make('Puddle-v1')
    env._max_episode_steps = 500
    policy = GreedyPolicy(nactions, weights)
    observation = env.reset()
    return_value = 0.0
    phi = features(observation)
    done = False
    current_gamma = 1.0
    step = 0.
    while done != True and step < 500:
        old_observation = observation
        action = policy.sample(phi)
        observation, reward, done, _ = env.step(action)
        if old_observation in frozen_states:
            reward = np.random.normal(loc=0.0, scale=8.0)
        return_value += current_gamma * reward
        current_gamma *= gamma_Q
        phi = features(observation)
        step += 1
    return return_value, step


# get variance in return for target policy
def VarReturnTargetPolicy(weights, gamma_Q, frozen_states, features, nactions):
    num_rollouts = 800
    env = gym.make('Puddle-v1')
    env._max_episode_steps = 500
    policy = GreedyPolicy(nactions, weights)
    return_dist = []
    for rollout in range(num_rollouts):
        observation = env.reset()
        return_value = 0.0
        phi = features(observation)
        done = False
        current_gamma = 1.0
        step = 0
        while done != True and step < 500:
            old_observation = observation
            action = policy.sample(phi)
            observation, reward, done, _ = env.step(action)
            if old_observation in frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)
            return_value += current_gamma * reward
            current_gamma *= gamma_Q
            phi = features(observation)
            step += 1
        return_dist.append(return_value)
    return np.var(return_dist)


def run_agent(features, nepisodes,
              frozen_states, nfeatures, nactions, num_states,
              temperature, gamma, lr_Q, lr_M, lr_pol, tradeoff, rng):
    history = np.zeros((nepisodes, 2),
                       dtype=np.float32)  # 1. Steps, 2. Return for target policy

    # Rnadom Behavioral policy
    behavior_policy = RandomPolicy(nactions, rng)

    # Target Softmax Policy
    target_policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)

    # Weights of J (state-action value function)
    weight_Q = np.zeros((nfeatures, nactions))
    # Weights of M (state-action reward square function)
    weight_M = np.zeros((nfeatures, nactions))
    V0 = np.zeros(nfeatures)

    get_variance_after_episode = 20
    list_eps = list(np.arange(0, nepisodes, get_variance_after_episode))
    list_eps.append(nepisodes - 1)
    var_return_list = []
    save_index = 0

    weight_policy_store = np.zeros((len(list_eps), num_states, nactions),
                                   dtype=np.float32)

    env = gym.make('Puddle-v1')
    env._max_episode_steps = 500
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
            if old_observation in frozen_states:
                reward = np.random.normal(loc=0.0, scale=8.0)
            reward_list.append(reward)  # discounted reward
        phi_list.append(features(observation))  # last observation
        G = get_off_policy_corrected_return(reward_list, rho_list, gamma)
        G_bar = get_G_bar(reward_list, rho_list, gamma)

        # updates of the weight matrix
        episode_length = len(reward_list)
        temp_weight_Q = np.zeros_like(weight_Q)
        temp_weight_M = np.zeros_like(weight_M)
        temp_weight_policy = np.zeros_like(target_policy.weights)

        V0[phi_list[0]] += lr_Q * (G[0] - V0[phi_list[0]])  # initial V(x_0) estimate
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
            new_tradeoff = lr_pol * tradeoff
            policy_gradient_update = lr_pol * I_Q * rho_list[t] * weight_Q[
                phi_list[t], action_list[t]] - new_tradeoff * (
                                             I_M * rho_list[t] * weight_M[phi_list[t], action_list[t]] + 2 * gamma *
                                             G_bar[t] * I_Q * rho_list[t] * weight_Q[
                                                 phi_list[t], action_list[t]] - 2 * V0[phi_list[0]] * I_Q * rho_list[
                                                 t] * weight_Q[phi_list[t], action_list[t]])
            temp_weight_policy[phi_list[t], :] -= policy_gradient_update * actions_pmf
            temp_weight_policy[phi_list[t], action_list[t]] += policy_gradient_update
            I_Q *= gamma
            I_M *= new_gamma
        target_policy.weights += temp_weight_policy
        weight_Q += temp_weight_Q
        weight_M += temp_weight_M

        return_target, step_target = ReturnTargetPolicy(target_policy.weights, gamma, frozen_states, features, nactions)
        if episode in list_eps:
            weight_policy_store[save_index] = target_policy.weights
            var_return_list.append(
                VarReturnTargetPolicy(target_policy.weights, gamma, frozen_states, features, nactions))
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
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=2500)
    parser.add_argument('--nruns', help="Number of runs", type=int, default=1)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=1)

    args = parser.parse_args()
    env = gym.make('Puddle-v1')
    outer_dir = "../../Results/Results_Puddle"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "PuddleDiscreteTamar", "LRP" + str(args.lr_pol) + "_LRC" + str(args.lr_Q),
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

    num_states = env.observation_space.n
    nactions = env.action_space.n
    frozen_states = GetFrozenStates()

    features = Tabular(num_states)
    nfeatures = len(features)
    history, var_return, weight_policy = run_agent(features,
                                                   args.nepisodes,
                                                   frozen_states, nfeatures, nactions, num_states, args.temperature,
                                                   args.gamma, args.lr_Q,
                                                   args.lr_M, args.lr_pol, args.tradeoff,
                                                   np.random.RandomState(args.seed))

    np.save(os.path.join(dir_name, 'History.npy'), history)
    np.save(os.path.join(dir_name, 'Mean.npy'), history[:, 0])
    np.save(os.path.join(dir_name, 'Step.npy'), history[:, 1])
    np.save(os.path.join(dir_name, 'Var.npy'), var_return)
    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), weight_policy)
