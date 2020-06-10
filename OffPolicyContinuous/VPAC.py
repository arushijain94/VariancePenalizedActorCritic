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


class StateActionLearning:
    def __init__(self, gamma, lr, weights, policy, behavioral_policy, variance):
        self.lr = lr
        self.gamma = gamma
        self.weights = weights
        self.policy = policy
        self.behavioral_policy = behavioral_policy
        self.variance = variance  # binary value (0: its is Q(s,a) value, 1: sigma(s,a) value)

    def start(self, phi, action):
        self.last_phi = phi
        self.last_action = action
        self.last_value = self.value(phi, action)

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    # Update for the parameter of the Q and Sigma value (s,a)
    def update(self, phi, action, reward, done):
        update_target = reward
        if not done:
            current_rho = min(1, (self.policy.pmf(phi)[int(action)] / self.behavioral_policy.pmf()[int(action)]))
            current_value = self.value(phi, action)
            if self.variance:
                update_target += self.gamma * (current_rho ** 2.0) * current_value
            else:
                update_target += self.gamma * current_rho * current_value
        # Weight gradient update step
        tderror = update_target - self.last_value
        self.weights[self.last_phi, self.last_action] += self.lr * tderror
        if not done:
            self.last_value = current_value
        self.last_action = action
        self.last_phi = phi
        return tderror


class PolicyGradient:
    def __init__(self, policy, lr):
        self.lr = lr
        self.policy = policy

    # Updation of the theta parameter of the policy
    def update(self, phi, action, critic, sigma, psi, I_Q, I_sigma, rho_Q, rho_sigma, first_time_step):
        actions_pmf = self.policy.pmf(phi)
        const = 2
        if first_time_step:
            const = 1
        if psi != 0.0:  # variance as regularization factor to optimization criterion
            var_constant = -self.lr * psi * I_sigma * rho_sigma * sigma * const
            self.policy.weights[phi, :] -= var_constant * actions_pmf
            self.policy.weights[phi, action] += var_constant

        Q_constant = self.lr * I_Q * rho_Q * critic
        self.policy.weights[phi, :] -= Q_constant * actions_pmf
        self.policy.weights[phi, action] += Q_constant


def save_params(args, dir_name):
    f = os.path.join(dir_name, "Params.txt")
    with open(f, "w") as f_w:
        for param, val in sorted(vars(args).items()):
            f_w.write("{0}:{1}\n".format(param, val))


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


# Get return for target policy
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


def run_agent(nepisodes, temperature, gamma_Q, gamma_var,
              lr_critic, lr_sigma, lr_theta, psi, rng, maxDiscreteStates, ntiles):
    env = gym.make('PuddleEnv-v0')
    env._max_episode_steps = 5000
    features_range = env.observation_space.high - env.observation_space.low
    features = TileFeature(ntiles, 5, maxDiscreteStates, features_range)  # 5X5 tiles over joint space
    nfeatures = int(maxDiscreteStates)
    nactions = env.action_space.n

    get_variance_after_episode = 100
    list_eps = list(np.arange(0, nepisodes, get_variance_after_episode))
    list_eps.append(nepisodes - 1)
    var_return_list = []
    save_index = 0

    behavioral_policy = RandomPolicy(nactions, rng)
    storing_arr_dim = len(list_eps)

    history = np.zeros((nepisodes, 2))  # 1. Return from Target 2. Steps in target
    # storage the weights of the trained model
    weight_policy = np.zeros((storing_arr_dim, nfeatures, nactions),
                             dtype=np.float32)

    # Target policy is as softmax policy
    policy = SoftmaxPolicy(rng, nfeatures, nactions, temperature)
    # Action_critic is Q value of state-action pair
    weights_QVal = np.zeros((nfeatures, nactions))  # positive weight initialization
    action_critic = StateActionLearning(gamma_Q, lr_critic, weights_QVal, policy, behavioral_policy, 0)

    # Variance is sigma of state-action pair
    weights_var = np.zeros((nfeatures, nactions))  # weight initialization
    sigma = StateActionLearning(gamma_var, lr_sigma, weights_var, policy,
                                behavioral_policy, 1)

    # Policy gradient improvement step
    policy_improvement = PolicyGradient(policy, lr_theta)

    for episode in range(nepisodes):
        first_time_step = 1
        observation = env.reset()
        phi = features(observation)
        action = behavioral_policy.sample()
        action_critic.start(phi, action)
        sigma.start(phi, action)
        step = 0
        done = False
        I_Q = 1.0
        I_sigma = 1.0
        # Using Retrace for correcting discrepancy between two policy distribution
        rho_Q = min(1, (policy.pmf(phi)[int(action)] / behavioral_policy.pmf()[int(action)]))
        rho_sigma = rho_Q
        while done != True and step < 5000:
            old_observation = observation
            old_phi = phi
            old_action = action
            observation, reward, done, _ = env.step(action)

            # Frozen state receives a variable normal reward[-8, 8], where reward is given when transition is made
            # out of that state
            if check_if_agent_near_puddle(old_observation):
                reward = tweak_reward_near_puddle(reward)
            phi = features(observation)
            # Get action from the behavioral policy
            action = behavioral_policy.sample()

            # Critic update
            tderror = action_critic.update(phi, action, reward, done)
            if psi != 0.0:
                try:
                    td_square = pow(tderror, 2.0)
                # Just to prevent the overflow error
                except OverflowError:
                    td_square = tderror * 2
                sigma.update(phi, action, td_square, done)
                sigma_val = sigma.value(old_phi, old_action)
                if sigma_val == np.nan:
                    sigma_val = 0.0
            else:
                sigma_val = 0.0

            critic_val = action_critic.value(old_phi, old_action)
            policy_improvement.update(old_phi, old_action, critic_val, sigma_val, psi,
                                      I_Q, I_sigma, rho_Q, rho_sigma, first_time_step)
            first_time_step = 0
            step += 1
            I_Q *= gamma_Q
            I_sigma *= gamma_var
            rho = min(1, (policy.pmf(phi)[int(action)] / behavioral_policy.pmf()[int(action)]))
            rho_Q *= rho
            rho_sigma *= rho ** 2

        return_target, step_target = ReturnTargetPolicy(policy, gamma_Q, features, env)
        if episode in list_eps:
            weight_policy[save_index] = policy.weights
            var_return_list.append(VarReturnTargetPolicy(policy, gamma_Q, features, env))
            save_index += 1

        history[episode, 0] = return_target
        history[episode, 1] = step_target

    return history, weight_policy, np.asarray(var_return_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', help='Discount factor', type=float, default=0.99)
    parser.add_argument('--lmbda', help='Lambda', type=float, default=1.0)
    parser.add_argument('--lr_critic', help="Learning rate for Q value", type=float, default=0.5)
    parser.add_argument('--lr_theta', help="Learning rate for policy parameterization theta", type=float,
                        default=0.1)
    parser.add_argument('--lr_sigma', help="Learning rate for sigma variance of return", type=float, default=0.25)
    parser.add_argument('--temperature', help="Temperature parameter for softmax", type=float, default=50)
    parser.add_argument('--psi', help="Psi regularizer for Variance in return", type=float, default=0.0)
    parser.add_argument('--nepisodes', help="Number of episodes per run", type=int, default=3000)
    parser.add_argument('--nruns', help="Number of run for target policy", type=int, default=50)
    parser.add_argument('--seed', help="seed value for experiment", type=int, default=1)
    parser.add_argument('--maxDiscreteStates', help="discrete states", type=int, default=1024)
    parser.add_argument('--ntiles', help="tiles", type=int, default=10)

    args = parser.parse_args()
    outer_dir = "../../Results/Results_PuddleCont"
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)
    outer_dir = os.path.join(outer_dir, "PuddleCont", "Psi" + str(args.psi))
    if not os.path.exists(outer_dir):
        os.makedirs(outer_dir)

    dir_name = "R" + str(args.nruns) + "_E" + str(args.nepisodes) + "_Psi" + str(args.psi) + \
               "_LRC" + str(args.lr_critic) + "_LRTheta" + str(args.lr_theta) + "_LRV" + str(args.lr_sigma) + \
               "_temp" + str(args.temperature) + "_lambda" + str(args.lmbda) + "_seed" + str(args.seed)

    dir_name = os.path.join(outer_dir, dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    save_params(args, dir_name)
    threads = []
    gamma_var = pow(args.gamma * args.lmbda, 2.0)
    # Tile coding, therefore diving learning rate by num of tiles
    args.lr_theta /= args.ntiles
    args.lr_critic /= args.ntiles
    args.lr_sigma /= args.ntiles

    history, weight_policy, var_return = run_agent(args.nepisodes, args.temperature,
                                                   args.gamma, gamma_var, args.lr_critic,
                                                   args.lr_sigma, args.lr_theta, args.psi,
                                                   np.random.RandomState(args.seed),
                                                   args.maxDiscreteStates, args.ntiles)

    np.save(os.path.join(dir_name, 'History.npy'), history)
    np.save(os.path.join(dir_name, 'Mean.npy'), history[:, 0])
    np.save(os.path.join(dir_name, 'Step.npy'), history[:, 1])
    np.save(os.path.join(dir_name, 'Var.npy'), var_return)
    np.save(os.path.join(dir_name, 'Weights_Policy.npy'), weight_policy)
