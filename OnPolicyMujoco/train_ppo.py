from agent import Actor, Critic
from agent import Critic as Var_Critic
import hyper_params_vpac as h
import torch
from torch import nn
from torch import distributions
import gym
import copy
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import numpy as np
from itertools import count
import torch.nn.functional as F
import re
import copy
import time
import math
import pathlib
import time
import pickle
import os
from dataclasses import dataclass
from dotmap import DotMap

hp = h.HyperParameters()

ENV = hp.ENV
WORKSPACE_PATH = "../VPACResults"
hp.exp_name = hp.ENV.split("-")[0]
EXPERIMENT_NAME = hp.exp_name + "_Psi" + str(hp.psi)
res_dir = hp.dir
container = 'run_' + str(hp.name) + '_'
_INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")
log_dir = os.path.join(WORKSPACE_PATH, res_dir, EXPERIMENT_NAME, str(hp.name))
CHECK_PATH_CHECKPOINT = os.path.join(WORKSPACE_PATH, 'checkpoints', res_dir, EXPERIMENT_NAME)
BASE_CHECKPOINT_PATH = os.path.join(WORKSPACE_PATH, 'checkpoints', res_dir, EXPERIMENT_NAME, container)
os.makedirs(log_dir, exist_ok=True)

# tensorboard --logdir  "{WORKSPACE_PATH}/logs"
# Environment parameters

_MIN_REWARD_VALUES = torch.full([hp.parallel_rollouts], hp.min_reward)
ENV_MASK_VELOCITY = False

# Save metrics for viewing with tensorboard.
SAVE_METRICS_TENSORBOARD = True

# Save actor & critic parameters for viewing in tensorboard.
SAVE_PARAMETERS_TENSORBOARD = False

# Save training state frequency in PPO iterations.
CHECKPOINT_FREQUENCY = 100

# Step env asynchronously using multiprocess or synchronously.
ASYNCHRONOUS_ENVIRONMENT = False

# Force using CPU for gathering trajectories.
FORCE_CPU_GATHER = True

batch_count = hp.parallel_rollouts * hp.rollout_steps / hp.recurrent_seq_len / hp.batch_size
print(f"batch_count: {batch_count}")
assert batch_count >= 1., "Less than 1 batch per trajectory.  Are you sure that's what you want?"


def save_parameters(writer, tag, model, batch_idx):
    """
    Save model parameters for tensorboard.
    """
    for k, v in model.state_dict().items():
        shape = v.shape
        # Fix shape definition for tensorboard.
        shape_formatted = _INVALID_TAG_CHARACTERS.sub("_", str(shape))
        # Don't do this for single weights or biases
        if np.any(np.array(shape) > 1):
            mean = torch.mean(v)
            std_dev = torch.std(v)
            maximum = torch.max(v)
            minimum = torch.min(v)
            writer.add_scalars(
                "{}_weights/{}{}".format(tag, k, shape_formatted),
                {"mean": mean, "std_dev": std_dev, "max": maximum, "min": minimum},
                batch_idx,
            )
        else:
            writer.add_scalar("{}_{}{}".format(tag, k, shape_formatted), v.data, batch_idx)


def get_env_space():
    """
    Return obsvervation dimensions, action dimensions and whether or not action space is continuous.
    """
    env = gym.make(ENV)
    continuous_action_space = type(env.action_space) is gym.spaces.box.Box
    if continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    obsv_dim = env.observation_space.shape[0]
    return obsv_dim, action_dim, continuous_action_space


def get_last_checkpoint_iteration():
    """
    Determine latest checkpoint iteration.
    """
    if os.path.exists(BASE_CHECKPOINT_PATH + f"{hp.last_checkpoint}"):
        last_saved_checkpoint = hp.last_checkpoint
        return last_saved_checkpoint

    if os.path.exists(BASE_CHECKPOINT_PATH):
        max_checkpoint_iteration = max([int(dirname) for dirname in os.listdir(BASE_CHECKPOINT_PATH)])
    else:
        max_checkpoint_iteration = 0
    return max_checkpoint_iteration


def save_checkpoint(actor, critic, var_critic, actor_optimizer, critic_optimizer, var_critic_optimizer, iteration,
                    stop_conditions):
    """
    Save training checkpoint.
    """
    checkpoint = DotMap()
    checkpoint.env = ENV
    checkpoint.env_mask_velocity = ENV_MASK_VELOCITY
    checkpoint.iteration = iteration
    checkpoint.stop_conditions = stop_conditions
    checkpoint.hp = hp
    CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
    pathlib.Path(CHECKPOINT_PATH).mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH + "parameters.pt", "wb") as f:
        pickle.dump(checkpoint, f)
    with open(CHECKPOINT_PATH + "actor_class.pt", "wb") as f:
        pickle.dump(Actor, f)
    with open(CHECKPOINT_PATH + "critic_class.pt", "wb") as f:
        pickle.dump(Critic, f)
    with open(CHECKPOINT_PATH + "var_critic_class.pt", "wb") as f:
        pickle.dump(Var_Critic, f)
    torch.save(actor.state_dict(), CHECKPOINT_PATH + "actor.pt")
    torch.save(critic.state_dict(), CHECKPOINT_PATH + "critic.pt")
    torch.save(var_critic.state_dict(), CHECKPOINT_PATH + "var_critic.pt")
    torch.save(actor_optimizer.state_dict(), CHECKPOINT_PATH + "actor_optimizer.pt")
    torch.save(critic_optimizer.state_dict(), CHECKPOINT_PATH + "critic_optimizer.pt")
    torch.save(var_critic_optimizer.state_dict(), CHECKPOINT_PATH + "var_critic_optimizer.pt")


def start_or_resume_from_checkpoint():
    """
    Create actor, critic, actor optimizer and critic optimizer from scratch
    or load from latest checkpoint if it exists.
    """
    max_checkpoint_iteration = get_last_checkpoint_iteration()
    obsv_dim, action_dim, continuous_action_space = get_env_space()
    actor = Actor(obsv_dim,
                  action_dim,
                  continuous_action_space=continuous_action_space,
                  trainable_std_dev=hp.trainable_std_dev,
                  init_log_std_dev=hp.init_log_std_dev)
    critic = Critic(obsv_dim)
    var_critic = Var_Critic(obsv_dim)

    actor_optimizer = optim.AdamW(actor.parameters(), lr=hp.actor_learning_rate)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=hp.critic_learning_rate)
    var_critic_optimizer = optim.AdamW(var_critic.parameters(), lr=hp.var_critic_learning_rate)

    stop_conditions = StopConditions()
    stop_conditions.max_iterations = hp.max_iterations  # setting max_iterations here

    # If max checkpoint iteration is greater than zero initialise training with the checkpoint.
    if max_checkpoint_iteration > 0:
        actor_state_dict, critic_state_dict, var_critic_state_dict, actor_optimizer_state_dict, \
        critic_optimizer_state_dict, var_critic_optimizer_state_dict, stop_conditions = load_checkpoint(
            max_checkpoint_iteration)

        stop_conditions.max_iterations = hp.max_iterations
        actor.load_state_dict(actor_state_dict, strict=True)
        critic.load_state_dict(critic_state_dict, strict=True)
        var_critic.load_state_dict(var_critic_state_dict, strict=True)
        actor_optimizer.load_state_dict(actor_optimizer_state_dict)
        critic_optimizer.load_state_dict(critic_optimizer_state_dict)
        var_critic_optimizer.load_state_dict(var_critic_optimizer_state_dict)

        '''We have to move manually move optimizer states to 
        TRAIN_DEVICE manually since optimizer doesn't yet have a "to" method.#'''

        for state in actor_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(TRAIN_DEVICE)

        for state in critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(TRAIN_DEVICE)

        for state in var_critic_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(TRAIN_DEVICE)

    return actor, critic, var_critic, actor_optimizer, critic_optimizer, var_critic_optimizer, max_checkpoint_iteration, stop_conditions


def load_checkpoint(iteration):
    """
    Load from training checkpoint.
    """
    CHECKPOINT_PATH = BASE_CHECKPOINT_PATH + f"{iteration}/"
    with open(CHECKPOINT_PATH + "parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)

    assert ENV == checkpoint.env, "To resume training environment must match current settings."
    assert ENV_MASK_VELOCITY == checkpoint.env_mask_velocity, "To resume training model architecture must match current settings."
    # assert hp == checkpoint.hp, "To resume training hyperparameters must match current settings."

    actor_state_dict = torch.load(CHECKPOINT_PATH + "actor.pt", map_location=torch.device(TRAIN_DEVICE))
    critic_state_dict = torch.load(CHECKPOINT_PATH + "critic.pt", map_location=torch.device(TRAIN_DEVICE))
    var_critic_state_dict = torch.load(CHECKPOINT_PATH + "var_critic.pt", map_location=torch.device(TRAIN_DEVICE))
    actor_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "actor_optimizer.pt",
                                            map_location=torch.device(TRAIN_DEVICE))
    critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "critic_optimizer.pt",
                                             map_location=torch.device(TRAIN_DEVICE))
    var_critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "var_critic_optimizer.pt",
                                                 map_location=torch.device(TRAIN_DEVICE))

    return (actor_state_dict, critic_state_dict, var_critic_state_dict,
            actor_optimizer_state_dict, critic_optimizer_state_dict,
            var_critic_optimizer_state_dict, checkpoint.stop_conditions)


@dataclass
class StopConditions():
    """
    Store parameters and variables used to stop training.
    """
    best_reward: float = -1e6
    fail_to_improve_count: int = 0
    max_iterations: int = 1000000


class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observatiable.
    """

    def __init__(self, env):
        super(MaskVelocityWrapper, self).__init__(env)
        if ENV == "CartPole-v1":
            self.mask = np.array([1., 0., 1., 0.])
        elif ENV == "Pendulum-v0":
            self.mask = np.array([1., 1., 0.])
        elif ENV == "LunarLander-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1, ])
        elif ENV == "LunarLanderContinuous-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1, ])
        else:
            raise NotImplementedError

    def observation(self, observation):
        return observation * self.mask


def split_trajectories_episodes(trajectory_tensors):
    """
    Split trajectories by episode.
    """

    len_episodes = []
    trajectory_episodes = {key: [] for key in trajectory_tensors.keys()}
    for i in range(hp.parallel_rollouts):
        terminals_tmp = trajectory_tensors["terminals"].clone()
        terminals_tmp[0, i] = 1
        terminals_tmp[-1, i] = 1
        split_points = (terminals_tmp[:, i] == 1).nonzero() + 1

        split_lens = split_points[1:] - split_points[:-1]
        split_lens[0] += 1

        len_episode = [split_len.item() for split_len in split_lens]
        len_episodes += len_episode
        for key, value in trajectory_tensors.items():
            # Value includes additional step.
            if key == "values" or key == "var_values":
                value_split = list(torch.split(value[:, i], len_episode[:-1] + [len_episode[-1] + 1]))
                # Append extra 0 to values to represent no future reward.
                for j in range(len(value_split) - 1):
                    value_split[j] = torch.cat((value_split[j], torch.zeros(1)))
                trajectory_episodes[key] += value_split
            else:
                trajectory_episodes[key] += torch.split(value[:, i], len_episode)
    return trajectory_episodes, len_episodes


def pad_and_compute_returns(trajectory_episodes, len_episodes):
    """
    Pad the trajectories up to hp.rollout_steps so they can be combined in a
    single tensor.
    Add advantages, discounted_returns and discounted TD error square to the trajectories.
    """

    episode_count = len(len_episodes)
    padded_trajectories = {key: [] for key in trajectory_episodes.keys()}
    padded_trajectories["advantages"] = []
    padded_trajectories["discounted_returns"] = []
    padded_trajectories["discounted_returns_var"] = []

    for i in range(episode_count):
        single_padding = torch.zeros(hp.rollout_steps - len_episodes[i])
        for key, value in trajectory_episodes.items():
            if value[i].ndim > 1:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], value[0].shape[1], dtype=value[i].dtype)
            else:
                padding = torch.zeros(hp.rollout_steps - len_episodes[i], dtype=value[i].dtype)
            padded_trajectories[key].append(torch.cat((value[i], padding)))

        padded_trajectories["advantages"].append(
            torch.cat((compute_advantages(rewards=trajectory_episodes["rewards"][i],
                                          values=trajectory_episodes["values"][i],
                                          variances=trajectory_episodes["var_values"][i],
                                          discount=hp.discount,
                                          gae_lambda=hp.gae_lambda,
                                          tradeoff=hp.psi), single_padding)))

        padded_trajectories["discounted_returns"].append(
            torch.cat((calc_discounted_return(rewards=trajectory_episodes["rewards"][i],
                                              discount=hp.discount,
                                              final_value=trajectory_episodes["values"][i][-1]), single_padding)))

        padded_trajectories["discounted_returns_var"].append(
            torch.cat((calc_discounted_reward_var(rewards=trajectory_episodes["rewards"][i],
                                                  discount=hp.discount,
                                                  values=trajectory_episodes["values"][i],
                                                  final_variance=trajectory_episodes["var_values"][i][-1]),
                       single_padding)))

    return_val = {k: torch.stack(v) for k, v in padded_trajectories.items()}
    return_val["seq_len"] = torch.tensor(len_episodes)

    return return_val


@dataclass
class TrajectorBatch():
    """
    Dataclass for storing data batch.
    """
    states: torch.tensor
    actions: torch.tensor
    action_probabilities: torch.tensor
    advantages: torch.tensor
    discounted_returns: torch.tensor
    discounted_returns_var: torch.tensor
    batch_size: torch.tensor


class TrajectoryDataset():
    """
    Fast dataset for producing training batches from trajectories.
    """

    def __init__(self, trajectories, batch_size, device, batch_len):

        # Combine multiple trajectories into
        self.trajectories = {key: value for key, value in trajectories.items()}
        self.batch_len = batch_len
        truncated_seq_len = torch.clamp(trajectories["seq_len"] - batch_len + 1, 0, hp.rollout_steps)
        self.cumsum_seq_len = np.cumsum(np.concatenate((np.array([0]), truncated_seq_len.numpy())))
        self.batch_size = batch_size

    def __iter__(self):
        self.valid_idx = np.arange(self.cumsum_seq_len[-1])
        self.batch_count = 0
        return self

    def __next__(self):
        if self.batch_count * self.batch_size >= math.ceil(self.cumsum_seq_len[-1] / self.batch_len):
            raise StopIteration
        else:
            actual_batch_size = min(len(self.valid_idx), self.batch_size)
            start_idx = np.random.choice(self.valid_idx, size=actual_batch_size, replace=False)
            self.valid_idx = np.setdiff1d(self.valid_idx, start_idx)
            eps_idx = np.digitize(start_idx, bins=self.cumsum_seq_len, right=False) - 1
            seq_idx = start_idx - self.cumsum_seq_len[eps_idx]
            series_idx = np.linspace(seq_idx, seq_idx + self.batch_len - 1, num=self.batch_len, dtype=np.int64)
            self.batch_count += 1
            return TrajectorBatch(**{key: value[eps_idx, series_idx] for key, value
                                     in self.trajectories.items() if key in TrajectorBatch.__dataclass_fields__.keys()},
                                  batch_size=actual_batch_size)


def calc_discounted_return(rewards, discount, final_value):
    """
    Calculate discounted returns based on rewards and discount factor.
    """
    seq_len = len(rewards)
    discounted_returns = torch.zeros(seq_len)
    discounted_returns[-1] = rewards[-1] + discount * final_value
    for i in range(seq_len - 2, -1, -1):
        discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
    return discounted_returns


def calc_discounted_reward_var(rewards, discount, values, final_variance):
    """
    Calculate discounted**2 TD error square as a cumulant for variance.
    var = \sum_t (discount**2)^t (delta_t **2)
    """
    deltas = rewards + discount * values[1:] - values[:-1]
    deltas_square = deltas ** 2  # TD error square
    discount_var = discount ** 2
    seq_len = len(rewards)
    discounted_delta_square = torch.zeros(seq_len)  # cumulant for variance
    discounted_delta_square[-1] = deltas_square[-1] + discount_var * final_variance
    for i in range(seq_len - 2, -1, -1):
        discounted_delta_square[i] = deltas_square[i] + discount_var * discounted_delta_square[i + 1]
    return discounted_delta_square


def compute_val_advantages(rewards, values, discount, gae_lambda):
    """
    Compute General Advantage for Value.
    """
    deltas = rewards + discount * values[1:] - values[:-1]
    seq_len = len(rewards)
    advs = torch.zeros(seq_len + 1)
    multiplier = discount * gae_lambda
    for i in range(seq_len - 1, -1, -1):
        advs[i] = advs[i + 1] * multiplier + deltas[i]
    return advs[:-1]


def compute_var_advantages(rewards, values, variances, discount, gae_lambda):
    """
    Compute General Advantage for Variance.
    """
    deltas = rewards + discount * values[1:] - values[:-1]
    reward_var = deltas ** 2
    discount_var = discount ** 2
    deltas_var = reward_var + discount_var * variances[1:] - variances[:-1]
    seq_len = len(deltas_var)
    advs_var = torch.zeros(seq_len + 1)
    multiplier = discount_var * gae_lambda
    for i in range(seq_len - 1, -1, -1):
        advs_var[i] = advs_var[i + 1] * multiplier + deltas_var[i]
    return advs_var[:-1]


def compute_advantages(rewards, values, variances, discount, gae_lambda, tradeoff):
    """
    Compute total GAE advantage = advantage_value - tradeoff* advantage_variance
    """
    val_advantage = compute_val_advantages(rewards, values, discount, gae_lambda)
    var_advantage = compute_var_advantages(rewards, values, variances, discount, gae_lambda)
    return val_advantage - tradeoff * var_advantage


'''Sampling data from the environment'''


def gather_trajectories(input_data):
    """
    Gather policy trajectories from gym environment.
    """

    # Unpack inputs.
    env = input_data["env"]
    actor = input_data["actor"]
    critic = input_data["critic"]
    var_critic = input_data["var_critic"]

    # Initialise variables.
    obsv = env.reset()
    trajectory_data = {"states": [],
                       "actions": [],
                       "action_probabilities": [],
                       "rewards": [],
                       "true_rewards": [],
                       "values": [],
                       "terminals": [],
                       "var_values": []}

    terminal = torch.ones(hp.parallel_rollouts)

    with torch.no_grad():
        # Reset actor and critic state.
        # Take 1 additional step in order to collect the state and value for the final state.
        for i in range(hp.rollout_steps):
            # Choose next action
            state = torch.tensor(obsv, dtype=torch.float32)
            trajectory_data["states"].append(state)
            value = critic(state.to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
            # var value here
            var_value = var_critic(state.to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))

            trajectory_data["values"].append(value.squeeze(1).cpu())
            trajectory_data["var_values"].append(var_value.squeeze(1).cpu())
            action_dist = actor(state.to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
            action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
            if not actor.continuous_action_space:
                action = action.squeeze(1)
            trajectory_data["actions"].append(action.cpu())
            trajectory_data["action_probabilities"].append(action_dist.log_prob(action).cpu())

            # Step environment
            action_np = action.cpu().numpy()
            obsv, reward, done, _ = env.step(action_np)
            terminal = torch.tensor(done).float()
            transformed_reward = hp.scale_reward * torch.max(_MIN_REWARD_VALUES,
                                                             torch.tensor(reward).float())

            trajectory_data["rewards"].append(transformed_reward)
            trajectory_data["true_rewards"].append(torch.tensor(reward).float())
            trajectory_data["terminals"].append(terminal)

        # Compute final value to allow for incomplete episodes.
        state = torch.tensor(obsv, dtype=torch.float32)
        value = critic(state.to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
        var_value = var_critic(state.to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
        # Future value for terminal episodes is 0.
        trajectory_data["values"].append(value.squeeze(1).cpu() * (1 - terminal))
        trajectory_data["var_values"].append(var_value.squeeze(1).cpu() * (1 - terminal))

    # Combine step lists into tensors.
    trajectory_tensors = {key: torch.stack(value) for key, value in trajectory_data.items()}
    return trajectory_tensors


'''Main training loop'''


def train_model(actor, critic, var_critic, actor_optimizer, critic_optimizer, var_critic_optimizer, iteration,
                stop_conditions):
    # Vector environment manages multiple instances of the environment.
    # A key difference between this and the standard gym environment is it automatically resets.
    # Therefore when the done flag is active in the done vector the corresponding state is the first new state.
    env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)
    if ENV_MASK_VELOCITY:
        env = MaskVelocityWrapper(env)

    while iteration < stop_conditions.max_iterations + 1:

        actor = actor.to(GATHER_DEVICE)
        critic = critic.to(GATHER_DEVICE)
        var_critic = var_critic.to(GATHER_DEVICE)
        start_gather_time = time.time()

        # Gather trajectories.
        input_data = {"env": env, "actor": actor, "critic": critic, "var_critic": var_critic, "discount": hp.discount,
                      "gae_lambda": hp.gae_lambda}
        trajectory_tensors = gather_trajectories(input_data)
        trajectory_episodes, len_episodes = split_trajectories_episodes(trajectory_tensors)
        trajectories = pad_and_compute_returns(trajectory_episodes, len_episodes)

        # Calculate mean reward.
        complete_episode_count = trajectories["terminals"].sum().item()
        terminal_episodes_rewards = (
                trajectories["terminals"].sum(axis=1) * trajectories["true_rewards"].sum(axis=1)).sum()
        mean_reward = terminal_episodes_rewards / (complete_episode_count)

        # Check stop conditions.
        if mean_reward > stop_conditions.best_reward:
            stop_conditions.best_reward = mean_reward
            stop_conditions.fail_to_improve_count = 0
        else:
            stop_conditions.fail_to_improve_count += 1
        if stop_conditions.fail_to_improve_count > hp.patience:
            print(f"Policy has not yielded higher reward for {hp.patience} iterations...  Stopping now.")
            break

        trajectory_dataset = TrajectoryDataset(trajectories, batch_size=hp.batch_size,
                                               device=TRAIN_DEVICE, batch_len=hp.recurrent_seq_len)
        end_gather_time = time.time()
        start_train_time = time.time()

        actor = actor.to(TRAIN_DEVICE)
        critic = critic.to(TRAIN_DEVICE)
        var_critic = var_critic.to(TRAIN_DEVICE)

        # Train actor and critic.
        for epoch_idx in range(hp.ppo_epochs):
            for batch in trajectory_dataset:
                # Get batch
                # Update actor
                actor_optimizer.zero_grad()
                action_dist = actor(batch.states[-1, :].to(TRAIN_DEVICE))
                # Action dist runs on cpu as a workaround to CUDA illegal memory access.
                action_probabilities = action_dist.log_prob(batch.actions[-1, :].to("cpu")).to(TRAIN_DEVICE)
                # Compute probability ratio from probabilities in logspace.
                probabilities_ratio = torch.exp(action_probabilities -
                                                batch.action_probabilities[-1, :].to(TRAIN_DEVICE))
                surrogate_loss_0 = probabilities_ratio * batch.advantages[-1, :].to(TRAIN_DEVICE)
                surrogate_loss_1 = torch.clamp(probabilities_ratio, 1. - hp.ppo_clip,
                                               1. + hp.ppo_clip) * batch.advantages[-1, :].to(TRAIN_DEVICE)
                surrogate_loss_2 = action_dist.entropy().to(TRAIN_DEVICE)
                actor_loss = -torch.mean(torch.min(surrogate_loss_0, surrogate_loss_1)) - torch.mean(
                    hp.entropy_factor * surrogate_loss_2)
                actor_loss.backward()
                torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), hp.max_grad_norm)
                actor_optimizer.step()

                # Update critic
                critic_optimizer.zero_grad()
                values = critic(batch.states[-1, :].to(TRAIN_DEVICE))
                critic_loss = F.mse_loss(batch.discounted_returns[-1, :].to(TRAIN_DEVICE),
                                         values.squeeze(1))
                torch.nn.utils.clip_grad.clip_grad_norm_(critic.parameters(), hp.max_grad_norm)
                critic_loss.backward()
                critic_optimizer.step()

                # Update variance critic
                var_critic_optimizer.zero_grad()
                var_values = var_critic(batch.states[-1, :].to(TRAIN_DEVICE))
                var_critic_loss = F.mse_loss(batch.discounted_returns_var[-1, :].to(TRAIN_DEVICE),
                                             var_values.squeeze(1))
                torch.nn.utils.clip_grad.clip_grad_norm_(var_critic.parameters(), hp.max_grad_norm)
                var_critic_loss.backward()
                var_critic_optimizer.step()

        end_train_time = time.time()
        print(f"Iteration: {iteration},  Mean reward: {mean_reward}, Mean Entropy: {torch.mean(surrogate_loss_2)}, " +
              f"complete_episode_count: {complete_episode_count}, Gather time: {end_gather_time - start_gather_time:.2f}s, " +
              f"Train time: {end_train_time - start_train_time:.2f}s")

        if SAVE_METRICS_TENSORBOARD:
            writer.add_scalar("complete_episode_count", complete_episode_count, iteration)
            writer.add_scalar("total_reward", mean_reward, iteration)
            writer.add_scalar("actor_loss", actor_loss, iteration)
            writer.add_scalar("critic_loss", critic_loss, iteration)
            writer.add_scalar("var_critic_loss", var_critic_loss, iteration)
            writer.add_scalar("policy_entropy", torch.mean(surrogate_loss_2), iteration)
        if SAVE_PARAMETERS_TENSORBOARD:
            save_parameters(writer, "actor", actor, iteration)
            save_parameters(writer, "value", critic, iteration)
            save_parameters(writer, "var_value", var_critic, iteration)
        if iteration % CHECKPOINT_FREQUENCY == 0:
            save_checkpoint(actor, critic, var_critic, actor_optimizer,
                            critic_optimizer, var_critic_optimizer, iteration, stop_conditions)
        iteration += 1

    return stop_conditions.best_reward


'''Setting up the environment'''

# RANDOM_SEED = 0
# Set random seed for consistant runs.
# torch.random.manual_seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
# Set maximum threads for torch to avoid inefficient use of multiple cpu cores.
torch.set_num_threads(15)
TRAIN_DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
GATHER_DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"

'''Start training'''
writer = SummaryWriter(log_dir=log_dir)
actor, critic, var_critic, actor_optimizer, critic_optimizer, var_critic_optimizer, iteration, stop_conditions = start_or_resume_from_checkpoint()
score = train_model(actor, critic, var_critic, actor_optimizer, critic_optimizer, var_critic_optimizer, iteration,
                    stop_conditions)
