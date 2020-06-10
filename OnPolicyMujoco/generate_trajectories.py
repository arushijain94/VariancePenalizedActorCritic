from agent import Actor, Critic
from agent import Critic as Var_Critic
import hyper_params_vpac as h
import torch
import gym
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import numpy as np
import torch.nn.functional as F
import re
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
RETURNDIST_PATH = "../VPACResults/ReturnDistFixedSeed"
os.makedirs(RETURNDIST_PATH, exist_ok=True)
hp.exp_name = hp.ENV.split("-")[0]
EXPERIMENT_NAME = hp.exp_name + "_Psi" + str(hp.psi)
res_dir = hp.dir
container = 'run_' + str(hp.name) + '_'
iter_container = container + str(hp.max_iterations)
_INVALID_TAG_CHARACTERS = re.compile(r"[^-/\w\.]")
log_dir = os.path.join(WORKSPACE_PATH, res_dir, EXPERIMENT_NAME, str(hp.name))
LOAD_CHECKPOINT_PATH = os.path.join(WORKSPACE_PATH, 'checkpoints', res_dir, EXPERIMENT_NAME, iter_container)
os.makedirs(log_dir, exist_ok=True)

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


def get_last_checkpoint_iteration():
    max_checkpoint_iteration = LOAD_CHECKPOINT_PATH
    return max_checkpoint_iteration


def get_env_space(ENV):
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


def start_or_resume_from_checkpoint():
    """
    Create actor, critic, actor optimizer and critic optimizer from scratch
    or load from latest checkpoint if it exists.
    """
    max_checkpoint_iteration = get_last_checkpoint_iteration()
    stop_conditions = StopConditions()
    actor_state_dict, critic_state_dict, var_critic_state_dict, \
    actor_optimizer_state_dict, critic_optimizer_state_dict, \
    var_critic_optimizer_state_dict, stop_conditions, ENV, hp = load_checkpoint(max_checkpoint_iteration)

    '''Defining the models'''
    obsv_dim, action_dim, continuous_action_space = get_env_space(ENV)
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

    '''Loading the checkpoint in the model'''

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

    return actor, critic, var_critic, actor_optimizer, critic_optimizer, var_critic_optimizer, \
           max_checkpoint_iteration, stop_conditions, ENV, hp


class StopConditions():
    """
    Store parameters and variables used to stop training.
    """
    best_reward: float = -1e6
    fail_to_improve_count: int = 0
    max_iterations = hp.max_iterations


def load_checkpoint(iteration):
    """
    Load from training checkpoint.
    """

    CHECKPOINT_PATH = iteration
    if not os.path.exists(CHECKPOINT_PATH):
        print("Path not exist: ", CHECKPOINT_PATH, " ***********")
        exit()
    with open(CHECKPOINT_PATH + "/parameters.pt", "rb") as f:
        checkpoint = pickle.load(f)
    print("Critic params:", checkpoint.hp.critic_learning_rate)
    ENV = checkpoint.env  # "To resume training environment must match current settings."
    hp = checkpoint.hp  # "To resume training hyperparameters must match current settings."

    actor_state_dict = torch.load(CHECKPOINT_PATH +
                                  "/actor.pt", map_location=torch.device(TRAIN_DEVICE))
    critic_state_dict = torch.load(CHECKPOINT_PATH +
                                   "/critic.pt", map_location=torch.device(TRAIN_DEVICE))
    var_critic_state_dict = torch.load(CHECKPOINT_PATH +
                                       "/var_critic.pt", map_location=torch.device(TRAIN_DEVICE))

    actor_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "/actor_optimizer.pt",
                                            map_location=torch.device(TRAIN_DEVICE))
    critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "/critic_optimizer.pt",
                                             map_location=torch.device(TRAIN_DEVICE))
    var_critic_optimizer_state_dict = torch.load(CHECKPOINT_PATH + "/var_critic_optimizer.pt",
                                                 map_location=torch.device(TRAIN_DEVICE))

    return (actor_state_dict, critic_state_dict, var_critic_state_dict,
            actor_optimizer_state_dict, critic_optimizer_state_dict, var_critic_optimizer_state_dict,
            checkpoint.stop_conditions, ENV, hp)


def generate_trajs(ENV, actor):
    hp.parallel_rollouts = 1
    env = gym.vector.make(ENV, hp.parallel_rollouts, asynchronous=ASYNCHRONOUS_ENVIRONMENT)
    G = 0
    actor = actor.to(GATHER_DEVICE)

    # we fix env seed to remove randomness arising from different initial states
    env.seed(0)
    obsv = env.reset()
    terminal = torch.ones(hp.parallel_rollouts)
    return_episode = []
    i = 0
    with torch.no_grad():
        while True:
            state = torch.tensor(obsv, dtype=torch.float32)
            action_dist = actor(state.to(GATHER_DEVICE), terminal.to(GATHER_DEVICE))
            action = action_dist.sample().reshape(hp.parallel_rollouts, -1)
            action_np = action.cpu().numpy()
            obsv_prime, reward, done, _ = env.step(action_np)
            terminal = torch.tensor(done).float()
            if done == True:
                i += 1
                print('Done at:', i, '\n Return', G)
                return_episode.append(G)
                G = 0
                env.seed(0)
                obsv = env.reset()
                if i >= 100:  # rolled out 100 trajectory per policy
                    print('Finished')
                    break
            else:
                G = reward + hp.discount * G
                obsv = obsv_prime
    # save return of last episode
    save_dir = os.path.join(RETURNDIST_PATH, EXPERIMENT_NAME, str(hp.name))
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "return.npy"), np.asarray(return_episode))


torch.set_num_threads(15)
TRAIN_DEVICE = "cpu"
GATHER_DEVICE = "cuda" if torch.cuda.is_available() and not FORCE_CPU_GATHER else "cpu"

actor, critic, var_critic, actor_optimizer, critic_optimizer, var_critic_optimizer, iteration, stop_conditions, ENV, hp = start_or_resume_from_checkpoint()
generate_trajs(ENV, actor)
