import argparse

ENV_MASK_VELOCITY = False
# Default Hyperparameters
SCALE_REWARD: float = 0.01
MIN_REWARD: float = -1000.
HIDDEN_SIZE: int = 256
BATCH_SIZE: int = 512
DISCOUNT: float = 0.99
GAE_LAMBDA: float = 0.95
PPO_CLIP: float = 0.2
PSI: float = 0.0
PPO_EPOCHS: int = 5
MAX_GRAD_NORM: float = 1.
ENTROPY_FACTOR: float = 0.
ACTOR_LEARNING_RATE: float = 9e-4
CRITIC_LEARNING_RATE: float = 1e-3
VAR_CRITIC_LEARNING_RATE: float = 1e-3
RECURRENT_SEQ_LEN: int = 8
RECURRENT_LAYERS: int = 1
ROLLOUT_STEPS: int = 2048
PARALLEL_ROLLOUTS: int = 8
PATIENCE: int = int(1e6)
TRAINABLE_STD_DEV: bool = False
INIT_LOG_STD_DEV: float = 0.0
MU: float = 0.15
parser = argparse.ArgumentParser()

parser.add_argument("--ENV", default='Hopper-v2', type=str, help='Environment_id')
parser.add_argument("-n", "--name", default=1, type=int, help="Name of the run")
parser.add_argument("--dir", default='PPOResults', type=str, help="folder name for saving the results")
parser.add_argument("--exp_name", default='Hopper', type=str, help="directory for experiment name")
parser.add_argument('--max_iterations', default=int(2500), type=int, help='Total run iterations')
parser.add_argument('--rollout_steps', default=ROLLOUT_STEPS, type=int, help='Number of steps takes by the policy')
parser.add_argument("--gae_lambda", default=GAE_LAMBDA, type=float, help="Lambda")
parser.add_argument("--ppo_clip", default=PPO_CLIP, type=float, help="cliping range")
parser.add_argument("--psi", default=PSI, type=float, help="mean-variance tradeoff")
parser.add_argument("--entropy_factor", default=ENTROPY_FACTOR, type=float, help='Entropy Coefficient')
parser.add_argument('--seed', default=0, type=int, help='Initial seed')
parser.add_argument('--actor_learning_rate', default=ACTOR_LEARNING_RATE, type=float, help='Actor learning rate')
parser.add_argument('--critic_learning_rate', default=CRITIC_LEARNING_RATE, type=float, help='Critic learning rate')
parser.add_argument('--var_critic_learning_rate', default=VAR_CRITIC_LEARNING_RATE, type=float,
                    help='Variance Critic learning rate')
parser.add_argument('--hidden_size', default=HIDDEN_SIZE, type=int, help='Hidden state dimension')
parser.add_argument('--parallel_rollouts', default=PARALLEL_ROLLOUTS, type=int, help='Number of parallel rollouts')
parser.add_argument('--patience', default=int(1e7), type=int, help='Stopping condition')
parser.add_argument('--recurrent_seq_len', default=RECURRENT_SEQ_LEN, type=int,
                    help='Recurrent sequence used for training')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='Batch size for training')
parser.add_argument('--recurrent_layers', default=RECURRENT_LAYERS, type=int, help='Recurrent Layers in RNN')
parser.add_argument('--scale_reward', default=SCALE_REWARD, type=float, help='Reward Scale')
parser.add_argument('--discount', default=DISCOUNT, type=float, help='Discount factor')
parser.add_argument('--ppo_epochs', default=PPO_EPOCHS, type=float, help='Epochs for PPO')
parser.add_argument('--max_grad_norm', default=MAX_GRAD_NORM, type=float, help='Max norm for the gradient')
parser.add_argument('--trainable_std_dev', default='store_false', type=bool, help='Std deviation for policy')
parser.add_argument('--init_log_std_dev', default=INIT_LOG_STD_DEV, type=float, help='Initial std deviation for policy')
parser.add_argument('--mu', default=MU, type=float, help='Variance trade-off parameter.')
parser.add_argument('--last_checkpoint', default=0, type=int, help='Last checkpoint')
args = parser.parse_args()


class HyperParameters():
    ENV = args.ENV
    dir = args.dir
    exp_name = args.exp_name
    name = args.name
    scale_reward = args.scale_reward
    hidden_size = args.hidden_size
    batch_size = args.batch_size
    discount = args.discount
    gae_lambda = args.gae_lambda
    ppo_clip = args.ppo_clip
    psi = args.psi
    ppo_epochs = args.ppo_epochs
    max_grad_norm = args.max_grad_norm
    entropy_factor = args.entropy_factor
    actor_learning_rate = args.actor_learning_rate
    critic_learning_rate = args.critic_learning_rate
    var_critic_learning_rate = args.var_critic_learning_rate
    recurrent_seq_len = args.recurrent_seq_len
    recurrent_layers = args.recurrent_layers
    rollout_steps = args.rollout_steps
    parallel_rollouts = args.parallel_rollouts
    patience = args.patience
    max_iterations = args.max_iterations
    # Apply to continuous action spaces only
    trainable_std_dev = args.trainable_std_dev
    init_log_std_dev = args.init_log_std_dev
    mu = args.mu
    last_checkpoint = args.last_checkpoint

    min_reward: float = MIN_REWARD
