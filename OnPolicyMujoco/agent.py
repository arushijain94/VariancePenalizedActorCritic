import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
import hyper_params as hp
import numpy as np

hp = hp.HyperParameters()
activation = torch.tanh


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous_action_space,
                 trainable_std_dev, init_log_std_dev=None):
        super().__init__()
        self.hidden_dim_1 = hp.hidden_size
        self.hidden_dim_2 = hp.hidden_size // 2
        logit_hidden_dim = int(np.sqrt(hp.hidden_size * action_dim))

        self.layer_1 = nn.Linear(state_dim, self.hidden_dim_1)
        self.layer_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.layer_3 = nn.Linear(self.hidden_dim_2, logit_hidden_dim)

        self.layer_policy_logits = nn.Linear(logit_hidden_dim, action_dim)
        self.action_dim = action_dim
        self.continuous_action_space = continuous_action_space
        self.log_std_dev = nn.Parameter(init_log_std_dev * torch.ones((action_dim), dtype=torch.float),
                                        requires_grad=trainable_std_dev)
        self.covariance_eye = torch.eye(self.action_dim)
        self.hidden_layer = None

    def forward(self, state, terminal=None):

        batch_size = state.shape[0]
        device = state.device

        phi_s = activation(self.layer_1(state))
        phi_s = activation(self.layer_2(phi_s))
        phi_s = activation(self.layer_3(phi_s))

        policy_logits_out = self.layer_policy_logits(phi_s)

        if self.continuous_action_space:
            cov_matrix = self.covariance_eye.to(device).expand(batch_size, self.action_dim,
                                                               self.action_dim) * torch.exp(self.log_std_dev.to(device))
            ''' We define the distribution on the CPU since otherwise 
            operations fail with CUDA illegal memory access error.'''
            policy_dist = torch.distributions.multivariate_normal.MultivariateNormal(policy_logits_out.to("cpu"),
                                                                                     cov_matrix.to("cpu"))
        else:
            policy_dist = distributions.Categorical(F.softmax(policy_logits_out, dim=1).to("cpu"))
        return policy_dist


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.hidden_dim_1 = hp.hidden_size
        self.hidden_dim_2 = hp.hidden_size // 2

        logit_hidden_dim = int(np.sqrt(hp.hidden_size * state_dim))

        self.layer_1 = nn.Linear(state_dim, self.hidden_dim_1)
        self.layer_2 = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)
        self.layer_3 = nn.Linear(self.hidden_dim_2, logit_hidden_dim)
        self.layer_value = nn.Linear(logit_hidden_dim, 1)

    def forward(self, state, terminal=None):
        # batch_size = state.shape[1]
        phi_s = activation(self.layer_1(state))
        phi_s = activation(self.layer_2(phi_s))
        phi_s = activation(self.layer_3(phi_s))
        value_out = self.layer_value(phi_s)
        return value_out
