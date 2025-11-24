import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TwinCritic, self).__init__()

        self.Q1 = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

        self.Q2 = nn.Sequential(
            layer_init(nn.Linear(state_dim + action_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.Q1(sa), self.Q2(sa)

class SoftActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(SoftActor, self).__init__()
        
        self.max_action = max_action
        
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 2 * action_dim), std=1.0)
        )

    def forward(self, state):
        mean_log_std = self.net(state)
        mean, log_std = mean_log_std.chunk(2, dim=-1)
        
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

    def get_action(self, state):
        mean_log_std = self.net(state)
        mean, _ = mean_log_std.chunk(2, dim=-1)
        return self.max_action * torch.tanh(mean)