import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        self.net = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=1.0)
        )

    def forward(self, state):
        x = self.net(state)
        return self.max_action * torch.tanh(x)


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
        
        q1 = self.Q1(sa)
        q2 = self.Q2(sa)
        
        return q1, q2

    def q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.Q1(sa)