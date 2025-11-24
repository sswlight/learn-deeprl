import torch
import torch.nn as nn
class QNetwork(nn.Module):
    def __init__(self,state,action):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action)
        )
    def forward(self, x):
        return self.network(x)