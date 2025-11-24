import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from model import QNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, device="cuda", n_step=1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device
        self.n_step = n_step
        self.gamma_n = gamma ** n_step

        self.q_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def take_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()

    def update(self, replay_buffer, batch_size):
        if replay_buffer.size < batch_size:
            return

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        q_current = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            q_next_max = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma_n * q_next_max * (1 - dones)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def sync_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())