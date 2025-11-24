import torch
import numpy as np

class PPOBuffer:
    def __init__(self, capacity, state_dim, action_dim, device, gamma=0.99, gae_lambda=0.95):
        self.capacity = capacity
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ptr = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.masks = np.zeros((capacity, 1), dtype=np.float32)
        self.log_probs = np.zeros((capacity, 1), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)

        self.advantages = np.zeros((capacity, 1), dtype=np.float32)
        self.returns = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, mask, log_prob, value):
        if self.ptr < self.capacity:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.masks[self.ptr] = mask
            self.log_probs[self.ptr] = log_prob
            self.values[self.ptr] = value
            self.ptr += 1

    def compute_gae(self, next_value):
        last_gae_lam = 0
        for t in reversed(range(self.capacity)):
            if t == self.capacity - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]

            mask = self.masks[t]
            delta = self.rewards[t] + self.gamma * next_val * mask - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * mask * last_gae_lam
            
            self.advantages[t] = last_gae_lam

        self.returns = self.advantages + self.values

    def get_mini_batches(self, batch_size):
        indices = np.arange(self.capacity)
        np.random.shuffle(indices)

        for start in range(0, self.capacity, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            
            yield (
                torch.as_tensor(self.states[idx], device=self.device),
                torch.as_tensor(self.actions[idx], device=self.device),
                torch.as_tensor(self.log_probs[idx], device=self.device),
                torch.as_tensor(self.returns[idx], device=self.device),
                torch.as_tensor(self.advantages[idx], device=self.device)
            )

    def clear(self):
        self.ptr = 0