import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity, state_dim, device, n_step=3, gamma=0.99):
        self.device = device
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        if len(self.n_step_buffer) < self.n_step and not done:
            return

        while len(self.n_step_buffer) > 0:
            ret_state, ret_action = self.n_step_buffer[0][:2]
            ret_reward = 0
            for i, t in enumerate(self.n_step_buffer):
                ret_reward += (self.gamma ** i) * t[2]
            
            ret_next_state = self.n_step_buffer[-1][3]
            ret_done = self.n_step_buffer[-1][4]

            self._store(ret_state, ret_action, ret_reward, ret_next_state, ret_done)

            if not done:
                self.n_step_buffer.popleft()
                break
            
            self.n_step_buffer.popleft()

    def _store(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        batch_data = [
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        ]
        return [torch.as_tensor(x, device=self.device) for x in batch_data]