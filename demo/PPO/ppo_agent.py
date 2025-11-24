import torch
import torch.nn as nn
import torch.optim as optim
from model import ActorCritic

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_rate=0.2, entropy_coef=0.0, value_coef=0.5, epochs=10, device="cuda"):
        self.device = device
        self.gamma = gamma
        self.clip_rate = clip_rate
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.epochs = epochs

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state)
        return action.item(), log_prob.item(), value.item()

    def get_value(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            value = self.policy.critic(state)
        return value.item()

    def update(self, buffer, batch_size=64):
        for _ in range(self.epochs):
            for states, actions, old_log_probs, returns, advantages in buffer.get_mini_batches(batch_size):
                
                old_log_probs = old_log_probs.flatten()
                returns = returns.flatten()
                advantages = advantages.flatten()
                actions = actions.flatten()

                new_values, new_log_probs, entropy = self.policy.evaluate(states, actions)
                
                new_values = new_values.flatten()

                ratio = torch.exp(new_log_probs - old_log_probs)

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_rate, 1.0 + self.clip_rate) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self.mse_loss(new_values, returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()