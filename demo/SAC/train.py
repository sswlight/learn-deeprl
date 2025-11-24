import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sac_agent import SACAgent
from buffer import ReplayBuffer

ENV_NAME = "LunarLanderContinuous-v3"
SEED = 0
START_TIMESTEPS = 10000 
MAX_TIMESTEPS = 500000
BATCH_SIZE = 256
DISCOUNT = 0.99
TAU = 0.005
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(ENV_NAME, render_mode=None)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"Env: {ENV_NAME} | Device: {DEVICE}")

    agent = SACAgent(
        state_dim, action_dim, max_action, DEVICE,
        DISCOUNT, TAU, lr=LR
    )

    buffer = ReplayBuffer(MAX_TIMESTEPS, state_dim, action_dim, DEVICE)

    state, _ = env.reset(seed=SEED)
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    return_list = []

    for t in range(int(MAX_TIMESTEPS)):
        episode_timesteps += 1

        if t < START_TIMESTEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, deterministic=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done_bool = float(terminated)

        buffer.add(state, action, reward, next_state, done_bool)

        state = next_state
        episode_reward += reward

        if t >= START_TIMESTEPS:
            agent.train(buffer, BATCH_SIZE)

        if done:
            print(f"Total T: {t+1} | Episode: {episode_num+1} | Reward: {episode_reward:.3f} | Alpha: {agent.alpha:.4f}")
            return_list.append(episode_reward)
            
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    torch.save(agent.actor.state_dict(), "./results/sac_actor.pth")
    torch.save(agent.critic.state_dict(), "./results/sac_critic.pth")
    
    plt.plot(return_list)
    plt.title(f"SAC on {ENV_NAME}")
    plt.show()

if __name__ == "__main__":
    train()