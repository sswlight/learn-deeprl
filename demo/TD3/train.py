import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from td3_agent import TD3Agent
from buffer import ReplayBuffer

ENV_NAME = "LunarLanderContinuous-v3"
SEED = 0
START_TIMESTEPS = 25000   # 热身步数：前25k步随机探索
MAX_TIMESTEPS = 1000000   # 总步数
EXPL_NOISE = 0.1          # 探索噪音标准差
BATCH_SIZE = 256
DISCOUNT = 0.99
TAU = 0.005
POLICY_NOISE = 0.2        # 训练时加给Target Actor的噪音
NOISE_CLIP = 0.5
POLICY_FREQ = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    if not os.path.exists("./results"):
        os.makedirs("./results")

    env = gym.make(ENV_NAME, render_mode=None)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print(f"Env: {ENV_NAME} | State: {state_dim} | Action: {action_dim} | Max Action: {max_action}")

    agent = TD3Agent(
        state_dim, action_dim, max_action, DEVICE,
        DISCOUNT, TAU, POLICY_NOISE, NOISE_CLIP, POLICY_FREQ
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
            action = agent.select_action(state)
            noise = np.random.normal(0, max_action * EXPL_NOISE, size=action_dim)
            action = (action + noise).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        done_bool = float(done) 

        buffer.add(state, action, reward, next_state, done_bool)

        state = next_state
        episode_reward += reward

        if t >= START_TIMESTEPS:
            agent.train(buffer, BATCH_SIZE)

        if done:
            print(f"Total T: {t+1} | Episode Num: {episode_num+1} | Reward: {episode_reward:.3f}")
            return_list.append(episode_reward)
            
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    torch.save(agent.actor.state_dict(), "./results/td3_actor.pth")
    torch.save(agent.critic.state_dict(), "./results/td3_critic.pth")
    
    plt.plot(return_list)
    plt.title("TD3 on LunarLanderContinuous-v3")
    plt.show()

if __name__ == "__main__":
    train()