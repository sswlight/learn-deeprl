import gymnasium as gym
import torch
import numpy as np
from model import Actor

ENV_NAME = "LunarLanderContinuous-v3"
MODEL_PATH = "./results/td3_actor.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test():
    env = gym.make(ENV_NAME, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    actor = Actor(state_dim, action_dim, max_action).to(DEVICE)
    actor.load_state_dict(torch.load(MODEL_PATH))
    actor.eval()

    state, _ = env.reset()
    total_reward = 0
    
    while True:
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(DEVICE)
        
        with torch.no_grad():
            # 测试时直接输出动作，不加噪音
            action = actor(state_tensor).cpu().data.numpy().flatten()
            
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode Reward: {total_reward:.2f}")
            state, _ = env.reset()
            total_reward = 0

if __name__ == "__main__":
    test()