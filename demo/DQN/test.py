import gymnasium as gym
import torch
from model import QNetwork

ENV_NAME = "LunarLander-v3"
MODEL_PATH = "dqn_lunarlander.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test():
    env = gym.make(ENV_NAME, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = QNetwork(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    state, _ = env.reset()
    total_reward = 0
    
    while True:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            action = model(state_tensor).argmax().item()
            
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode Finished. Total Reward: {total_reward:.2f}")
            state, _ = env.reset()
            total_reward = 0

if __name__ == "__main__":
    test()