import gymnasium as gym
import torch
from model import ActorCritic

ENV_NAME = "LunarLander-v3"
MODEL_PATH = "ppo_lunarlander.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test():
    env = gym.make(ENV_NAME, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    state, _ = env.reset()
    total_reward = 0
    
    while True:
        state_tensor = torch.FloatTensor(state).to(DEVICE)
        
        with torch.no_grad():
            action, _, _ = model.act(state_tensor)
            
        state, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        
        if terminated or truncated:
            print(f"Score: {total_reward:.2f}")
            state, _ = env.reset()
            total_reward = 0

if __name__ == "__main__":
    test()