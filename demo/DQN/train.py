import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from buffer import ReplayBuffer

ENV_NAME = "LunarLander-v3"
NUM_EPISODES = 1000
MAX_STEPS = 1000
BATCH_SIZE = 64
BUFFER_CAPACITY = 100000
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    env = gym.make(ENV_NAME, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Env: {ENV_NAME} | State: {state_dim} | Action: {action_dim} | Device: {DEVICE}")

    agent = DQNAgent(
        state_dim, action_dim, 
        lr=LEARNING_RATE, 
        gamma=GAMMA, 
        epsilon=EPSILON_START, 
        epsilon_decay=EPSILON_DECAY, 
        min_epsilon=EPSILON_END, 
        device=DEVICE
    )
    
    buffer = ReplayBuffer(BUFFER_CAPACITY, state_dim, device=DEVICE)
    return_list = []

    for i in range(NUM_EPISODES):
        state, info = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(state, action, reward, next_state, int(done))
            
            state = next_state
            episode_reward += reward

            if buffer.size > BATCH_SIZE:
                agent.update(buffer, BATCH_SIZE)
            
            if done:
                break

        return_list.append(episode_reward)
        agent.update_epsilon()

        if i % TARGET_UPDATE_FREQ == 0:
            agent.sync_target_network()

        if i % 10 == 0:
            print(f"Episode: {i}/{NUM_EPISODES}, Score: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    print("Training finished.")
    torch.save(agent.q_net.state_dict(), "dqn_lunarlander.pth")
    print("Model saved.")
    
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'DQN on {ENV_NAME}')
    plt.show()

if __name__ == "__main__":
    train()