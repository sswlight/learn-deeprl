import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from ppo_agent import PPOAgent
from buffer import PPOBuffer

ENV_NAME = "LunarLander-v3"
TOTAL_TIMESTEPS = 1000000 
STEPS_PER_ROLLOUT = 2048
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RATE = 0.2
ENTROPY_COEF = 0.0 
VALUE_COEF = 0.5
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    env = gym.make(ENV_NAME, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Env: {ENV_NAME} | Device: {DEVICE}")

    agent = PPOAgent(
        state_dim, action_dim,
        lr=LEARNING_RATE, 
        gamma=GAMMA, 
        clip_rate=CLIP_RATE,
        entropy_coef=ENTROPY_COEF, 
        value_coef=VALUE_COEF,
        epochs=EPOCHS, 
        device=DEVICE
    )

    buffer = PPOBuffer(
        STEPS_PER_ROLLOUT, state_dim, action_dim,
        device=DEVICE, gamma=GAMMA, gae_lambda=GAE_LAMBDA
    )

    state, info = env.reset()
    episode_reward = 0
    return_list = []
    total_steps = 0
    
    num_updates = TOTAL_TIMESTEPS // STEPS_PER_ROLLOUT

    for update in range(num_updates):
        for step in range(STEPS_PER_ROLLOUT):
            total_steps += 1
            
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            mask = 0.0 if done else 1.0

            buffer.add(state, action, reward, mask, log_prob, value)
            
            state = next_state
            episode_reward += reward

            if done:
                return_list.append(episode_reward)
                state, info = env.reset()
                episode_reward = 0

        next_value = agent.get_value(state)
        buffer.compute_gae(next_value)

        agent.update(buffer, BATCH_SIZE)
        buffer.clear()

        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * LEARNING_RATE
        agent.optimizer.param_groups[0]["lr"] = lrnow

        avg_score = np.mean(return_list[-10:]) if len(return_list) > 0 else 0
        print(f"Update: {update+1}/{num_updates} | Steps: {total_steps} | Avg: {avg_score:.2f} | LR: {lrnow:.2e}")

    torch.save(agent.policy.state_dict(), "ppo_lunarlander.pth")
    plt.plot(return_list)
    plt.show()

if __name__ == "__main__":
    train()