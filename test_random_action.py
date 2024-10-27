import gymnasium as gym
import numpy as np

# Create the Ant-v4 environment
env = gym.make("Ant-v4", render_mode="human")

# Set the number of episodes
num_episodes = 5

for episode in range(num_episodes):
    observation, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Take a random action
        action = env.action_space.sample()
        
        # Step the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        total_reward += reward

    print(f"Episode {episode + 1} finished with total reward: {total_reward}")

env.close()
