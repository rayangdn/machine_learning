import numpy as np
import random
import matplotlib.pyplot as plt
import pygame
import time
from grid_world import SimpleGridWorld
from agent import QLearningAgent
from grid_world_sim import GridWorldEnv

# Training loop
env = SimpleGridWorld(8)
agent = QLearningAgent(8, 4)
sim_env = GridWorldEnv(8)

episodes = 1000
all_rewards = []

try:
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(env.actions[action])
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            
            # Visualize every 100th episode
            if episode % 100 == 0:
                try:
                    sim_env.render(state, env.goal)
                    time.sleep(0.1)  # Add a small delay to see the movement
                except pygame.error:
                    print("Pygame window was closed. Continuing without visualization.")
                    sim_env = None
        
        all_rewards.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

except KeyboardInterrupt:
    print("Training interrupted by user.")

finally:
    if sim_env:
        sim_env.close()

# Plot learning curve
plt.plot(range(len(all_rewards)), all_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curve')
plt.savefig('learning_curve.png')
plt.close()

# Test the trained agent
print("\nTesting the trained agent:")
state = env.reset()
done = False

# Reinitialize the visualization environment for testing
sim_env = GridWorldEnv(8)

try:
    while not done:
        sim_env.render(state, env.goal)
        action = agent.get_action(state)
        state, reward, done = env.step(env.actions[action])
        time.sleep(0.5)  # Add a delay to see the movement clearly

    print("Goal reached!")

except pygame.error:
    print("Pygame window was closed. Visualization stopped.")

finally:
    if sim_env:
        sim_env.close()