from snake_env import SnakeEnv
from snake_agent import SnakeAgent
import numpy as np
import matplotlib.pyplot as plt
import pickle

def train_snake(episodes, max_steps=1000):
    env = SnakeEnv()
    agent = SnakeAgent(state_size=11, action_size=3)
    scores = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if episode % 5000 == 0:
                env.render()

            if done:
                break

        scores.append(total_reward)
        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

    env.close()
    
    # Save the trained Q-table
    with open('snake_qtable.pkl', 'wb') as f:
        pickle.dump(agent.q_table, f)
    
    return scores, agent

# Train the agent
episodes = 10000  # Increased number of episodes
scores, trained_agent = train_snake(episodes)

# Plot the scores
plt.figure(figsize=(10, 5))
plt.plot(range(len(scores)), scores)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Snake RL Training Progress')
plt.savefig('snake_training_progress.png')
plt.close()

# Test the trained agent
def test_trained_agent(agent, num_games=5):
    env = SnakeEnv()
    agent.epsilon = 0  # Use only learned policy

    for game in range(num_games):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:  # Added step limit to prevent infinite loops
            action = agent.get_action(state)
            state, reward, done = env.step(action)
            total_reward += reward
            env.render()
            steps += 1

        print(f"Game {game + 1} - Score: {total_reward}, Steps: {steps}")

    env.close()

# Load the trained Q-table
with open('snake_qtable.pkl', 'rb') as f:
    trained_agent.q_table = pickle.load(f)

# Test the trained agent
test_trained_agent(trained_agent)