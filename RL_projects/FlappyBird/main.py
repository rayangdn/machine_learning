import pygame
from flappybird_env import FlappyBirdEnv
from flappybird_agent import Agent
import time
import pickle

# Game parameters
FPS = 30

def train_agent():
    env = FlappyBirdEnv()
    agent = Agent(state_size=4, action_size=2)
    episodes = 500
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            if episode % 100 == 0:
                env.render()
                env.clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return None
        
        if episode % 10 == 0:
            agent.update_target_network()
        
        print(f"Episode: {episode+1}/{episodes}, Score: {env.score:.2f}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    env.close()
    return agent

def watch_agent_play(agent, num_games=5):
    env = FlappyBirdEnv()
    
    for game in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.act(state)
            state, _, done = env.step(action)
            
            env.render()
            env.clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
        
        print(f"Game {game + 1}: Score = {env.score:.2f}")
        
        # Pause for a moment between games
        time.sleep(2)
    
    env.close()

if __name__ == "__main__":
    trained_agent = train_agent()
    
    if trained_agent is not None:
        print("\nTraining complete. Now watching the agent play 5 times.")
        time.sleep(2)  # Pause for 2 seconds before starting the games
        watch_agent_play(trained_agent)