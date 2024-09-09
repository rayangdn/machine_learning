from racingcar_env import RacingEnv
from racingcar_agent import QLearningAgent, discretize_state

def main():
    env = RacingEnv()
    agent = QLearningAgent(env.action_space, 100000)  # Adjust state space size based on discretization

    num_episodes = 10000
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and steps < 1000:  # Added step limit to prevent infinite loops
            discrete_state = discretize_state(state)
            action = agent.get_action(discrete_state)
            next_state, reward, done, _ = env.step(action)
            next_discrete_state = discretize_state(next_state)
            
            agent.update(discrete_state, action, reward, next_discrete_state)
            state = next_state
            total_reward += reward
            steps += 1
            
            if episode % 5 == 0:  # Render more frequently
                env.render()
        
        agent.decay_exploration()
        
        if episode % 5 == 0:  # Print more frequently
            print(f"Episode {episode}, Steps: {steps}, Total Reward: {total_reward:.2f}, Exploration Rate: {agent.exploration_rate:.2f}")

    env.close()

if __name__ == "__main__":
    main()