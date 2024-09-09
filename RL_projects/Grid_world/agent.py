import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = np.zeros((state_size, state_size, action_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def get_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update(self, state, action, reward, next_state):
        current_q = self.q_table[state[0], state[1], action]
        next_max_q = np.max(self.q_table[next_state[0], next_state[1]])
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state[0], state[1], action] = new_q