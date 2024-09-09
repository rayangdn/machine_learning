import numpy as np

class SnakeAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}

    def get_action(self, state):
        state = tuple(state)
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.get_q_values(state))

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        return self.q_table[state]

    def update(self, state, action, reward, next_state, done):
        state = tuple(state)
        next_state = tuple(next_state)
        current_q = self.get_q_values(state)[action]
        next_max_q = np.max(self.get_q_values(next_state))
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q - current_q)
        self.get_q_values(state)[action] = new_q

        if done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)