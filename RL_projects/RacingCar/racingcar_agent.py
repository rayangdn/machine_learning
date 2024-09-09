import numpy as np

class QLearningAgent:
    def __init__(self, action_space, state_space, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0, exploration_decay=0.9995):
        self.action_space = action_space
        self.state_space = state_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        self.q_table = np.zeros((state_space, action_space))

    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_space)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_exploration(self):
        self.exploration_rate = max(0.01, self.exploration_rate * self.exploration_decay)

def discretize_state(state):
    x, y, angle, speed, distance = state
    x_bin = min(int(x / 40), 19)  # 20 bins for x
    y_bin = min(int(y / 30), 19)  # 20 bins for y
    angle_bin = min(int((angle % 360) / 45), 7)  # 8 bins for angle
    speed_bin = min(int(speed / 2), 4)  # 5 bins for speed
    distance_bin = min(int(distance / 20), 4)  # 5 bins for distance
    return x_bin * 8000 + y_bin * 400 + angle_bin * 50 + speed_bin * 10 + distance_bin