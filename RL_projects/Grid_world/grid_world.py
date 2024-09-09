class SimpleGridWorld:
    def __init__(self, size=5):
        self.size = size
        self.state = [0, 0]  # Start at (0, 0)
        self.goal = [size-1, size-1]  # Goal at bottom-right corner
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    def reset(self):
        self.state = [0, 0]
        return self.state

    def step(self, action):
        new_state = [
            max(0, min(self.size-1, self.state[0] + action[0])),
            max(0, min(self.size-1, self.state[1] + action[1]))
        ]
        self.state = new_state
        
        done = (self.state == self.goal)
        reward = 10 if done else -1
        
        return self.state, reward, done

    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if [i, j] == self.state:
                    print('R', end=' ')
                elif [i, j] == self.goal:
                    print('G', end=' ')
                else:
                    print('.', end=' ')
            print()
        print()