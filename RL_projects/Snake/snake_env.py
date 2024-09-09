import pygame
import numpy as np
import random

class SnakeEnv:
    def __init__(self, width=400, height=400, grid_size=20):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.reset()
        
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake RL")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = random.choice([(0, -self.grid_size), (0, self.grid_size), (-self.grid_size, 0), (self.grid_size, 0)])
        self.apple = self.spawn_apple()
        self.score = 0
        self.steps = 0
        return self.get_state()

    def spawn_apple(self):
        while True:
            apple = (random.randint(0, (self.width - self.grid_size) // self.grid_size) * self.grid_size,
                     random.randint(0, (self.height - self.grid_size) // self.grid_size) * self.grid_size)
            if apple not in self.snake:
                return apple

    def get_state(self):
        head = self.snake[0]
        point_l = (head[0] - self.grid_size, head[1])
        point_r = (head[0] + self.grid_size, head[1])
        point_u = (head[0], head[1] - self.grid_size)
        point_d = (head[0], head[1] + self.grid_size)
        
        dir_l = self.direction == (-self.grid_size, 0)
        dir_r = self.direction == (self.grid_size, 0)
        dir_u = self.direction == (0, -self.grid_size)
        dir_d = self.direction == (0, self.grid_size)

        state = [
            (dir_l and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),

            (dir_u and self.is_collision(point_l)) or
            (dir_d and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_d)) or
            (dir_r and self.is_collision(point_u)),

            (dir_d and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_r)) or
            (dir_r and self.is_collision(point_d)) or
            (dir_l and self.is_collision(point_u)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            self.apple[0] < head[0],  # apple is left
            self.apple[0] > head[0],  # apple is right
            self.apple[1] < head[1],  # apple is up
            self.apple[1] > head[1]   # apple is down
        ]
        return np.array(state, dtype=int)

    def is_collision(self, point):
        return (point in self.snake[1:] or
                point[0] < 0 or point[0] >= self.width or
                point[1] < 0 or point[1] >= self.height)

    def step(self, action):
        # 0: left, 1: right, 2: straight
        if action == 0:
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 1:
            self.direction = (-self.direction[1], self.direction[0])

        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, new_head)
        self.steps += 1
        
        reward = 0
        done = False
        if self.is_collision(new_head):
            reward = -10
            done = True
        elif new_head == self.apple:
            self.score += 1
            reward = 10
            self.apple = self.spawn_apple()
        else:
            self.snake.pop()
        
        if self.steps > 100*len(self.snake):
            reward = -10
            done = True

        return self.get_state(), reward, done

    def render(self):
        self.screen.fill((0, 0, 0))
        for segment in self.snake:
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(segment[0], segment[1], self.grid_size, self.grid_size))
        pygame.draw.rect(self.screen, (255, 0, 0), pygame.Rect(self.apple[0], self.apple[1], self.grid_size, self.grid_size))
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pygame.quit()