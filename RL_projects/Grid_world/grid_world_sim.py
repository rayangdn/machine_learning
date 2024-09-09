import pygame
import time

class GridWorldEnv:
    def __init__(self, grid_size, cell_size=100):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid World")
        
        self.robot_img = pygame.Surface((self.cell_size, self.cell_size))
        self.robot_img.fill((255, 0, 0))  # Red square for the robot
        
        self.goal_img = pygame.Surface((self.cell_size, self.cell_size))
        self.goal_img.fill((0, 255, 0))  # Green square for the goal

    def render(self, state, goal):
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (200, 200, 200), (0, i * self.cell_size), (self.width, i * self.cell_size))
            pygame.draw.line(self.screen, (200, 200, 200), (i * self.cell_size, 0), (i * self.cell_size, self.height))
        
        # Draw goal
        self.screen.blit(self.goal_img, (goal[1] * self.cell_size, goal[0] * self.cell_size))
        
        # Draw robot
        self.screen.blit(self.robot_img, (state[1] * self.cell_size, state[0] * self.cell_size))
        
        pygame.display.flip()

    def close(self):
        pygame.quit()