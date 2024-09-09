import pygame
import math
import numpy as np

class RacingEnv:
    def __init__(self):
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("RL Racing Game")

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)

        # Car properties
        self.car_img = pygame.Surface((20, 40), pygame.SRCALPHA)
        pygame.draw.polygon(self.car_img, self.RED, [(10, 0), (0, 40), (20, 40)])
        self.car_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.car_angle = 0
        self.car_speed = 0

        # Track
        self.track_outer = [(100, 100), (700, 100), (700, 500), (100, 500)]
        self.track_inner = [(200, 200), (600, 200), (600, 400), (200, 400)]

        # RL-specific
        self.action_space = 4  # 0: Accelerate, 1: Brake, 2: Turn left, 3: Turn right
        self.observation_space = 5  # x, y, angle, speed, distance to nearest wall

    def reset(self):
        self.car_pos = [self.WIDTH // 2, self.HEIGHT // 2]
        self.car_angle = 0
        self.car_speed = 0
        return self.get_state()

    def step(self, action):
        self._move_car(action)
        
        new_state = self.get_state()
        reward = self.get_reward()
        done = self.is_done()
        
        return new_state, reward, done, {}

    def _move_car(self, action):
        if action == 0:  # Accelerate
            self.car_speed = min(self.car_speed + 0.5, 10)
        elif action == 1:  # Brake
            self.car_speed = max(self.car_speed - 0.5, 0)
        elif action == 2:  # Turn left
            self.car_angle += 15
        elif action == 3:  # Turn right
            self.car_angle -= 15
        
        rad = math.radians(self.car_angle)
        self.car_pos[0] += self.car_speed * math.sin(rad)
        self.car_pos[1] -= self.car_speed * math.cos(rad)

    def get_state(self):
        distance = self._distance_to_nearest_wall()
        return np.array([self.car_pos[0], self.car_pos[1], self.car_angle, self.car_speed, distance])

    def get_reward(self):
        reward = self.car_speed * 0.1
        if self._is_collision():
            reward -= 5
        return reward

    def is_done(self):
        return self._is_collision()

    def _is_collision(self):
        return not (self._point_inside_polygon(self.car_pos, self.track_outer) and
                    not self._point_inside_polygon(self.car_pos, self.track_inner))

    def _point_inside_polygon(self, point, polygon):
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _distance_to_nearest_wall(self):
        return min(self._distance_to_line(self.car_pos, self.track_outer[i], self.track_outer[(i+1)%4]) 
                   for i in range(4))

    def _distance_to_line(self, point, line_start, line_end):
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        return abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / math.sqrt((y2-y1)**2 + (x2-x1)**2)

    def render(self):
        self.screen.fill(self.BLACK)
        self._draw_track()
        
        rotated_car = pygame.transform.rotate(self.car_img, self.car_angle)
        car_rect = rotated_car.get_rect(center=self.car_pos)
        self.screen.blit(rotated_car, car_rect)
        
        pygame.display.flip()

    def _draw_track(self):
        pygame.draw.lines(self.screen, self.WHITE, True, self.track_outer, 2)
        pygame.draw.lines(self.screen, self.WHITE, True, self.track_inner, 2)

    def close(self):
        pygame.quit()