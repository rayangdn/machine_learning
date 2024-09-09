import pygame
import random

# Game constants
WIDTH = 400
HEIGHT = 600
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

# Game parameters
BIRD_WIDTH = 40
BIRD_HEIGHT = 30
GRAVITY = 1
FLAP_STRENGTH = -10
PIPE_WIDTH = 70
PIPE_GAP = 200
PIPE_FREQUENCY = 1500

class Bird(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BIRD_WIDTH, BIRD_HEIGHT))
        self.image.fill((255, 255, 0))
        self.rect = self.image.get_rect()
        self.rect.center = (WIDTH // 4, HEIGHT // 2)
        self.velocity = 0

    def update(self):
        self.velocity += GRAVITY
        self.rect.y += self.velocity
        if self.rect.top <= 0:
            self.rect.top = 0
            self.velocity = 0
        if self.rect.bottom >= HEIGHT:
            self.rect.bottom = HEIGHT
            self.velocity = 0

    def flap(self):
        self.velocity = FLAP_STRENGTH

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, height, is_top=False):
        super().__init__()
        self.image = pygame.Surface((PIPE_WIDTH, height))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        if is_top:
            self.rect.bottomleft = (x, height)
        else:
            self.rect.topleft = (x, HEIGHT - height)

    def update(self):
        self.rect.x -= 5
        if self.rect.right < 0:
            self.kill()

class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Flappy Bird RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.reset()

    def reset(self):
        self.bird = Bird()
        self.pipes = pygame.sprite.Group()
        self.all_sprites = pygame.sprite.Group(self.bird)
        self.score = 0
        self.frame_iteration = 0
        return self._get_state()

    def _get_state(self):
        if not self.pipes:
            pipe_top = pipe_bottom = WIDTH
            pipe_y = HEIGHT // 2
        else:
            next_pipe = min([p for p in self.pipes if p.rect.right > self.bird.rect.left], key=lambda p: p.rect.right)
            pipe_top = next_pipe.rect.left
            pipe_y = next_pipe.rect.top if next_pipe.rect.top > 0 else next_pipe.rect.bottom
            pipe_bottom = HEIGHT - pipe_y if next_pipe.rect.top > 0 else pipe_y

        return [
            self.bird.rect.y / HEIGHT,
            self.bird.velocity / 10,
            pipe_top / WIDTH,
            pipe_y / HEIGHT,
        ]

    def step(self, action):
        reward = 0.1
        done = False

        if action == 1:
            self.bird.flap()

        self.all_sprites.update()

        if self.frame_iteration % 60 == 0:
            self._spawn_pipe()

        self.frame_iteration += 1

        for pipe in self.pipes:
            if pipe.rect.right < self.bird.rect.left and not hasattr(pipe, 'scored'):
                self.score += 0.5
                pipe.scored = True
                reward = 1

        if pygame.sprite.spritecollide(self.bird, self.pipes, False) or self.bird.rect.top <= 0 or self.bird.rect.bottom >= HEIGHT:
            reward = -10
            done = True

        return self._get_state(), reward, done

    def _spawn_pipe(self):
        height = random.randint(100, HEIGHT - PIPE_GAP - 100)
        top_pipe = Pipe(WIDTH, height, is_top=True)
        bottom_pipe = Pipe(WIDTH, HEIGHT - height - PIPE_GAP)
        self.pipes.add(top_pipe, bottom_pipe)
        self.all_sprites.add(top_pipe, bottom_pipe)

    def render(self):
        self.screen.fill(WHITE)
        self.all_sprites.draw(self.screen)
        score_text = self.font.render(f"Score: {int(self.score)}", True, BLACK)
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()

    def close(self):
        pygame.quit()