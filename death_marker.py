# death_marker.py
import pygame
from constants import PIPE_VELOCITY, PIPE_WIDTH, SCALE

class DeathMarker:
    def __init__(self, x, y):
        original_image = pygame.image.load('art/red_x.png')
        self.size = int(80 * SCALE)  # Scale the marker size
        self.image = pygame.transform.scale(original_image, (self.size, self.size))
        # Center the X on death location
        self.x = x - self.size // 2
        self.y = y - self.size // 2
        
    def move(self):
        self.x -= PIPE_VELOCITY
        
    def is_offscreen(self):
        return self.x + self.size < -PIPE_WIDTH
        
    def draw(self, screen):
        screen.blit(self.image, (self.x, self.y))