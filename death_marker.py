# death_marker.py
import pygame
from constants import PIPE_VELOCITY, PIPE_WIDTH

class DeathMarker:
    def __init__(self, x, y):
        """
        Initialize a death marker at the specified coordinates.
        The marker is centered on the death location.
        
        Args:
            x (int): X coordinate of death location
            y (int): Y coordinate of death location
        """
        original_image = pygame.image.load('art/red_x.png')
        self.size = 80  # Size of the marker
        self.image = pygame.transform.scale(original_image, (self.size, self.size))
        # Center the X on death location
        self.x = x - self.size // 2
        self.y = y - self.size // 2
        
    def move(self):
        """Move the death marker left at the same speed as pipes"""
        self.x -= PIPE_VELOCITY
        
    def is_offscreen(self):
        """Check if the death marker has moved completely off screen"""
        return self.x + self.size < -PIPE_WIDTH
        
    def draw(self, screen):
        """
        Draw the death marker on the screen.
        
        Args:
            screen: Pygame surface to draw on
        """
        screen.blit(self.image, (self.x, self.y))