# background.py
import pygame
from constants import SCALE, PIPE_VELOCITY

class Background:
    def __init__(self, screen_width, screen_height):
        # Load and scale background image
        self.background_img = pygame.image.load('art/background.png')
        
        # Scale background to match screen height while maintaining aspect ratio
        bg_aspect_ratio = self.background_img.get_width() / self.background_img.get_height()
        self.background_img = pygame.transform.scale(
            self.background_img, 
            (int(screen_height * bg_aspect_ratio), screen_height)
        )
        
        # Set up scrolling parameters
        self.width = self.background_img.get_width()
        self.x1 = 0
        self.x2 = self.width  # Second image starts where first image ends
        
        # Make background move slower than pipes for parallax effect
        # Background moves at 40% of pipe speed
        self.velocity = PIPE_VELOCITY * 0.4
        
    def move(self):
        # Move both images to the left
        self.x1 -= self.velocity
        self.x2 -= self.velocity
        
        # If an image has moved completely off screen to the left,
        # move it back to the right edge
        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width
        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width
            
    def draw(self, screen):
        screen.blit(self.background_img, (self.x1, 0))
        screen.blit(self.background_img, (self.x2, 0))