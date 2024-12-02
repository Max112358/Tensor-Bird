# pipe.py
import pygame
import random
from constants import SCREEN_HEIGHT, PIPE_WIDTH, PIPE_GAP, PIPE_VELOCITY

class Pipe:
    def __init__(self, x):
        self.UP_PIPE_IMG = pygame.image.load('art/purple_pipe.png')
        self.DOWN_PIPE_IMG = pygame.image.load('art/purple_pipe.png')
        
        up_pipe_aspect_ratio = self.UP_PIPE_IMG.get_height() / self.UP_PIPE_IMG.get_width()
        down_pipe_aspect_ratio = self.DOWN_PIPE_IMG.get_height() / self.DOWN_PIPE_IMG.get_width()
        
        self.UP_PIPE_IMG = pygame.transform.scale(self.UP_PIPE_IMG, 
                                       (PIPE_WIDTH, int(PIPE_WIDTH * up_pipe_aspect_ratio)))
        self.DOWN_PIPE_IMG = pygame.transform.scale(self.DOWN_PIPE_IMG, 
                                         (PIPE_WIDTH, int(PIPE_WIDTH * down_pipe_aspect_ratio)))
        
        self.x = x
        self.height = random.randrange(300, 800)
        self.passed = False
        
        self.top_y = self.height - self.DOWN_PIPE_IMG.get_height()
        self.bottom_y = self.height + PIPE_GAP
        
        self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        self.bottom_rect = pygame.Rect(self.x, self.bottom_y, PIPE_WIDTH, SCREEN_HEIGHT - self.bottom_y)
        
    def move(self):
        self.x -= PIPE_VELOCITY
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x
        
    def draw(self, screen):
        screen.blit(self.DOWN_PIPE_IMG, (self.x, self.top_y))
        screen.blit(self.UP_PIPE_IMG, (self.x, self.bottom_y))