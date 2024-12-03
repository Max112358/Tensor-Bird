# bird.py
import pygame
import colorsys
import random
from constants import (BIRD_SIZE, GRAVITY, BIRD_JUMP_VELOCITY, 
                      MAX_FALL_SPEED, SCALE)

class Bird:
    def __init__(self, x, y):
        original_image = pygame.image.load('art/bird.png')
        # Scale bird image according to BIRD_SIZE
        self.original_image = pygame.transform.scale(original_image, 
                                                   (BIRD_SIZE, BIRD_SIZE))
        self.bird_img = self.original_image.copy()
        self.width = BIRD_SIZE
        self.height = BIRD_SIZE
        
        # Generate random color variation for each bird
        random_hue = random.random()
        for px in range(self.width):
            for py in range(self.height):
                r, g, b, a = self.bird_img.get_at((px, py))
                if a == 0:  # Skip transparent pixels
                    continue
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                new_h = (h + random_hue) % 1.0
                new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, s, v)
                self.bird_img.set_at((px, py), 
                    (int(new_r*255), int(new_g*255), int(new_b*255), a))
        
        # Physics properties
        self.x = x
        self.y = y
        self.velocity = 0
        self.terminal_velocity = MAX_FALL_SPEED
        
        # Visual properties - scale rotation speeds
        self.angle = 0
        self.max_upward_angle = 20
        self.max_downward_angle = -90
        self.rotation_speed = 4 * SCALE  # Scale rotation speed
        
    def move(self):
        # Update velocity with gravity and terminal velocity
        self.velocity += GRAVITY
        if self.velocity > self.terminal_velocity:
            self.velocity = self.terminal_velocity
            
        # Update position
        self.y += self.velocity
        
        # Update rotation based on velocity
        if self.velocity < 0:
            target_angle = self.max_upward_angle
        else:
            fall_ratio = min(self.velocity / self.terminal_velocity, 1.0)
            target_angle = self.max_downward_angle * fall_ratio
            
        # Smoothly interpolate to target angle
        if self.angle < target_angle:
            self.angle = min(target_angle, self.angle + self.rotation_speed)
        elif self.angle > target_angle:
            self.angle = max(target_angle, self.angle - self.rotation_speed)
        
    def jump(self):
        self.velocity = BIRD_JUMP_VELOCITY
        
    def draw(self, screen):
        rotated_bird = pygame.transform.rotate(self.bird_img, self.angle)
        new_rect = rotated_bird.get_rect(
            center=(self.x + self.width//2, self.y + self.height//2))
        screen.blit(rotated_bird, new_rect.topleft)