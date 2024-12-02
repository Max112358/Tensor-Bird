# bird.py
import pygame
import colorsys
import random
from constants import BIRD_SIZE, GRAVITY, BIRD_JUMP_VELOCITY

class Bird:
    def __init__(self, x, y):
        original_image = pygame.image.load('art/bird.png')
        self.original_image = pygame.transform.scale(original_image, (BIRD_SIZE, BIRD_SIZE))
        self.bird_img = self.original_image.copy()
        self.width = BIRD_SIZE
        self.height = BIRD_SIZE
        
        random_hue = random.random()
        
        for x in range(self.width):
            for y in range(self.height):
                r, g, b, a = self.bird_img.get_at((x, y))
                
                if a == 0:
                    continue
                    
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                new_h = (h + random_hue) % 1.0
                new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, s, v)
                
                self.bird_img.set_at((x, y), 
                    (int(new_r*255), int(new_g*255), int(new_b*255), a))
        
        self.x = x
        self.y = y
        self.velocity = 0
        self.time = 0
        self.angle = 0
        self.max_upward_angle = 45
        self.max_downward_angle = -90
        self.rotation_speed = 8
        
    def move(self):
        self.time += 1
        displacement = self.velocity * self.time + 0.5 * GRAVITY * self.time ** 2
        self.y += displacement
        
        current_velocity = self.velocity + GRAVITY * self.time
        
        if current_velocity < 0:
            target_angle = self.max_upward_angle
        else:
            fall_ratio = min(current_velocity / 30.0, 1.0)
            target_angle = self.max_downward_angle * fall_ratio
        
        if self.angle < target_angle:
            self.angle = min(target_angle, self.angle + self.rotation_speed)
        elif self.angle > target_angle:
            self.angle = max(target_angle, self.angle - self.rotation_speed)
            
    def jump(self):
        self.velocity = BIRD_JUMP_VELOCITY
        self.time = 0
        
    def draw(self, screen):
        rotated_bird = pygame.transform.rotate(self.bird_img, self.angle)
        new_rect = rotated_bird.get_rect(center=(self.x + self.width//2, 
                                                self.y + self.height//2))
        screen.blit(rotated_bird, new_rect.topleft)