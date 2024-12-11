# car.py
import pygame
import numpy as np
from constants import (
    CAR_LENGTH, CAR_WIDTH, MAX_VELOCITY, MIN_VELOCITY,
    MAX_ACCELERATION, MAX_DECELERATION, NPC_COLOR
)

class Car:
    def __init__(self, x, y):
        """
        Initialize a car with position and physics properties.
        
        Args:
            x (float): Initial x position
            y (float): Initial y position
        """
        # Position
        self.x = float(x)  # Screen position
        self.relative_x = float(x)  # World position
        self.y = float(y)
        self.target_y = self.y
        
        # Screen boundaries
        self.left_boundary = 0
        self.right_boundary = 0  # Will be set to SCREEN_WIDTH
        self.top_boundary = 0
        self.bottom_boundary = 0
        
        # Physics
        self.velocity = MIN_VELOCITY
        self.acceleration = 0
        
        # State
        self.is_changing_lanes = False
        self.collision = False
        self.is_active = True
        
        # Visual properties
        self.color = NPC_COLOR
        self.width = CAR_WIDTH
        self.length = CAR_LENGTH

    def check_boundaries(self):
        """Check if car is outside screen boundaries"""
        if (self.x < self.left_boundary - self.length or 
            self.x > self.right_boundary + self.length or
            self.y < self.top_boundary - self.width/2 or 
            self.y > self.bottom_boundary + self.width/2):
            self.destroy()
            return True
        return False

    def update(self, world_offset=0):
        """
        Update car's position and velocity.
        
        Args:
            world_offset (float): Current world offset for position calculations
        """
        # Update velocity with acceleration
        self.velocity += self.acceleration
        self.velocity = np.clip(self.velocity, MIN_VELOCITY, MAX_VELOCITY)
        
        # Update relative position (actual position in world)
        self.relative_x += self.velocity
        
        # Update screen position based on world offset
        self.x = self.relative_x - world_offset
        
        # Check if car is out of bounds
        self.check_boundaries()
        
    def accelerate(self, amount):
        """
        Apply acceleration within vehicle limits.
        
        Args:
            amount (float): Acceleration amount (-1 to 1)
        """
        # Clamp acceleration input
        amount = np.clip(amount, -1, 1)
        
        # Apply acceleration based on whether we're speeding up or slowing down
        if amount >= 0:
            self.acceleration = amount * MAX_ACCELERATION
        else:
            self.acceleration = amount * MAX_DECELERATION
            
    def move_toward_y(self, target_y, speed):
        """
        Move towards a target y position at given speed.
        Ensures precise stopping at target position.
        
        Args:
            target_y (float): Target y position
            speed (float): Speed of lateral movement
        """
        if abs(self.y - target_y) > 0.1:  # Add small threshold to prevent oscillation
            self.is_changing_lanes = True
            self.target_y = target_y
            
            # Move toward target
            diff = target_y - self.y
            if abs(diff) < speed:
                self.y = target_y  # Snap to exact position
                self.is_changing_lanes = False
            else:
                direction = np.sign(diff)
                self.y += direction * speed
        else:
            self.y = target_y  # Ensure exact positioning
            self.is_changing_lanes = False
                
    def get_rect(self):
        """Get collision rectangle"""
        return pygame.Rect(
            self.x - self.length/2,  # Center the rectangle on the car's screen position
            self.y - self.width/2,
            self.length,
            self.width
        )
        
    def draw(self, screen):
        """Draw the car on the screen"""
        # Create rectangle for car body
        rect = self.get_rect()
        
        # Draw the car
        pygame.draw.rect(screen, self.color, rect)
        
        # Add direction indicator (darker rectangle at front of car)
        front_rect = pygame.Rect(
            rect.x + rect.width * 0.7,  # Front 30% of car
            rect.y,
            rect.width * 0.3,
            rect.height
        )
        # Darken the base color for the front rectangle
        darker_color = tuple(max(0, c - 50) for c in self.color)
        pygame.draw.rect(screen, darker_color, front_rect)
        
    def get_state(self):
        """Get the current state of the car for the AI"""
        return {
            'x': self.x,
            'relative_x': self.relative_x,
            'y': self.y,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'is_changing_lanes': self.is_changing_lanes,
            'collision': self.collision
        }
        
    def destroy(self):
        """Mark the car as inactive (for cleanup)"""
        self.is_active = False