# npc_car.py
import pygame
import random
import numpy as np
from car import Car
from constants import (
    MIN_NPC_VELOCITY, MAX_NPC_VELOCITY,
    MAX_NPC_ACCELERATION, MAX_NPC_DECELERATION,
    NPC_LANE_CHANGE_PROBABILITY, MIN_NPC_FOLLOWING_DISTANCE
)

class NPCCar(Car):
    def __init__(self, x, y):
        """
        Initialize an NPC car with autonomous behavior.
        
        Args:
            x (float): Initial x position
            y (float): Initial y position
        """
        super().__init__(x, y)
        
        # Set random target velocity within NPC limits
        self.target_velocity = random.uniform(MIN_NPC_VELOCITY, MAX_NPC_VELOCITY)
        
        # NPC-specific state
        self.time_until_next_decision = random.uniform(1.0, 3.0)  # Seconds between decisions
        self.current_decision_time = 0
        self.last_lane_change_time = 0
        self.lane_change_cooldown = random.uniform(2.0, 5.0)  # Seconds between lane changes
        
        # Behavior parameters (randomized per car)
        self.aggression = random.uniform(0.5, 1.5)  # Affects following distance and lane change frequency
        self.desired_following_distance = MIN_NPC_FOLLOWING_DISTANCE * self.aggression
        self.lane_change_threshold = NPC_LANE_CHANGE_PROBABILITY * (2 - self.aggression)
        
    def detect_nearby_cars(self, cars, road_y_min, road_y_max):
        """
        Detect cars in immediate vicinity.
        
        Args:
            cars (list): List of all cars in the environment
            road_y_min (float): Top edge of road
            road_y_max (float): Bottom edge of road
            
        Returns:
            dict: Information about nearby cars
        """
        nearby = {
            'front': None,      # Closest car ahead in same lane
            'front_left': None, # Closest car ahead in left lane
            'front_right': None,# Closest car ahead in right lane
            'back': None,       # Closest car behind in same lane
            'back_left': None,  # Closest car behind in left lane
            'back_right': None  # Closest car behind in right lane
        }
        
        car_lane_height = self.width * 1.5  # Define what we consider "same lane"
        
        for car in cars:
            if car is self or not car.is_active:
                continue
                
            # Check if car is within reasonable distance to consider
            # Use relative positions for distance calculations
            dx = car.relative_x - self.relative_x
            if abs(dx) > self.desired_following_distance * 2:
                continue
                
            dy = car.y - self.y
            
            # Determine which lane the car is in relative to us
            vertical_position = ''
            if abs(dy) < car_lane_height:
                vertical_position = 'same'
            elif dy < -car_lane_height and self.y > road_y_min + self.width:
                vertical_position = 'left'
            elif dy > car_lane_height and self.y < road_y_max - self.width:
                vertical_position = 'right'
            else:
                continue
                
            # Determine if car is in front or behind using relative positions
            position = 'front' if dx > 0 else 'back'
            
            # Create key for nearby dict
            key = position if vertical_position == 'same' else f"{position}_{vertical_position}"
            
            # Update if this is the closest car in that position
            if key in nearby:
                if nearby[key] is None or abs(dx) < abs(nearby[key].relative_x - self.relative_x):
                    nearby[key] = car
        
        return nearby
    
    def adjust_velocity(self, nearby_cars):
        """Adjust velocity based on nearby traffic"""
        # Get car directly in front
        front_car = nearby_cars['front']
        
        if front_car:
            # Calculate distance and relative velocity using relative positions
            distance = front_car.relative_x - self.relative_x
            rel_velocity = self.velocity - front_car.velocity
            
            # Simple time-to-collision calculation
            if rel_velocity > 0:  # Only if we're getting closer
                time_to_collision = distance / rel_velocity
                
                if time_to_collision < 1.0:  # Emergency braking
                    self.accelerate(-1.0)
                elif distance < self.desired_following_distance:  # Too close
                    brake_force = min(1.0, (self.desired_following_distance - distance) 
                                    / self.desired_following_distance)
                    self.accelerate(-brake_force)
                else:  # Match speed with some randomness
                    target = front_car.velocity * random.uniform(0.9, 1.1)
                    self.adjust_to_target_velocity(target)
            else:
                self.adjust_to_target_velocity(self.target_velocity)
        else:
            # No car in front, maintain target velocity
            self.adjust_to_target_velocity(self.target_velocity)
    
    def adjust_to_target_velocity(self, target):
        """Smoothly adjust to target velocity"""
        if self.velocity < target:
            self.accelerate(min(1.0, (target - self.velocity) / MAX_NPC_ACCELERATION))
        else:
            self.accelerate(max(-1.0, (target - self.velocity) / MAX_NPC_DECELERATION))
    
    def consider_lane_change(self, nearby_cars, road_y_min, road_y_max):
        """Consider changing lanes based on traffic conditions"""
        if (self.is_changing_lanes or 
            self.current_decision_time - self.last_lane_change_time < self.lane_change_cooldown):
            return None
        
        # Random lane change with probability
        if random.random() < self.lane_change_threshold:
            # Determine available directions
            can_go_left = (self.y > road_y_min + self.width * 2 and 
                          not (nearby_cars['front_left'] or nearby_cars['back_left']))
            can_go_right = (self.y < road_y_max - self.width * 2 and 
                          not (nearby_cars['front_right'] or nearby_cars['back_right']))
            
            if can_go_left and can_go_right:
                return random.choice([-1, 1])
            elif can_go_left:
                return -1
            elif can_go_right:
                return 1
            
        # Consider lane change if car in front is too slow
        front_car = nearby_cars['front']
        if front_car and front_car.velocity < self.target_velocity:
            distance = front_car.relative_x - self.relative_x
            if distance < self.desired_following_distance:
                # Check left lane
                if (self.y > road_y_min + self.width * 2 and 
                    not (nearby_cars['front_left'] or nearby_cars['back_left'])):
                    return -1
                # Check right lane
                if (self.y < road_y_max - self.width * 2 and 
                    not (nearby_cars['front_right'] or nearby_cars['back_right'])):
                    return 1
        
        return 0
    
    def update(self, dt, all_cars, road_y_min, road_y_max, world_offset=0):
        """
        Update NPC car behavior
        
        Args:
            dt (float): Time step in seconds
            all_cars (list): List of all cars in environment
            road_y_min (float): Top edge of road
            road_y_max (float): Bottom edge of road
            world_offset (float): Current world offset for position calculations
        """
        self.current_decision_time += dt
        
        # Get information about nearby cars
        nearby_cars = self.detect_nearby_cars(all_cars, road_y_min, road_y_max)
        
        # Adjust velocity based on traffic
        self.adjust_velocity(nearby_cars)
        
        # Consider lane changes
        if self.current_decision_time >= self.time_until_next_decision:
            lane_change = self.consider_lane_change(nearby_cars, road_y_min, road_y_max)
            # Only process lane change if it's a valid direction (-1, 0, or 1)
            if lane_change is not None and lane_change != 0:
                target_y = self.y + (lane_change * self.width * 1.5)  # Move one lane width
                self.move_toward_y(target_y, self.width * 0.1)  # Smooth lane change
                self.last_lane_change_time = self.current_decision_time
            
            # Reset decision timer with some randomness
            self.time_until_next_decision = random.uniform(1.0, 3.0)
            self.current_decision_time = 0
        
        # Call parent update with world offset
        super().update(world_offset)