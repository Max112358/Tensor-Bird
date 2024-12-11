# player_car.py
import numpy as np
from car import Car
from constants import (
    PLAYER_COLOR, MAX_VELOCITY, MIN_VELOCITY,
    MAX_ACCELERATION, MAX_DECELERATION,
    DEBUG_COLOR, DEBUG_FONT_SIZE, SHOW_DEBUG_INFO,
    NUM_LANES
)
import pygame
from ai_input_processor import get_car_inputs

class PlayerCar(Car):
    def __init__(self, x, y, brain=None):
        """Initialize player car with NEAT neural network brain"""
        super().__init__(x, y)
        self.brain = brain
        self.fixed_x = x
        
        # Ensure initial position is exactly at lane center
        if hasattr(self, 'target_y'):
            self.y = self.target_y
            
        self.color = PLAYER_COLOR
        
        # Performance metrics
        self.total_distance = 0
        self.total_reward = 0
        self.actions_taken = 0
        self.lane_changes = 0
        self.collisions = 0
        self.avg_speed = MIN_VELOCITY
        self.max_speed_achieved = MIN_VELOCITY
        self.min_speed_achieved = MAX_VELOCITY
        
        # Fitness tracking
        self.fitness = 0
        self.time_alive = 0
        self.smooth_driving_score = 0
        self.last_acceleration = 0
        self.last_lane_change = 0
        
        # Lane tracking
        self.current_lane = None
        self.target_lane = None
        self.is_changing_lanes = False
        
        # Debug font setup
        if SHOW_DEBUG_INFO:
            self.debug_font = pygame.font.Font(None, DEBUG_FONT_SIZE)
    
    def think(self, traffic_manager):
        """Use NEAT brain to make driving decisions"""
        if not self.brain:
            return 0, 0
            
        # Get normalized inputs
        inputs = get_car_inputs(self, traffic_manager)
        
        # Get neural network outputs
        outputs = self.brain.activate(inputs)
        
        # Parse outputs [acceleration, lane_change]
        acceleration = np.clip(outputs[0], -1, 1)
        lane_change = np.clip(outputs[1], -1, 1)
        
        return acceleration, lane_change
    
    def update(self, traffic_manager=None, dt=1/60):
        """Update car position and track statistics"""
        prev_relative_x = self.relative_x
        
        if traffic_manager and self.brain:
            # Get AI decisions
            acceleration, lane_change = self.think(traffic_manager)
            
            # Apply acceleration
            self.accelerate(acceleration)
            
            # Handle lane changes if output is strong enough
            if abs(lane_change) > 0.5 and not self.is_changing_lanes:
                # Get current lane safely
                current_lane = traffic_manager._get_lane(self.y)
                if current_lane is not None:  # Only proceed if we're in a valid lane
                    # Calculate target lane
                    direction = 1 if lane_change > 0 else -1
                    target_lane = current_lane + direction
                    
                    # Verify target lane is valid
                    if 0 <= target_lane < NUM_LANES:
                        # Get target y position for the lane
                        target_y = traffic_manager._get_lane_y(target_lane)
                        self.move_toward_y(target_y, self.width * 0.1)
                        self.is_changing_lanes = True
                        self.target_lane = target_lane
        
        # Update velocity with acceleration
        self.velocity += self.acceleration
        self.velocity = np.clip(self.velocity, MIN_VELOCITY, MAX_VELOCITY)
        
        # Update relative position (true position in world)
        self.relative_x += self.velocity
        
        # Update screen position based on world offset
        if traffic_manager:
            self.x = self.relative_x - traffic_manager.world_offset
        
        # Check boundaries and destroy if out of bounds
        if self.check_boundaries():
            self.handle_collision()
            return
        
        # Update statistics
        distance_moved = self.relative_x - prev_relative_x
        self.total_distance += distance_moved
        
        # Update lane change status
        if self.is_changing_lanes and abs(self.y - self.target_y) < 1:
            self.is_changing_lanes = False
            self.current_lane = self.target_lane
    
    def _update_fitness(self):
        """Calculate fitness score based on multiple factors"""
        # Base fitness from distance and time alive
        self.fitness = self.total_distance * 1.0
        
        # Bonus for maintaining higher speeds
        speed_bonus = (self.avg_speed / MAX_VELOCITY) * 0.5
        self.fitness += speed_bonus * self.total_distance
        
        # Bonus for smooth driving
        smoothness_bonus = self.smooth_driving_score * 0.1
        self.fitness += smoothness_bonus
        
        # Heavy penalties for collisions
        collision_penalty = self.collisions * 1000
        self.fitness -= collision_penalty
        
        # Ensure fitness doesn't go negative
        self.fitness = max(0, self.fitness)
    
    def handle_collision(self):
        """Handle collision event"""
        self.collisions += 1
        
        # Debug collision info
        print(f"\nCar {self.genome_id} died:")
        print(f"Position: x={self.x:.1f}, y={self.y:.1f}")
        print(f"Relative position: {self.relative_x:.1f}")
        print(f"Velocity: {self.velocity:.1f}")
        print(f"Time alive: {self.time_alive:.1f} seconds")
        print(f"Distance traveled: {self.total_distance:.1f}")
        
        if self.y < self.top_boundary:
            print("Cause: Hit top boundary")
        elif self.y > self.bottom_boundary:
            print("Cause: Hit bottom boundary")
        elif self.x < self.left_boundary:
            print("Cause: Too far left")
        elif self.x > self.right_boundary:
            print("Cause: Too far right")
        else:
            print("Cause: Collision with NPC car")
            
        self._update_fitness()
        self.is_active = False  # Mark car as inactive after collision
        
    def check_boundaries(self):
        """Check if car is outside screen boundaries with debug info"""
        print(f"\nDEBUG: Boundary check for car {self.genome_id}")
        print(f"Car position: ({self.x}, {self.y})")
        print(f"Car boundaries: top={self.top_boundary}, bottom={self.bottom_boundary}")
        print(f"Car dimensions: width={self.width}, length={self.length}")
        
        # Check each boundary with debug output
        if self.y - self.width/2 < self.top_boundary:
            print(f"Top boundary violation: {self.y - self.width/2} < {self.top_boundary}")
            return True
        if self.y + self.width/2 > self.bottom_boundary:
            print(f"Bottom boundary violation: {self.y + self.width/2} > {self.bottom_boundary}")
            return True
        if self.x < self.left_boundary:
            print(f"Left boundary violation: {self.x} < {self.left_boundary}")
            return True
        if self.x > self.right_boundary:
            print(f"Right boundary violation: {self.x} > {self.right_boundary}")
            return True
            
        print("All boundaries OK")
        return False
