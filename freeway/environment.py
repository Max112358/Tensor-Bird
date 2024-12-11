# environment.py
import pygame
import numpy as np
from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, NUM_LANES, LANE_WIDTH, SHOULDER_WIDTH,
    ROAD_COLOR, LANE_COLOR, SHOULDER_COLOR, SPAWN_DISTANCE, DESPAWN_DISTANCE,
    TRAFFIC_DENSITY, MIN_CAR_SPACING, NUM_CARS_VISIBLE_AHEAD,
    BASE_REWARD_PER_FRAME, COLLISION_PENALTY, OFFROAD_PENALTY,
    REWARD_SPEED_MULTIPLIER, LANE_CHANGE_SPEED
)
from car import Car

class Environment:
    def __init__(self):
        """Initialize the freeway environment"""
        # Calculate road dimensions
        self.road_top = SHOULDER_WIDTH
        self.road_bottom = SHOULDER_WIDTH + (NUM_LANES * LANE_WIDTH)
        
        # Initialize lists for tracking cars
        self.player_car = None
        self.npc_cars = []
        
        # Tracking
        self.score = 0
        self.frame_count = 0
        self.distance_traveled = 0
        self.world_offset = 0
        
    def _lane_to_y(self, lane):
        """Convert lane number to y-coordinate"""
        if not 0 <= lane < NUM_LANES:
            raise ValueError(f"Invalid lane number: {lane}")
        return self.road_top + (lane + 0.5) * LANE_WIDTH
    
    def _y_to_lane(self, y):
        """Convert y-coordinate to nearest lane number"""
        if y < self.road_top or y > self.road_bottom:
            return None
        lane = int((y - self.road_top) / LANE_WIDTH)
        return max(0, min(lane, NUM_LANES - 1))
    
    def reset(self):
        """Reset the environment to initial state"""
        self.npc_cars.clear()
        self.score = 0
        self.frame_count = 0
        self.distance_traveled = 0
        self.world_offset = 0
        
        # Create player car in middle lane with fixed screen position
        middle_lane = NUM_LANES // 2
        player_y = self._lane_to_y(middle_lane)
        self.player_car = Car(SCREEN_WIDTH * 0.2, player_y)  # 20% from left
        self.player_car.relative_x = SCREEN_WIDTH * 0.2  # Initial relative position
        
        # Initialize NPC cars
        self._spawn_initial_traffic()
        
        return self.get_state()
    
    def _spawn_initial_traffic(self):
        """Spawn initial set of NPC cars"""
        self.npc_cars.clear()
        
        # Calculate max cars per lane based on spacing
        lane_length = SCREEN_WIDTH + SPAWN_DISTANCE
        max_cars_per_lane = int(lane_length / MIN_CAR_SPACING)
        cars_per_lane = int(max_cars_per_lane * TRAFFIC_DENSITY)
        
        for lane in range(NUM_LANES):
            # Space cars evenly in each lane
            for i in range(cars_per_lane):
                # Calculate relative position with some randomness
                relative_x = (SCREEN_WIDTH * 0.2) + (i * MIN_CAR_SPACING * random.uniform(1.0, 1.5))
                y = self._lane_to_y(lane)
                
                car = Car(relative_x, y)
                car.relative_x = relative_x
                car.x = relative_x  # Initial screen position matches relative position
                self.npc_cars.append(car)
    
    def _handle_car_spawning(self):
        """Manage spawning of new NPC cars"""
        # Count cars in each lane
        cars_in_lane = [0] * NUM_LANES
        for car in self.npc_cars:
            lane = self._y_to_lane(car.y)
            if lane is not None and car.is_active:
                cars_in_lane[lane] += 1
        
        # Spawn new cars where needed
        for lane in range(NUM_LANES):
            if cars_in_lane[lane] < NUM_CARS_VISIBLE_AHEAD:
                # Find rightmost car in this lane
                rightmost_x = self.player_car.relative_x
                lane_y = self._lane_to_y(lane)
                for car in self.npc_cars:
                    if abs(car.y - lane_y) < LANE_WIDTH and car.is_active:
                        rightmost_x = max(rightmost_x, car.relative_x)
                
                # Check if we can spawn
                spawn_relative_x = max(
                    self.player_car.relative_x + SCREEN_WIDTH + SPAWN_DISTANCE,
                    rightmost_x + MIN_CAR_SPACING
                )
                
                # Create new car with correct positions
                car = Car(spawn_relative_x - self.world_offset, lane_y)
                car.relative_x = spawn_relative_x
                self.npc_cars.append(car)
    
    def _check_collisions(self):
        """Check for collisions between cars"""
        if not self.player_car or self.player_car.collision:
            return False
            
        player_rect = self.player_car.get_rect()
        
        # Check if player is off road
        if (self.player_car.y - self.player_car.width/2 < self.road_top or 
            self.player_car.y + self.player_car.width/2 > self.road_bottom):
            self.player_car.collision = True
            return True
        
        # Check collisions with NPC cars
        for car in self.npc_cars:
            if car.is_active and car.get_rect().colliderect(player_rect):
                self.player_car.collision = True
                car.collision = True
                return True
                
        return False
    
    def _calculate_reward(self):
        """Calculate reward for current frame"""
        if self.player_car.collision:
            return COLLISION_PENALTY
            
        # Base reward multiplied by speed
        speed_factor = self.player_car.velocity / self.player_car.MAX_VELOCITY
        reward = BASE_REWARD_PER_FRAME * (1 + REWARD_SPEED_MULTIPLIER * speed_factor)
        
        # Penalty for being partially off road
        player_rect = self.player_car.get_rect()
        if (player_rect.top < self.road_top or player_rect.bottom > self.road_bottom):
            reward += OFFROAD_PENALTY * min(
                abs(player_rect.top - self.road_top),
                abs(player_rect.bottom - self.road_bottom)
            ) / LANE_WIDTH
            
        return reward
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: dict with 'acceleration' (-1 to 1) and 'lane_change' (-1, 0, 1)
            
        Returns:
            tuple: (state, reward, done, info)
        """
        self.frame_count += 1
        
        # Apply actions
        if self.player_car and not self.player_car.collision:
            # Apply acceleration
            self.player_car.accelerate(action['acceleration'])
            
            # Handle lane changes
            if action['lane_change'] != 0:
                current_lane = self._y_to_lane(self.player_car.y)
                if current_lane is not None:
                    target_lane = current_lane + action['lane_change']
                    if 0 <= target_lane < NUM_LANES:
                        target_y = self._lane_to_y(target_lane)
                        self.player_car.move_toward_y(target_y, LANE_CHANGE_SPEED)
        
        # Update world offset based on player movement
        if self.player_car:
            self.player_car.update()
            self.world_offset = self.player_car.relative_x - self.player_car.fixed_x
            self.distance_traveled = max(self.distance_traveled, self.player_car.relative_x)
            
        # Update all NPC cars with world offset
        for car in self.npc_cars:
            car.update(self.world_offset)
        
        # Clean up off-screen cars
        self.npc_cars = [car for car in self.npc_cars 
                        if car.is_active and 
                        car.relative_x > self.player_car.relative_x + DESPAWN_DISTANCE]
        
        # Spawn new cars as needed
        self._handle_car_spawning()
        
        # Check collisions
        collision = self._check_collisions()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Get state
        state = self.get_state()
        
        # Check if episode is done
        done = collision or (self.player_car and not self.player_car.is_active)
        
        info = {
            'score': self.score,
            'distance': self.distance_traveled,
            'collision': collision
        }
        
        return state, reward, done, info
    
    def get_state(self):
        """Get current state for AI input"""
        if not self.player_car:
            return None
            
        # Get player car state
        state = self.player_car.get_state()
        
        # Add lane information
        state['current_lane'] = self._y_to_lane(self.player_car.y)
        
        # Get nearby cars in each lane
        nearby_cars = {lane: [] for lane in range(NUM_LANES)}
        for car in self.npc_cars:
            lane = self._y_to_lane(car.y)
            if lane is not None:
                nearby_cars[lane].append(car)
                
        # Sort cars by relative position and keep only visible ones
        for lane in nearby_cars:
            nearby_cars[lane].sort(key=lambda c: c.relative_x)
            nearby_cars[lane] = nearby_cars[lane][:NUM_CARS_VISIBLE_AHEAD]
            
        # Add car information to state
        state['nearby_cars'] = nearby_cars
        
        return state
    
    def draw(self, screen):
        """Draw the environment"""
        # Draw road and shoulders
        screen.fill(SHOULDER_COLOR)
        road_rect = pygame.Rect(0, self.road_top, SCREEN_WIDTH, 
                              self.road_bottom - self.road_top)
        pygame.draw.rect(screen, ROAD_COLOR, road_rect)
        
        # Draw lane markers
        for lane in range(1, NUM_LANES):
            y = self.road_top + (lane * LANE_WIDTH)
            pygame.draw.line(screen, LANE_COLOR, (0, y), 
                           (SCREEN_WIDTH, y), 2)
        
        # Draw NPC cars
        for car in self.npc_cars:
            car.draw(screen)
            
        # Draw player car
        if self.player_car:
            self.player_car.draw(screen)
            
        pygame.display.flip()