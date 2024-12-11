# traffic_manager.py
import random
from npc_car import NPCCar
from constants import (
    SCREEN_WIDTH, NUM_LANES, SPAWN_DISTANCE, DESPAWN_DISTANCE,
    TRAFFIC_DENSITY, MIN_CAR_SPACING, NUM_CARS_VISIBLE_AHEAD,
    MIN_NPC_VELOCITY, MAX_NPC_VELOCITY
)

class TrafficManager:
    def __init__(self, road_top, road_bottom):
        """Initialize the traffic manager."""
        self.road_top = road_top
        self.road_bottom = road_bottom
        self.lane_height = (road_bottom - road_top) / NUM_LANES
        self.num_lanes = NUM_LANES
        self.cars = []
        
        # Calculate maximum cars based on road length and spacing
        visible_road_length = SCREEN_WIDTH + (2 * SPAWN_DISTANCE)
        self.max_cars_per_lane = int(visible_road_length / MIN_CAR_SPACING)
        self.target_cars_per_lane = int(self.max_cars_per_lane * TRAFFIC_DENSITY)
        
        # Track spawn cooldowns per lane and direction
        self.spawn_cooldowns = {
            'left': [0.0] * NUM_LANES,
            'right': [0.0] * NUM_LANES
        }
        
        self.world_offset = 0
        self.lead_car = None

    def _get_lane_y(self, lane):
        """Convert lane number to y-coordinate center."""
        return self.road_top + (lane * self.lane_height) + (self.lane_height / 2)
    
    def _get_lane(self, y):
        """Convert y-coordinate to lane number."""
        if y < self.road_top or y > self.road_bottom:
            return None
        lane = int((y - self.road_top) / self.lane_height)
        return max(0, min(lane, NUM_LANES - 1))

    def _count_cars_in_lane(self, lane, direction='both'):
        """Count number of cars in the given lane and direction."""
        lane_y = self._get_lane_y(lane)
        return sum(1 for car in self.cars 
                  if (abs(car.y - lane_y) < self.lane_height and 
                      car.is_active and
                      (direction == 'both' or
                       (direction == 'right' and car.velocity > 0) or
                       (direction == 'left' and car.velocity < 0))))

    def _can_spawn_in_lane(self, lane, spawn_x, direction='right'):
        """Check if it's safe to spawn a car at the given position."""
        lane_y = self._get_lane_y(lane)
        safe_distance = MIN_CAR_SPACING * 2  # Double spacing for extra safety
        
        # Check for nearby cars in the same lane
        for car in self.cars:
            if not car.is_active:
                continue
            
            if abs(car.y - lane_y) < self.lane_height:  # Same lane
                # Check distance using relative positions
                if abs(car.relative_x - spawn_x) < safe_distance:
                    return False
                    
        return True

    def _get_spawn_position(self, direction='right'):
        """Get safe spawn position relative to lead car."""
        if direction == 'right':
            # Spawn well ahead of the rightmost visible area
            return self.lead_car.relative_x + SCREEN_WIDTH + SPAWN_DISTANCE
        else:
            # Spawn well behind the leftmost visible area
            return self.lead_car.relative_x - SPAWN_DISTANCE - SCREEN_WIDTH

    def _spawn_car(self, lane, direction='right'):
        """Spawn a new car in the specified lane and direction."""
        # Get safe spawn position
        spawn_relative_x = self._get_spawn_position(direction)
        
        # Only spawn if it's safe
        if self._can_spawn_in_lane(lane, spawn_relative_x, direction):
            # Set appropriate velocity based on direction
            if direction == 'right':
                velocity = random.uniform(MIN_NPC_VELOCITY, MAX_NPC_VELOCITY)
            else:
                velocity = -random.uniform(MIN_NPC_VELOCITY, MAX_NPC_VELOCITY)
            
            # Create new car at spawn position
            screen_x = spawn_relative_x - self.world_offset
            new_car = NPCCar(screen_x, self._get_lane_y(lane))
            new_car.relative_x = spawn_relative_x
            new_car.velocity = velocity
            
            # Set boundaries
            new_car.left_boundary = 0
            new_car.right_boundary = SCREEN_WIDTH
            new_car.top_boundary = self.road_top
            new_car.bottom_boundary = self.road_bottom
            
            self.cars.append(new_car)
            
            # Set spawn cooldown
            self.spawn_cooldowns[direction][lane] = random.uniform(1.0, 3.0)

    def _manage_spawning(self):
        """Manage continuous spawning of traffic in both directions."""
        for lane in range(self.num_lanes):
            # Count cars moving in each direction
            right_moving = self._count_cars_in_lane(lane, 'right')
            left_moving = self._count_cars_in_lane(lane, 'left')
            
            # Target is split between directions
            target_per_direction = self.target_cars_per_lane // 2
            
            # Try spawning from right if needed
            if (right_moving < target_per_direction and 
                self.spawn_cooldowns['right'][lane] <= 0):
                self._spawn_car(lane, 'right')
                    
            # Try spawning from left if needed
            if (left_moving < target_per_direction and 
                self.spawn_cooldowns['left'][lane] <= 0):
                self._spawn_car(lane, 'left')

    def get_nearby_cars(self, x, y, max_distance):
        """Get list of cars near a specific point."""
        nearby = []
        for car in self.cars:
            if not car.is_active:
                continue
            
            dx = car.relative_x - x
            dy = car.y - y
            distance = (dx * dx + dy * dy) ** 0.5
            
            if distance <= max_distance:
                nearby.append((car, distance))
        
        # Sort by distance
        nearby.sort(key=lambda x: x[1])
        return [car for car, _ in nearby]

    def get_closest_car_in_lane(self, lane, reference_x, ahead=True):
        """
        Find closest car in specific lane.
        
        Args:
            lane (int): Lane number to check
            reference_x (float): Reference x coordinate in world space
            ahead (bool): Whether to look ahead or behind
            
        Returns:
            NPCCar or None: Closest car in lane, if any
        """
        lane_y = self._get_lane_y(lane)
        closest_car = None
        closest_distance = float('inf')
        
        for car in self.cars:
            if not car.is_active:
                continue
                
            # Check if car is in the specified lane
            if abs(car.y - lane_y) < self.lane_height:
                # Calculate relative distance using relative positions
                dx = car.relative_x - reference_x
                
                # Check if car is in the desired direction
                if (ahead and dx > 0) or (not ahead and dx < 0):
                    distance = abs(dx)
                    if distance < closest_distance:
                        closest_car = car
                        closest_distance = distance
        
        return closest_car

    def check_collision(self, player_car):
        """Check if player car collides with any NPC cars or boundaries."""
        player_rect = player_car.get_rect()
        
        # Check if player is off road
        if (player_car.y - player_car.width/2 < self.road_top or 
            player_car.y + player_car.width/2 > self.road_bottom):
            return True
            
        # Check collisions with NPC cars
        for car in self.cars:
            if car.is_active and car.get_rect().colliderect(player_rect):
                return True
                
        return False

    def update(self, dt, ai_car_or_cars):
        """Update traffic state."""
        # Handle both single car and multiple cars cases
        ai_cars = [ai_car_or_cars] if not isinstance(ai_car_or_cars, list) else ai_car_or_cars
        
        # Find the lead AI car (furthest ahead)
        self.lead_car = max(ai_cars, key=lambda car: car.relative_x)
        
        # Update world offset based on lead car
        self.world_offset = self.lead_car.relative_x - (SCREEN_WIDTH * 0.2)
        
        # Update spawn cooldowns
        for direction in self.spawn_cooldowns:
            self.spawn_cooldowns[direction] = [max(0, cd - dt) for cd in self.spawn_cooldowns[direction]]
        
        # Clean up cars that are too far away from the lead car
        cleanup_distance = SCREEN_WIDTH + SPAWN_DISTANCE
        self.cars = [car for car in self.cars 
                    if (car.is_active and 
                        abs(car.relative_x - self.lead_car.relative_x) <= cleanup_distance)]
        
        # Update all NPC cars
        for car in self.cars:
            car.update(dt, [*self.cars, *ai_cars], self.road_top, self.road_bottom, self.world_offset)
        
        # Spawn new cars where needed
        self._manage_spawning()

    def spawn_initial_traffic(self):
        """Create initial set of NPC cars, ensuring they're off screen."""
        self.cars.clear()
        
        # Initial reference position (assume no lead car yet)
        reference_x = SCREEN_WIDTH * 0.2  # Where player will start
        
        for lane in range(self.num_lanes):
            # Spawn right-moving traffic ahead
            for i in range(self.target_cars_per_lane // 2):
                # Start spawning beyond right edge of screen
                relative_x = reference_x + SCREEN_WIDTH + SPAWN_DISTANCE + (i * MIN_CAR_SPACING * 2)
                car = NPCCar(relative_x, self._get_lane_y(lane))
                car.relative_x = relative_x
                car.velocity = random.uniform(MIN_NPC_VELOCITY, MAX_NPC_VELOCITY)
                self.cars.append(car)
            
            # Spawn left-moving traffic behind
            for i in range(self.target_cars_per_lane // 2):
                # Start spawning beyond left edge of screen
                relative_x = reference_x - SPAWN_DISTANCE - (i * MIN_CAR_SPACING * 2)
                car = NPCCar(relative_x, self._get_lane_y(lane))
                car.relative_x = relative_x
                car.velocity = -random.uniform(MIN_NPC_VELOCITY, MAX_NPC_VELOCITY)
                self.cars.append(car)

    def draw(self, screen):
        """Draw all active traffic."""
        for car in self.cars:
            if car.is_active:
                car.draw(screen)