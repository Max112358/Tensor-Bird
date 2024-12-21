from typing import List, Tuple
import numpy as np
import random
from game_init import get_constants

class Terrain:
    def __init__(self, screen_width: int, screen_height: int):
        # Get global constants
        const = get_constants()
        
        self.width = screen_width
        self.height = screen_height
        self.ground_height = const.GROUND_HEIGHT
        self.landing_pad_width = const.LANDING_PAD_WIDTH
        
        # Store points for line segments to enable proper collision detection
        self.segments = []
        
        # Set a fixed seed for reproducibility
        # this is for debugging only
        #np.random.seed(42)
        #random.seed(42)
        
        # Generate initial terrain
        self.points = self._generate()
        self._generate_segments()

    
    def _generate(self) -> List[Tuple[int, int]]:
        """Generate terrain with random height variations and flat landing pad"""
        const = get_constants()
        points = []
        segment_width = 20  # Distance between terrain points
        
        pad_left = self.landing_pad_x - self.landing_pad_width/2
        pad_right = self.landing_pad_x + self.landing_pad_width/2
        
        # Generate left side of terrain
        x = 0
        prev_height = self.ground_height
        while x < pad_left:
            new_height = prev_height + random.randint(-const.TERRAIN_ROUGHNESS, const.TERRAIN_ROUGHNESS)
            new_height = min(max(new_height, self.ground_height - 50), self.ground_height + 50)
            points.append((int(x), int(new_height)))
            prev_height = new_height
            x += segment_width
            
        # Add landing pad (ensure it connects smoothly)
        points.append((int(pad_left), self.ground_height))  # Left edge of pad
        points.append((int(pad_right), self.ground_height)) # Right edge of pad
        
        # Generate right side of terrain
        x = pad_right
        prev_height = self.ground_height
        while x <= self.width:
            new_height = prev_height + random.randint(-const.TERRAIN_ROUGHNESS, const.TERRAIN_ROUGHNESS)
            new_height = min(max(new_height, self.ground_height - 50), self.ground_height + 50)
            points.append((int(x), int(new_height)))
            prev_height = new_height
            x += segment_width
            
        return points
    
    
    '''
    #non random version for debugging
    def _generate(self) -> List[Tuple[int, int]]:
        """Generate terrain with fixed height variations and flat landing pad"""
        points = []
        segment_width = 20  # Distance between terrain points
        
        pad_left = self.landing_pad_x - self.landing_pad_width/2
        pad_right = self.landing_pad_x + self.landing_pad_width/2
        
        # Generate left side of terrain with fixed variations
        x = 0
        while x < pad_left:
            # Create a gentle slope up to the pad
            height_offset = 20 * (x / pad_left)  # Gradual rise
            new_height = self.ground_height - height_offset
            points.append((int(x), int(new_height)))
            x += segment_width
            
        # Add landing pad (ensure it connects smoothly)
        points.append((int(pad_left), self.ground_height))  # Left edge of pad
        points.append((int(pad_right), self.ground_height)) # Right edge of pad
        
        # Generate right side of terrain with fixed variations
        x = pad_right
        while x <= self.width:
            # Create a gentle slope down from the pad
            height_offset = 20 * ((x - pad_right) / (self.width - pad_right))  # Gradual descent
            new_height = self.ground_height - height_offset
            points.append((int(x), int(new_height)))
            x += segment_width
            
        return points
    '''

    def _generate_segments(self):
        """Create line segments from points for collision detection"""
        self.segments = []
        for i in range(len(self.points) - 1):
            self.segments.append((self.points[i], self.points[i + 1]))

    def check_collision(self, x: float, y: float, lander) -> bool:
        """Check if lander collides with terrain"""
        const = get_constants()
        
        # Quick bounds check
        if y >= self.height:
            return True
            
        # Get lander vertices and legs
        vertices = lander.get_vertices()
        left_leg, right_leg = lander.get_leg_positions()
        
        # Combine all points to check
        points_to_check = vertices + [left_leg[1], right_leg[1]]
        
        # Check each point against terrain
        for i, (point_x, point_y) in enumerate(points_to_check):
            point_type = "vertex" if i < len(vertices) else "leg"
            for (x1, y1), (x2, y2) in self.segments:
                if x1 <= point_x <= x2:  # Point is within segment x-range
                    if x2 - x1 == 0:  # Vertical line segment
                        terrain_y = y1
                    else:
                        slope = (y2 - y1) / (x2 - x1)
                        terrain_y = y1 + slope * (point_x - x1)
                    
                    # Add tolerance for leg points during landing
                    tolerance = const.LANDING_PAD_TOLERANCE if point_type == "leg" else 0
                    
                    if point_y > terrain_y + tolerance:
                        return True
                        
        return False

    def check_landing(self, x: float, y: float, velocity_y: float, lander) -> bool:
        """Check if lander has achieved safe landing on pad"""
        const = get_constants()
        
        pad_left = self.landing_pad_x - self.landing_pad_width/2
        pad_right = self.landing_pad_x + self.landing_pad_width/2
        
        # Get leg positions
        left_leg, right_leg = lander.get_leg_positions()
        left_foot = left_leg[1]
        right_foot = right_leg[1]
        
        # Check each landing condition individually
        left_foot_in_bounds = pad_left <= left_foot[0] <= pad_right
        right_foot_in_bounds = pad_left <= right_foot[0] <= pad_right
        feet_in_bounds = left_foot_in_bounds and right_foot_in_bounds
        
        left_foot_height_ok = abs(left_foot[1] - self.ground_height) < const.LANDING_PAD_TOLERANCE
        right_foot_height_ok = abs(right_foot[1] - self.ground_height) < const.LANDING_PAD_TOLERANCE
        feet_at_height = left_foot_height_ok and right_foot_height_ok
        
        velocity_ok = abs(velocity_y) < const.SAFE_LANDING_VELOCITY
        angle_ok = abs(lander.angle) < const.SAFE_LANDING_ANGLE
        
        return (feet_in_bounds and feet_at_height and velocity_ok and angle_ok)
        
    @property
    def landing_pad_x(self):
        """Get x coordinate of landing pad center"""
        # Randomize landing pad position between 20% and 80% of screen width
        if not hasattr(self, '_landing_pad_x'):
            pad_min = int(self.width * 0.2)
            pad_max = int(self.width * 0.8)
            self._landing_pad_x = random.randint(pad_min, pad_max)
            #self._landing_pad_x = pad_max #non random version for debugging
        return self._landing_pad_x