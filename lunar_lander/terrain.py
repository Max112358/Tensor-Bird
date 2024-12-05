from typing import List, Tuple
import numpy as np
import random
from constants import (SAFE_LANDING_VELOCITY, LANDING_PAD_TOLERANCE, 
                      LANDING_PAD_WIDTH, TERRAIN_ROUGHNESS, SAFE_LANDING_ANGLE)

class Terrain:
    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height
        self.ground_height = screen_height - 50
        self.landing_pad_width = LANDING_PAD_WIDTH
        
        # Randomize landing pad position between 20% and 80% of screen width
        pad_min = int(screen_width * 0.2)
        pad_max = int(screen_width * 0.8)
        self.landing_pad_x = random.randint(pad_min, pad_max)
        
        # Store points for line segments to enable proper collision detection
        self.segments = []
        
        # Generate initial terrain
        self.points = self._generate()
        self._generate_segments()

    def _generate(self) -> List[Tuple[int, int]]:
        """Generate terrain with random height variations and flat landing pad"""
        points = []
        segment_width = 20  # Distance between terrain points
        
        pad_left = self.landing_pad_x - self.landing_pad_width/2
        pad_right = self.landing_pad_x + self.landing_pad_width/2
        
        # Generate left side of terrain
        x = 0
        prev_height = self.ground_height
        while x < pad_left:
            new_height = prev_height + random.randint(-TERRAIN_ROUGHNESS, TERRAIN_ROUGHNESS)
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
            new_height = prev_height + random.randint(-TERRAIN_ROUGHNESS, TERRAIN_ROUGHNESS)
            new_height = min(max(new_height, self.ground_height - 50), self.ground_height + 50)
            points.append((int(x), int(new_height)))
            prev_height = new_height
            x += segment_width
            
        return points

    def _generate_segments(self):
        """Create line segments from points for collision detection"""
        self.segments = []
        for i in range(len(self.points) - 1):
            self.segments.append((self.points[i], self.points[i + 1]))

    def check_collision(self, x: float, y: float, lander) -> bool:
        """
        Check if lander collides with terrain with debug information
        """
        # Quick bounds check
        if y >= self.height:
            print("DEBUG: Collision - Lander below screen height")
            return True
            
        # Get lander vertices and legs
        vertices = lander.get_vertices()
        left_leg, right_leg = lander.get_leg_positions()
        
        """ print("\nDEBUG: Collision Check")
        print(f"Lander center: ({x}, {y})")
        print(f"Lander vertices: {vertices}")
        print(f"Left leg: {left_leg}")
        print(f"Right leg: {right_leg}") """
        
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
                    tolerance = LANDING_PAD_TOLERANCE if point_type == "leg" else 0
                    
                    # Debug point vs terrain height
                    if point_y > terrain_y + tolerance:
                        """ print(f"DEBUG: Collision detected!")
                        print(f"Point type: {point_type}")
                        print(f"Point position: ({point_x}, {point_y})")
                        print(f"Terrain height at point: {terrain_y}")
                        print(f"Terrain segment: ({x1}, {y1}) to ({x2}, {y2})") """
                        return True
                        
        return False

    def check_landing(self, x: float, y: float, velocity_y: float, lander) -> bool:
        """
        Check if lander has achieved safe landing on pad with debug information
        """
        pad_left = self.landing_pad_x - self.landing_pad_width/2
        pad_right = self.landing_pad_x + self.landing_pad_width/2
        
        # Get leg positions
        left_leg, right_leg = lander.get_leg_positions()
        left_foot = left_leg[1]
        right_foot = right_leg[1]
        """ 
        print("\nDEBUG: Landing Check")
        print(f"Pad bounds: {pad_left} to {pad_right}")
        print(f"Pad height: {self.ground_height}")
        print(f"Left foot position: {left_foot}")
        print(f"Right foot position: {right_foot}")
        print(f"Vertical velocity: {velocity_y}")
        print(f"Lander angle: {lander.angle}")
         """
        # Check each landing condition individually
        left_foot_in_bounds = pad_left <= left_foot[0] <= pad_right
        right_foot_in_bounds = pad_left <= right_foot[0] <= pad_right
        feet_in_bounds = left_foot_in_bounds and right_foot_in_bounds
        
        left_foot_height_ok = abs(left_foot[1] - self.ground_height) < LANDING_PAD_TOLERANCE
        right_foot_height_ok = abs(right_foot[1] - self.ground_height) < LANDING_PAD_TOLERANCE
        feet_at_height = left_foot_height_ok and right_foot_height_ok
        
        velocity_ok = abs(velocity_y) < SAFE_LANDING_VELOCITY
        angle_ok = abs(lander.angle) < SAFE_LANDING_ANGLE 
        
        """ 
        print("\nDEBUG: Landing Conditions")
        print(f"Left foot in bounds: {left_foot_in_bounds}")
        print(f"Right foot in bounds: {right_foot_in_bounds}")
        print(f"Left foot at correct height: {left_foot_height_ok}")
        print(f"Right foot at correct height: {right_foot_height_ok}")
        print(f"Safe vertical velocity: {velocity_ok}")
        print(f"Safe angle: {angle_ok}")
         """
        
        landing_successful = (
            feet_in_bounds and
            feet_at_height and
            velocity_ok and
            angle_ok
        )
        
        #print(f"Landing successful: {landing_successful}")
        return landing_successful