import math
import numpy as np
from typing import Dict, List
from game_init import get_constants
from utils import convert_degrees_to_custom_angle

class InputHandler:
    """Collects and normalizes inputs for the lunar lander"""
    
    def __init__(self):
        self.const = get_constants()
    
    def get_state(self, lander, terrain) -> np.ndarray:
        """
        Get the normalized state vector.
        Returns 3 normalized inputs:
        - distance_to_pad_x: x distance to pad normalized to [-1,1], negative=right of pad
        - velocity_y: y velocity normalized by safe landing velocity
        - angle: angle normalized to [-1,1] where:
          * 0.5 = facing right
          * -0.5 = facing left
          * ±1 = facing down (sign matches tilt direction)
        """
        # Calculate raw distances to landing pad and flip signs
        raw_distance_x = terrain.landing_pad_x - lander.x  # Flipped from (lander - pad) to (pad - lander)
        
        # Find the maximum possible distances
        max_distance_x = terrain.width
        
        # Normalize distances to [-1, 1] range and clamp values
        # If lander is right of pad, will be negative; if left of pad, will be positive
        distance_to_pad_x = np.clip(raw_distance_x / max_distance_x, -1.0, 1.0)
        
        # Normalize velocities and angles
        norm_vel_y = lander.velocity_y / self.const.SAFE_LANDING_VELOCITY
        
        # New angle normalization
        angle_degrees = math.degrees(lander.angle) % 360
        norm_angle = convert_degrees_to_custom_angle(angle_degrees)
        
        angular_vel = lander.angular_velocity
            
        return np.array([
            distance_to_pad_x,   # X distance to pad [-1,1], negative=right of pad
            norm_vel_y,          # Y velocity (relative to safe landing velocity) 
            norm_angle,          # Angle (relative to safe landing angle)
            angular_vel,         # Angular velocity (unchanged)
        ])
    
    def get_debug_info(self, lander, terrain) -> Dict[str, float]:
        """Get raw values for debug display"""
        return {
            'distance_x': terrain.landing_pad_x - lander.x,  # Flipped raw signed distance
            'velocity_y': lander.velocity_y,
            'angle': math.degrees(lander.angle),
            'angular_velocity': math.degrees(lander.angular_velocity)
        }