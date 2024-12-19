import math
import numpy as np
from typing import Dict, List
from game_init import get_constants

class InputHandler:
    """Collects and normalizes inputs for the lunar lander"""
    
    def __init__(self):
        self.const = get_constants()
    
    def get_state(self, lander, terrain) -> np.ndarray:
        """
        Get the normalized state vector.
        Returns 5 normalized inputs:
        - velocity_y: y velocity normalized by safe landing velocity
        - angle: angle normalized by safe landing angle
        - angular_velocity: raw angular velocity
        - distance_to_pad_x: x distance to pad normalized to [-1,1], negative=right of pad
        - distance_to_pad_y: y distance to pad normalized to [-1,1], negative=above pad
        """
        # Calculate raw distances to landing pad and flip signs
        raw_distance_x = terrain.landing_pad_x - lander.x  # Flipped from (lander - pad) to (pad - lander)
        raw_distance_y = terrain.ground_height - lander.y  # Flipped from (lander - ground) to (ground - lander)
        
        # Find the maximum possible distances
        max_distance_x = terrain.width
        max_distance_y = terrain.height
        
        # Normalize distances to [-1, 1] range and clamp values
        # If lander is right of pad, will be negative; if left of pad, will be positive
        distance_to_pad_x = np.clip(raw_distance_x / max_distance_x, -1.0, 1.0)
        # If lander is above pad, will be negative; if below pad, will be positive 
        distance_to_pad_y = np.clip(raw_distance_y / max_distance_y, -1.0, 1.0)
        
        # Normalize velocities and angles
        norm_vel_y = lander.velocity_y / self.const.SAFE_LANDING_VELOCITY
        angle_degrees = math.degrees(lander.angle)
        norm_angle = angle_degrees / self.const.SAFE_LANDING_ANGLE
        angular_vel = lander.angular_velocity
        
        return np.array([
            norm_vel_y,          # Y velocity (relative to safe landing velocity) 
            norm_angle,          # Angle (relative to safe landing angle)
            angular_vel,         # Angular velocity (unchanged)
            distance_to_pad_x,   # X distance to pad [-1,1], negative=right of pad
            distance_to_pad_y    # Y distance to pad [-1,1], negative=above pad
        ])
    
    def get_debug_info(self, lander, terrain) -> Dict[str, float]:
        """Get raw values for debug display"""
        return {
            'distance_x': terrain.landing_pad_x - lander.x,  # Flipped raw signed distance
            'distance_y': terrain.ground_height - lander.y,  # Flipped raw signed distance
            'velocity_y': lander.velocity_y,
            'angle': math.degrees(lander.angle),
            'angular_velocity': math.degrees(lander.angular_velocity)
        }