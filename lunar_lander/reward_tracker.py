from game_init import get_constants
import math
from typing import Dict, Any

class RewardTracker:
    """Tracks reward components for a single lander throughout its episode"""
    def __init__(self):
        # Get constants for reward calculations
        self.const = get_constants()
        
        # Initialize reward statistics
        self.stats = {
            'terminal_reward': 0.0,
            # Detailed survival components that accumulate over time
            'survival': {
                'fuel': 0.0,
                'distance': 0.0,
                'height': 0.0,
                'angle': 0.0,
                'velocity': 0.0
            },
            # Current frame ratios (not accumulated)
            'ratios': {
                'fuel': 0.0,
                'distance': 0.0,
                'height': 0.0,
                'angle': 0.0,
                'velocity': 0.0
            },
            'total_reward': 0.0
        }
        
        # Track accumulated survival reward
        self.accumulated_survival_reward = 0.0

    def calculate_survival_reward(self, lander, terrain) -> Dict[str, Any]:
        """
        Calculate reward components based on current lander state
        Returns dict containing all reward components and ratios for this frame
        """
        # Calculate base metrics
        distance_to_pad = abs(lander.x - terrain.landing_pad_x)
        height_diff = abs(lander.y - terrain.ground_height)
        current_angle_degrees = abs(math.degrees(lander.angle))
        
        # Calculate raw ratios for this frame
        fuel_ratio = lander.fuel / self.const.INITIAL_FUEL
        distance_ratio = 1.0 - (distance_to_pad / terrain.width)
        height_ratio = height_diff / terrain.height
        
        # New angle ratio calculation to penalize anything outside the safe landing angle
        if current_angle_degrees <= self.const.SAFE_LANDING_ANGLE:
            angle_ratio = 1.0  # Full points if within safe bounds
        else:
            angle_ratio = -1.0  # Penalize if outside safe bounds
            
              
        # Calculate velocity ratio with optimal velocity consideration
        optimal_velocity_y = 0.8 * self.const.SAFE_LANDING_VELOCITY
        if lander.velocity_y <= optimal_velocity_y:
            velocity_ratio = lander.velocity_y / optimal_velocity_y
        else:
            velocity_ratio = max(0, 1.0 - ((lander.velocity_y - optimal_velocity_y) / 
                                       (self.const.SAFE_LANDING_VELOCITY - optimal_velocity_y)))
        
        survival_reward_base = 20.0
        
        # Calculate individual survival components for this frame
        survival_components = {
            'fuel': survival_reward_base * 0.00 * fuel_ratio,
            'distance': survival_reward_base * 0.3 * distance_ratio,
            'height': survival_reward_base * 0.15 * height_ratio,
            'angle': survival_reward_base * 0.35 * angle_ratio,
            'velocity': survival_reward_base * 0.20 * velocity_ratio
        }
        
        # Calculate frame survival bonus
        frame_survival_bonus = sum(survival_components.values())
        
        # Store current frame ratios
        ratios = {
            'fuel': fuel_ratio,
            'distance': distance_ratio,
            'height': height_ratio,
            'angle': angle_ratio,
            'velocity': velocity_ratio
        }
        
        # Add frame survival bonus to accumulated total
        self.accumulated_survival_reward += frame_survival_bonus
        
        # Package reward components
        reward_components = {
            'survival_components': survival_components,
            'ratios': ratios,
            'frame_survival_bonus': frame_survival_bonus,
            'accumulated_survival_reward': self.accumulated_survival_reward,
            'terminal_reward': 0.0
        }
        
        return reward_components

    def calculate_terminal_reward(self, lander, terrain, reason: str) -> float:
        """Calculate terminal reward based on termination reason"""
        if reason == 'landed':
            return self.const.LANDING_REWARD
        elif reason == 'crashed':
            return self.const.CRASH_PENALTY
        elif reason == 'out_of_bounds':
            return self.const.OUT_OF_BOUNDS_PENALTY
        elif reason == 'out_of_fuel':
            return self.const.OUT_OF_FUEL_PENALTY
        return 0.0
        
    def add_rewards(self, reward_components: dict):
        """Add new reward components to tracking statistics"""
        # Update survival components (accumulate them)
        if 'survival_components' in reward_components:
            components = reward_components['survival_components']
            for key, value in components.items():
                self.stats['survival'][key] += value
            
        # Update latest ratios (don't accumulate)
        if 'ratios' in reward_components:
            self.stats['ratios'] = reward_components['ratios']
            
        # Update terminal reward if present
        if 'terminal_reward' in reward_components:
            self.stats['terminal_reward'] = reward_components['terminal_reward']
            
        # Update total reward (accumulated survival + terminal)
        self.stats['total_reward'] = self.accumulated_survival_reward + self.stats['terminal_reward']
        
    def get_total_reward(self) -> float:
        """Get the current total reward (accumulated survival + terminal)"""
        return self.stats['total_reward']
        
    def print_summary(self, lander_id: int, reason: str):
        """Print detailed breakdown of rewards and components"""
        print(f"\nLander {lander_id} terminated: {reason}")
        
        print(f"\n  Accumulated survival components:")
        print(f"    Fuel:         {self.stats['survival']['fuel']:10.1f}")
        print(f"    Distance:     {self.stats['survival']['distance']:10.1f}")
        print(f"    Height:       {self.stats['survival']['height']:10.1f}")
        print(f"    Angle:        {self.stats['survival']['angle']:10.1f}")
        print(f"    Velocity:     {self.stats['survival']['velocity']:10.1f}")
        
        print(f"\n  Final frame ratios:")
        print(f"    Fuel:         {self.stats['ratios']['fuel']:10.3f}")
        print(f"    Distance:     {self.stats['ratios']['distance']:10.3f}")
        print(f"    Height:       {self.stats['ratios']['height']:10.3f}")
        print(f"    Angle:        {self.stats['ratios']['angle']:10.3f}")
        print(f"    Velocity:     {self.stats['ratios']['velocity']:10.3f}")
        
        print(f"\n  Accumulated survival reward: {self.accumulated_survival_reward:10.1f}")
        
        if self.stats['terminal_reward'] != 0:
            print(f"  Terminal reward:           {self.stats['terminal_reward']:10.1f}")
        
        print(f"\nTotal Reward:                  {self.stats['total_reward']:10.1f}")