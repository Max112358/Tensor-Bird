import numpy as np
import math
from lander import Lander
from terrain import Terrain
from renderer import Renderer
from game_init import get_constants
from typing import List, Tuple, Dict, Any


class MultiLanderEnv:
    def __init__(self, num_landers: int = 20, fast_mode: bool = False):
        const = get_constants()
        
        # Initialize environment settings
        self.width = const.SCREEN_WIDTH
        self.height = const.SCREEN_HEIGHT
        self.num_landers = num_landers
        self.landers: List[Lander] = []
        self.terrain = None
        self.fast_mode = fast_mode
        
        # Initialize renderer if not in fast mode
        if not fast_mode:
            self.renderer = Renderer(self.width, self.height)
        else:
            self.renderer = None
            
        self.episode_rewards = None
        self.steps = 0
        self.running = True
        self.reset()
    
    def step(self, actions: 'List[int]') -> 'Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]':
        """Take environment step for all landers"""
        const = get_constants()
        self.steps += 1
        states = []
        rewards = []
        dones = []
        info = {'landers': []}
        
        # Calculate survival time bonus that scales with multiple factors
        max_survival_bonus = 20.0  # Base survival reward per frame
        
        # Process each lander
        for i, (lander, action) in enumerate(zip(self.landers, actions)):
            state = lander.step(action)
            states.append(state)
            
            # Calculate shaping rewards
            distance_to_pad = abs(lander.x - self.terrain.landing_pad_x)
            height_diff = abs(lander.y - self.terrain.ground_height)
            velocity_penalty = abs(lander.velocity_x) + abs(lander.velocity_y)
            angle_penalty = abs(lander.angle)
            
            # Calculate various ratios for survival bonus
            fuel_ratio = lander.fuel / const.INITIAL_FUEL  # 1.0 is full fuel
            distance_ratio = 1.0 - (distance_to_pad / self.width)  # 1.0 when at pad
            height_ratio = height_diff / self.height  # Higher when higher up
            
            # Calculate angle ratio based on safe landing angle
            current_angle_degrees = abs(math.degrees(lander.angle))
            angle_ratio = max(0, 1.0 - (current_angle_degrees / const.SAFE_LANDING_ANGLE))
            # This gives 1.0 when perfectly upright, and 0.0 when at or beyond safe angle
            
            # Calculate optimal vertical velocity (80% of safe landing speed)
            optimal_velocity_y = 0.8 * const.SAFE_LANDING_VELOCITY

            # Example values for context:
            # const.SAFE_LANDING_VELOCITY = 80.0
            # optimal_velocity_y = 64.0 (80% of safe landing velocity)
            # lander.velocity_y ranges from negative (going up) to positive (going down)

            if lander.velocity_y <= optimal_velocity_y:
                # This branch handles when we're moving up (negative) or moving down slower than optimal
                # Example: if velocity_y = 32.0, ratio = 32/64 = 0.5
                # Example: if velocity_y = -10.0, ratio = -10/64 = -0.15625
                velocity_ratio = lander.velocity_y / optimal_velocity_y
            else:
                # This branch handles when we're moving down faster than optimal but maybe still safe
                # Example: if velocity_y = 70.0:
                # ratio = 1.0 - ((70 - 64) / (80 - 64)) = 1.0 - (6/16) = 0.625
                velocity_ratio = max(0, 1.0 - ((lander.velocity_y - optimal_velocity_y) / 
                                            (const.SAFE_LANDING_VELOCITY - optimal_velocity_y)))

            # Calculate survival bonus that encourages staying alive while maintaining control
            survival_bonus = max_survival_bonus * (
                0.00 * fuel_ratio +      # Weight fuel conservation
                0.5 * distance_ratio +   # Weight proximity to target 
                0.15 * height_ratio +    # Weight maintaining height
                0.15 * angle_ratio +     # Weight staying upright 
                0.20 * velocity_ratio    # Weight maintaining optimal vertical speed
            )
            
            # Track reward components
            reward_components = {
                'distance_reward': -0.0 * (distance_to_pad / self.width),
                'height_reward': -0.0 * (height_diff / self.height),
                'velocity_penalty': -0.0 * velocity_penalty / 100.0,
                'angle_penalty': -0.0 * angle_penalty,
                'fuel_penalty': -0.0 * (const.INITIAL_FUEL - lander.fuel) / const.INITIAL_FUEL,
                'survival_bonus': survival_bonus,
                'terminal_reward': 0.0
            }
            
            # Calculate base reward from shaping
            reward = sum(reward_components.values())
            
            # For inactive landers, just return current state
            if not lander.active:
                rewards.append(reward)
                dones.append(True)
                info['landers'].append({
                    'active': False,
                    'reason': lander.terminate_reason,
                    'episode_reward': self.episode_rewards[i]
                })
                continue
            
            # Check termination conditions
            if self.terrain.check_landing(lander.x, lander.y, lander.velocity_y, lander):
                reward_components['terminal_reward'] = const.LANDING_REWARD
                reward += const.LANDING_REWARD
                terminate = True
                lander.terminate('landed')
            elif self.terrain.check_collision(lander.x, lander.y, lander):
                reward_components['terminal_reward'] = const.CRASH_PENALTY
                reward += const.CRASH_PENALTY
                terminate = True
                lander.terminate('crashed')
            elif (lander.x < 0 or lander.x > self.width or lander.y < 0):
                reward_components['terminal_reward'] = const.OUT_OF_BOUNDS_PENALTY
                reward += const.OUT_OF_BOUNDS_PENALTY
                terminate = True
                lander.terminate('out_of_bounds')
            elif (lander.fuel <= 0):
                reward_components['terminal_reward'] = const.OUT_OF_FUEL_PENALTY
                reward += const.OUT_OF_FUEL_PENALTY
                terminate = True
                lander.terminate('out_of_fuel')
            else:
                terminate = False
            
            # Update episode rewards and info
            self.episode_rewards[i] += reward
            rewards.append(reward)
            dones.append(terminate)
            info['landers'].append({
                'active': lander.active,
                'reason': lander.terminate_reason if terminate else None,
                'episode_reward': self.episode_rewards[i],
                'reward_components': reward_components
            })
        
        # Check if any landers are still active
        any_active = any(lander.active for lander in self.landers)
        all_done = not any_active
        
        # Update rendering if not in fast mode
        if not self.fast_mode:
            if not self.renderer.render(self.landers, self.terrain):
                self.running = False
                info['quit'] = True
                return states, rewards, [True] * len(self.landers), info
        
        # Set final info
        info['quit'] = False
        info['all_done'] = all_done
        info['steps'] = self.steps
        
        return states, rewards, [all_done] * len(self.landers), info
    
    
    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial states"""
        # Generate new terrain
        self.terrain = Terrain(self.width, self.height)
        
        # Clear existing landers
        self.landers = []
        self.steps = 0
        
        # Reset episode rewards tracking
        self.episode_rewards = [0] * self.num_landers
        
        # Calculate spawn position away from landing pad
        landing_pad_x = self.terrain.landing_pad_x
        
        # If landing pad is in left half, spawn on right side, and vice versa
        if landing_pad_x < self.width / 2:
            spawn_x = self.width * 0.8  # Spawn at 80% of screen width
        else:
            spawn_x = self.width * 0.2  # Spawn at 20% of screen width
        
        spawn_x = self.width * 0.2  # Spawn at 20% of screen width
        spawn_y = self.height * 0.1  # Start near top of screen
        
        # Create new landers all at the same position
        for _ in range(self.num_landers):
            self.landers.append(Lander(spawn_x, spawn_y, self.terrain))
            
        # Return initial states
        return [lander.get_state() for lander in self.landers]
    
    def render(self):
        """Render current state"""
        if self.fast_mode:
            return True
            
        if not self.renderer.render(self.landers, self.terrain):
            self.running = False
            return False
        return True
    
    def close(self):
        """Clean up resources"""
        if not self.fast_mode and self.renderer:
            self.renderer.close()
        
    def is_running(self) -> bool:
        """Check if environment is still running"""
        return self.running

    def get_episode_rewards(self) -> List[float]:
        """Get current episode rewards for all landers"""
        return self.episode_rewards.copy()

    def get_active_landers(self) -> int:
        """Get count of currently active landers"""
        return sum(1 for lander in self.landers if lander.active)

    def get_completed_landers(self) -> Dict[str, int]:
        """Get counts of landers by termination reason"""
        reasons = {}
        for lander in self.landers:
            if not lander.active:
                reasons[lander.terminate_reason] = reasons.get(lander.terminate_reason, 0) + 1
        return reasons