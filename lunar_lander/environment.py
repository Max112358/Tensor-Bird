from typing import List, Tuple, Dict
import numpy as np
from lander import Lander
from terrain import Terrain
from renderer import Renderer
from constants import *
import math


class MultiLanderEnv:
    def __init__(self, num_landers: int = 20, fast_mode: bool = False):
        # Previous initialization code remains the same
        self.width = 800
        self.height = 600
        self.num_landers = num_landers
        self.landers: List[Lander] = []
        self.terrain = None
        self.fast_mode = fast_mode
        if not fast_mode:
            self.renderer = Renderer(self.width, self.height)
        else:
            self.renderer = None
        self.episode_rewards = None
        self.steps = 0
        self.running = True
        self.reset()
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Take environment step for all landers
        
        Args:
            actions: List of actions, one per lander
            
        Returns:
            Tuple of (states, rewards, dones, info) where:
            - states: List of observation arrays for each lander
            - rewards: List of reward values for each lander
            - dones: List of done flags for each lander 
            - info: Dict containing additional information
        """
        self.steps += 1
        states = []
        rewards = []
        dones = []
        info = {'landers': []}
        
        # Process each lander
        for i, (lander, action) in enumerate(zip(self.landers, actions)):
            state = lander.step(action)
            states.append(state)
            
            # Calculate shaping rewards
            distance_to_pad = abs(lander.x - self.terrain.landing_pad_x)
            height_diff = abs(lander.y - self.terrain.ground_height)
            velocity_penalty = abs(lander.velocity_x) + abs(lander.velocity_y)
            angle_penalty = abs(lander.angle)
            
            # Track reward components
            reward_components = {
                'distance_reward': -50.0 * (distance_to_pad / self.width),
                'height_reward': -10.0 * (height_diff / self.height),
                'velocity_penalty': -1.0 * velocity_penalty / 100.0,
                'angle_penalty': -2.0 * angle_penalty,
                'fuel_penalty': -0.1 * (INITIAL_FUEL - lander.fuel) / INITIAL_FUEL,
                'terminal_reward': 0.0
            }
            
            # Calculate base reward from shaping
            reward = sum(reward_components.values())
            
            # For inactive landers, just return the current state
            if not lander.active:
                rewards.append(reward)
                dones.append(True)
                info['landers'].append({
                    'active': False,
                    'reason': lander.terminate_reason,
                    'episode_reward': self.episode_rewards[i]
                })
                continue
            
            # Check safety violations first
            terminate = False
            
            # Check for unsafe angle
            if abs(math.degrees(lander.angle)) > SAFE_LANDING_ANGLE:
                reward_components['terminal_reward'] = SAFETY_VIOLATION_PENALTY
                reward += SAFETY_VIOLATION_PENALTY
                terminate = True
                lander.terminate('unsafe_angle')
            
            # Check for unsafe velocity
            elif abs(lander.velocity_y) > SAFE_LANDING_VELOCITY:
                reward_components['terminal_reward'] = SAFETY_VIOLATION_PENALTY
                reward += SAFETY_VIOLATION_PENALTY
                terminate = True
                lander.terminate('unsafe_velocity')
            
            # If no safety violations, check other termination conditions
            elif self.terrain.check_landing(lander.x, lander.y, lander.velocity_y, lander):
                reward_components['terminal_reward'] = LANDING_REWARD
                reward += LANDING_REWARD
                terminate = True
                lander.terminate('landed')
            elif self.terrain.check_collision(lander.x, lander.y, lander):
                reward_components['terminal_reward'] = CRASH_PENALTY
                reward += CRASH_PENALTY
                terminate = True
                lander.terminate('crashed')
            elif (lander.x < 0 or lander.x > self.width or lander.y < 0):
                reward_components['terminal_reward'] = OUT_OF_BOUNDS_PENALTY
                reward += OUT_OF_BOUNDS_PENALTY
                terminate = True
                lander.terminate('out_of_bounds')
            elif (lander.fuel <= 0):
                reward_components['terminal_reward'] = OUT_OF_FUEL_PENALTY
                reward += OUT_OF_FUEL_PENALTY
                terminate = True
                lander.terminate('out_of_fuel')
            
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
        
        # Return same done state for all landers based on whether any are still active
        return states, rewards, [all_done] * len(self.landers), info
    
    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial states"""
        # Generate new terrain each reset
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
        
        spawn_y = self.height * 0.1  # Start near top of screen
        
        # Create new landers all at the same position
        for _ in range(self.num_landers):
            self.landers.append(Lander(spawn_x, spawn_y, self.terrain))
            
        # Return initial states
        return [lander.get_state() for lander in self.landers]
    
    
    
    def render(self):
        """Render the current state"""
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