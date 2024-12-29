import numpy as np
import math
import random
from lander import Lander
from terrain import Terrain
from renderer import Renderer
from game_init import get_constants
from typing import List, Tuple, Dict, Any
from reward_tracker import RewardTracker

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
        
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
        # Initialize renderer if not in fast mode
        if not fast_mode:
            self.renderer = Renderer(self.width, self.height)
        else:
            self.renderer = None
            
        # Add reward trackers for each lander
        self.reward_trackers = [RewardTracker() for _ in range(num_landers)]
            
        self.episode_rewards = None
        self.steps = 0
        self.running = True
        self.reset()
    
    def print_lander_termination(self, lander_idx: int, lander: Lander, reason: str, reward: float):
        """Print lander info when it terminates"""
        dist_to_pad = abs(lander.x - self.terrain.landing_pad_x)
        dist_to_ground = abs(lander.y - self.terrain.ground_height)
        vel_mag = (lander.velocity_x**2 + lander.velocity_y**2)**0.5
        print(f"\nLander {lander_idx} terminated: {reason}")
        print(f"Total Reward: {reward:.2f}")
        print(f"Final State:")
        print(f"  Distance to pad: {dist_to_pad:.1f}")
        print(f"  Height: {dist_to_ground:.1f}")
        print(f"  Velocity: {vel_mag:.1f}")
        print(f"  Angle: {math.degrees(lander.angle):.1f}°")
        print(f"  Fuel: {lander.fuel:.1f}")

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict[str, Any]]:
        """Take environment step for all landers"""
        const = get_constants()
        self.steps += 1
        states = []
        rewards = []
        dones = []
        info = {'landers': []}
        
        # Process each lander
        for i, (lander, action) in enumerate(zip(self.landers, actions)):
            # Get state from lander
            state = lander.step(action)
            states.append(state)
            
            # Get reward tracker for this lander
            reward_tracker = self.reward_trackers[i]
            
            # For inactive landers, just return current state
            if not lander.active:
                rewards.append(reward_tracker.get_total_reward())
                dones.append(True)
                info['landers'].append({
                    'active': False,
                    'reason': lander.terminate_reason,
                    'episode_reward': self.episode_rewards[i]
                })
                continue
            
            # Calculate reward components for active lander
            reward_components = reward_tracker.calculate_survival_reward(lander, self.terrain)
            
            # Check termination conditions
            terminal_reward = 0.0
            
            if self.terrain.check_landing(lander.x, lander.y, lander.velocity_y, lander):
                lander.terminate('landed')
                terminal_reward = reward_tracker.calculate_terminal_reward(lander, self.terrain, 'landed')
            elif self.terrain.check_collision(lander.x, lander.y, lander):
                lander.terminate('crashed')
                terminal_reward = reward_tracker.calculate_terminal_reward(lander, self.terrain, 'crashed')
            elif (lander.x < 0 or lander.x > self.width or lander.y < 0):
                lander.terminate('out_of_bounds')
                terminal_reward = reward_tracker.calculate_terminal_reward(lander, self.terrain, 'out_of_bounds')
            elif (lander.fuel <= 0):
                lander.terminate('out_of_fuel')
                terminal_reward = reward_tracker.calculate_terminal_reward(lander, self.terrain, 'out_of_fuel')
            
            # Add terminal reward to components if terminated
            if lander.terminated:
                reward_components['terminal_reward'] = terminal_reward
            
            # Update reward tracker with new components
            reward_tracker.add_rewards(reward_components)
            
            # Get total reward for this step
            reward = reward_tracker.get_total_reward()
            
            # Update episode totals
            self.episode_rewards[i] += reward
            rewards.append(reward)
            dones.append(lander.terminated)
            
            # Add info for this lander
            info['landers'].append({
                'active': lander.active,
                'reason': lander.terminate_reason if lander.terminated else None,
                'episode_reward': self.episode_rewards[i],
                'reward_components': reward_components
            })
            
            # Print termination info if needed
            #if lander.terminated:
            #    reward_tracker.print_summary(i, lander.terminate_reason)
        
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
        
        # Reset episode rewards tracking and reward trackers
        self.episode_rewards = [0] * self.num_landers
        self.reward_trackers = [RewardTracker() for _ in range(self.num_landers)]
        
        # Calculate spawn position away from landing pad
        landing_pad_x = self.terrain.landing_pad_x
        
        # If landing pad is in left half, spawn on right side, and vice versa
        if landing_pad_x < self.width / 2:
            spawn_x = self.width * 0.8  # Spawn at 80% of screen width
        else:
            spawn_x = self.width * 0.2  # Spawn at 20% of screen width
        
        #spawn_x = self.width * 0.1  # ignore above and always spawn left
        spawn_y = self.height * 0.1  # Start near top of screen
        
        # Create new landers all at the same position
        for _ in range(self.num_landers):
            self.landers.append(Lander(spawn_x, spawn_y, self.terrain))
            
        # Return initial states
        return [lander.get_state() for lander in self.landers]
    
    def render(self) -> bool:
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