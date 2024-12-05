from typing import List, Tuple, Dict
import numpy as np
from lander import Lander
from terrain import Terrain
from renderer import Renderer
from constants import *

class MultiLanderEnv:
    def __init__(self, num_landers: int = 20):
        # Screen dimensions 
        self.width = 800
        self.height = 600
        
        # Store number of landers for reset
        self.num_landers = num_landers
        self.landers: List[Lander] = []
        
        # Initialize terrain and renderer
        self.terrain = None  # Will be created in reset()
        self.renderer = Renderer(self.width, self.height)
        
        # Track episode progress and state
        self.episode_rewards = None
        self.steps = 0
        self.running = True
        
        # Initialize environment
        self.reset()
    
    def reset(self) -> List[np.ndarray]:
        """Reset environment and return initial states"""
        # Generate new terrain each reset
        self.terrain = Terrain(self.width, self.height)
        
        # Clear existing landers
        self.landers = []
        self.steps = 0
        
        # Reset episode rewards tracking
        self.episode_rewards = [0] * self.num_landers
        
        # Create new landers with random starting positions along top
        for _ in range(self.num_landers):
            x = np.random.uniform(self.width * 0.2, self.width * 0.8)
            y = self.height * 0.1  # Start near top of screen
            self.landers.append(Lander(x, y, self.terrain))
            
        # Return initial states
        return [lander.get_state() for lander in self.landers]
    
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Take environment step for all landers"""
        self.steps += 1
        states = []
        rewards = []
        dones = []
        info = {'landers': []}
        
        any_active = False  # Track if any landers are still active
        
        # Process each active lander
        for i, (lander, action) in enumerate(zip(self.landers, actions)):
            state = lander.step(action)
            states.append(state)
            
            # For inactive landers, maintain their final state
            if not lander.active:
                rewards.append(0)  # Keep this as 0 since we don't want to double-count penalties
                dones.append(True)
                info['landers'].append({
                    'active': False,
                    'reason': lander.terminate_reason,
                    'episode_reward': self.episode_rewards[i]  # This will contain the total accumulated reward
                })
                continue
            
            any_active = True  # We found an active lander
            
            # Calculate reward components with balanced scales
            distance_to_pad = abs(lander.x - self.terrain.landing_pad_x)
            height_diff = abs(lander.y - self.terrain.ground_height)
            velocity_penalty = abs(lander.velocity_x) + abs(lander.velocity_y)
            angle_penalty = abs(lander.angle)
            
            # Check termination conditions
            terminate = False
            reward = 0
            
            # Terminal rewards
            if self.terrain.check_landing(lander.x, lander.y, lander.velocity_y, lander):
                reward += LANDING_REWARD  # Significant positive reward for landing
                terminate = True
                lander.terminate('landed')
            elif self.terrain.check_collision(lander.x, lander.y, lander):
                reward += CRASH_PENALTY  # Significant negative reward for crashing
                terminate = True
                lander.terminate('crashed')
            elif (lander.x < 0 or lander.x > self.width or lander.y < 0):
                reward += OUT_OF_BOUNDS_PENALTY  # Equal penalty for going out of bounds
                terminate = True
                lander.terminate('out_of_bounds')
            elif (lander.fuel <= 0):
                reward += OUT_OF_BOUNDS_PENALTY  # Equal penalty for running out of fuel
                terminate = True
                lander.terminate('out_of_fuel')
            elif self.steps >= MAX_STEPS_PER_EPISODE:
                reward += OUT_OF_BOUNDS_PENALTY  # Penalty for timeout
                terminate = True
                lander.terminate('timeout')
            else:
                # Shaping rewards with balanced scales
                reward += (
                    -10.0 * (distance_to_pad / self.width)     # Distance from pad
                    - 10.0 * (height_diff / self.height)       # Height difference
                    - 1.0 * velocity_penalty / 100.0          # Excessive velocity
                    - 2.0 * angle_penalty                     # Tilting
                    - 0.1 * (INITIAL_FUEL - lander.fuel) / INITIAL_FUEL  # Fuel efficiency
                )
            
            # Update episode rewards
            self.episode_rewards[i] += reward
            rewards.append(reward)
            dones.append(terminate)
            info['landers'].append({
                'active': lander.active,
                'reason': lander.terminate_reason if terminate else None,
                'episode_reward': self.episode_rewards[i]
            })
        
        # Episode is only done when NO landers are active
        all_done = not any_active
        
        # Update renderer and check if window should close
        if not self.renderer.render(self.landers, self.terrain):
            self.running = False
            info['quit'] = True
            return states, rewards, [True] * len(self.landers), info
            
        info['quit'] = False
        info['all_done'] = all_done
        info['steps'] = self.steps
        
        return states, rewards, [all_done] * len(self.landers), info
    
    def render(self):
        """Render the current state"""
        if not self.renderer.render(self.landers, self.terrain):
            self.running = False
            return False
        return True
    
    def close(self):
        """Clean up resources"""
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