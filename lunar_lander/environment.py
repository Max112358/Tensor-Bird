from typing import List, Tuple, Dict
import numpy as np
from lander import Lander
from terrain import Terrain
from renderer import Renderer

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
        
        # Track episode progress
        self.episode_rewards = None
        self.steps = 0
        
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
            self.landers.append(Lander(x, y))
            
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
                rewards.append(0)
                dones.append(True)
                info['landers'].append({
                    'active': False,
                    'reason': lander.terminate_reason,
                    'episode_reward': self.episode_rewards[i]
                })
                continue
            
            any_active = True  # We found an active lander
            
            # Calculate reward components
            distance_to_pad = abs(lander.x - self.terrain.landing_pad_x)
            height_diff = abs(lander.y - self.terrain.ground_height)
            velocity_penalty = abs(lander.velocity_x) + abs(lander.velocity_y)
            angle_penalty = abs(lander.angle)
            
            # Check termination conditions
            terminate = False
            reward = 0
            
            # Pass the lander instance to check_landing and check_collision
            if self.terrain.check_landing(lander.x, lander.y, lander.velocity_y, lander):
                reward = 100
                terminate = True
                lander.terminate('landed')
            elif self.terrain.check_collision(lander.x, lander.y, lander):
                reward = -100
                terminate = True
                lander.terminate('crashed')
            elif (lander.x < 0 or lander.x > self.width or lander.y < 0):
                reward = -100
                terminate = True
                lander.terminate('out_of_bounds')
            else:
                # Shaping rewards for active landers
                reward = (
                    -0.1 * (distance_to_pad / self.width)  # Penalize distance from pad
                    - 0.1 * (height_diff / self.height)    # Penalize height difference
                    - 0.2 * velocity_penalty               # Penalize excessive velocity  
                    - 0.2 * angle_penalty                  # Penalize tilting
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
        
        info['all_done'] = all_done
        info['steps'] = self.steps
        
        return states, rewards, [all_done] * len(self.landers), info
    
    def render(self):
        """Render the current state"""
        self.renderer.render(self.landers, self.terrain)
    
    def close(self):
        """Clean up resources"""
        self.renderer.close()