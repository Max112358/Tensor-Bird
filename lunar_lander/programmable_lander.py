import pygame
import math
from lander import Lander
from terrain import Terrain
import game_init
from input_handler import InputHandler

class ProgrammableLunarLander:
    def __init__(self):
        # Initialize game constants
        self.const = game_init.init()
        
        # Initialize game components
        pygame.init()
        self.screen = pygame.display.set_mode((self.const.SCREEN_WIDTH, self.const.SCREEN_HEIGHT))
        pygame.display.set_caption("Programmable Lunar Lander")
        
        # Initialize input handler
        self.input_handler = InputHandler()
        
        # Game objects
        self.terrain = None
        self.lander = None
        self.clock = pygame.time.Clock()
        self.score = 0
        self.game_over = False
        
        # Display fonts
        self.score_font = pygame.font.Font(None, int(self.const.SCREEN_HEIGHT * 0.091))
        self.info_font = pygame.font.Font(None, int(self.const.SCREEN_HEIGHT * 0.05))
        self.debug_font = pygame.font.Font(None, int(self.const.SCREEN_HEIGHT * 0.033))
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        
        # Initialize game state
        self.reset()
        
    def compute_control(self):
        """
        Compute control actions based on current lander state.
        Override this method to implement custom control logic.
        Returns: action (0: no thrust, 1: left thrust, 2: main thrust, 3: right thrust)
        """
        # Get state from input handler
        state = self.input_handler.get_state(self.lander, self.terrain)
        
        # Example control logic using normalized inputs
        velocity_y = state[1]  # Normalized by safe landing velocity
        angle = state[2]       # Normalized by safe landing angle
        dist_x = state[4]      # Normalized [0,1]
        
        # Default control logic
        if abs(angle) > 0.5:  # If angle is over 50% of safe angle
            return 1 if angle > 0 else 3
        elif velocity_y > 0.8:  # If velocity is over 80% of safe velocity
            return 2
        elif abs(dist_x) > 0.3:  # If distance is over 30% of screen width
            return 1 if dist_x > 0 else 3
        return 0
        
    def step(self):
        """Execute one step of the game logic"""
        if not self.game_over:
            # Get control action
            action = self.compute_control()
            
            # Update lander physics
            self.lander.step(action)
            
            # Check various end conditions
            if self.terrain.check_landing(self.lander.x, self.lander.y, self.lander.velocity_y, self.lander):
                self.score += 100
                self.game_over = True
                self.lander.terminate('landed')
            
            elif self.lander.x < 0 or self.lander.x > self.const.SCREEN_WIDTH or self.lander.y < 0:
                self.score -= 50
                self.game_over = True
                self.lander.terminate('out_of_bounds')
            
            elif self.lander.fuel <= 0:
                self.score -= 50
                self.game_over = True
                self.lander.terminate('out_of_fuel')
            
            elif self.terrain.check_collision(self.lander.x, self.lander.y, self.lander):
                self.score -= 50
                self.game_over = True
                self.lander.terminate('crashed')
            
            else:
                self.score += 0.1
    
    def render(self):
        """Render the current game state"""
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw terrain
        pygame.draw.lines(self.screen, self.WHITE, False, self.terrain.points, 2)
        
        # Draw landing pad
        pad_left = self.terrain.landing_pad_x - self.terrain.landing_pad_width/2
        pygame.draw.line(
            self.screen, 
            self.RED,
            (pad_left, self.terrain.ground_height),
            (pad_left + self.terrain.landing_pad_width, self.terrain.ground_height),
            4
        )
        
        # Draw lander
        pygame.draw.polygon(self.screen, self.WHITE, self.lander.get_vertices())
        left_leg, right_leg = self.lander.get_leg_positions()
        pygame.draw.line(self.screen, self.WHITE, *left_leg)
        pygame.draw.line(self.screen, self.WHITE, *right_leg)
        
        # Get debug info from input handler
        debug_info = self.input_handler.get_debug_info(self.lander, self.terrain)
        
        # Draw debug info
        y_offset = 10
        line_height = 25
        metrics = [
            f"Dist to pad: {debug_info['distance_x']:.1f}x, {debug_info['distance_y']:.1f}y",
            f"Velocity: {debug_info['velocity_x']:.1f}x, {debug_info['velocity_y']:.1f}y",
            f"Angle: {debug_info['angle']:.1f}°",
            f"Angular vel: {debug_info['angular_velocity']:.1f}°/s",
            f"Fuel: {int(debug_info['fuel'])}",
            f"Score: {int(self.score)}"
        ]
        
        for i, metric in enumerate(metrics):
            text = self.debug_font.render(metric, True, self.WHITE)
            self.screen.blit(text, (10, y_offset + i * line_height))
        
        # Draw game over text if needed
        if self.game_over:
            game_over_text = self.info_font.render('Game Over! Press R to restart', True, self.WHITE)
            game_over_rect = game_over_text.get_rect(center=(self.const.SCREEN_WIDTH/2, self.const.SCREEN_HEIGHT/2))
            
            result_text = self.info_font.render(
                'Successful Landing!' if self.lander.terminate_reason == 'landed' 
                else f'Failed: {self.lander.terminate_reason}',
                True,
                self.GREEN if self.lander.terminate_reason == 'landed' else self.RED
            )
            result_rect = result_text.get_rect(center=(self.const.SCREEN_WIDTH/2, game_over_rect.top - 40))
            
            self.screen.blit(result_text, result_rect)
            self.screen.blit(game_over_text, game_over_rect)
        
        pygame.display.flip()
    
    def reset(self):
        """Reset the game state"""
        self.terrain = Terrain(self.const.SCREEN_WIDTH, self.const.SCREEN_HEIGHT)
        start_x = self.const.SCREEN_WIDTH * 0.5
        start_y = self.const.SCREEN_HEIGHT * 0.1
        self.lander = Lander(start_x, start_y, self.terrain)
        self.game_over = False
        self.score = 0
    
    def run(self):
        """Main game loop"""
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:
                        self.reset()
            
            # Update game state
            self.step()
            
            # Render
            self.render()
            
            # Control frame rate
            self.clock.tick(self.const.FPS)
    
    def close(self):
        """Clean up resources"""
        pygame.quit()