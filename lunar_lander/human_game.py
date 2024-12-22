import pygame
import sys
import math
from lander import Lander
from terrain import Terrain
from renderer import Renderer
import game_init

class HumanLunarLander:
    def __init__(self):
        # Initialize game constants first
        self.const = game_init.init()
        
        # Initialize game components
        pygame.init()
        self.screen = pygame.display.set_mode((self.const.SCREEN_WIDTH, self.const.SCREEN_HEIGHT))
        pygame.display.set_caption("Lunar Lander - WASD Controls")
        
        self.terrain = Terrain(self.const.SCREEN_WIDTH, self.const.SCREEN_HEIGHT)
        self.lander = None
        self.clock = pygame.time.Clock()
        self.score = 0
        self.game_over = False
        
        # Initialize fonts at different sizes
        self.score_font = pygame.font.Font(None, int(self.const.SCREEN_HEIGHT * 0.091))
        self.info_font = pygame.font.Font(None, int(self.const.SCREEN_HEIGHT * 0.05))
        self.debug_font = pygame.font.Font(None, int(self.const.SCREEN_HEIGHT * 0.033))
        self.game_over_font = pygame.font.Font(None, int(self.const.SCREEN_HEIGHT * 0.067))
        
        # Pre-render game over text and controls help
        self.game_over_text = self.game_over_font.render('Game Over! Press R to restart', True, (255, 255, 255))
        self.game_over_rect = self.game_over_text.get_rect(center=(self.const.SCREEN_WIDTH/2, self.const.SCREEN_HEIGHT/2))
        
        # Add control instructions
        self.controls_text = self.info_font.render('Controls: W-Main Engine, A-Left Thruster, D-Right Thruster', True, (255, 255, 255))
        self.controls_rect = self.controls_text.get_rect(bottomleft=(10, self.const.SCREEN_HEIGHT - 10))
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)
        
        # Initialize game state
        self.reset()
        
    def handle_input(self):
        """
        Handle keyboard input using WASD controls with combinations allowed.
        Main engine (W) can be combined with left/right thrusters (A/D).
        Returns the action number for the physics system.
        """
        keys = pygame.key.get_pressed()
        
        # Track active thrusters
        main_engine = keys[pygame.K_w]
        left_thruster = keys[pygame.K_a] 
        right_thruster = keys[pygame.K_d]
        
        # Handle combinations
        if main_engine and left_thruster:
            return 1  # Main + left thruster
        elif main_engine and right_thruster:
            return 3  # Main + right thruster
        elif main_engine:
            return 2  # Just main engine
        elif left_thruster:
            return 1  # Just left thruster
        elif right_thruster:
            return 3  # Just right thruster
                
        return 0  # No thrusters
        
    def render(self):
        """Render everything in the right order"""
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
        
        # Draw guidance line to pad center (training target)
        pad_center = (self.terrain.landing_pad_x, self.terrain.ground_height)
        # pygame.draw.line(self.screen, self.BLUE, (self.lander.x, self.lander.y), pad_center, 1)
        
        # Draw velocity vector (scaled)
        vel_scale = 2.0
        # pygame.draw.line(
        #     self.screen,
        #     self.YELLOW,
        #     (self.lander.x, self.lander.y),
        #     (self.lander.x + self.lander.velocity_x * vel_scale, 
        #      self.lander.y + self.lander.velocity_y * vel_scale),
        #     2
        # )
        
        # Draw angle indicator
        angle_radius = 30
        angle_end = (
            self.lander.x + math.cos(self.lander.angle) * angle_radius,
            self.lander.y + math.sin(self.lander.angle) * angle_radius
        )
        # pygame.draw.line(self.screen, self.GREEN, (self.lander.x, self.lander.y), angle_end, 2)
        
        # Draw UI elements
        # Score
        score_text = self.score_font.render(str(int(self.score)), True, self.WHITE)
        score_rect = score_text.get_rect(center=(self.const.SCREEN_WIDTH/2, self.const.SCREEN_HEIGHT * 0.091))
        self.screen.blit(score_text, score_rect)
        
        # Training metrics (left side)
        y_offset = 10
        line_height = 25
        
        # Distance to pad
        distance_x = abs(self.lander.x - self.terrain.landing_pad_x)
        distance_y = abs(self.lander.y - self.terrain.ground_height)
        metrics = [
            f"Dist to pad: {distance_x:.1f}x, {distance_y:.1f}y",
            f"Velocity: {abs(self.lander.velocity_x):.1f}x, {abs(self.lander.velocity_y):.1f}y",
            f"Angle: {math.degrees(self.lander.angle):.1f}°",
            f"Angular vel: {math.degrees(self.lander.angular_velocity):.1f}°/s",
            f"Fuel: {int(self.lander.fuel)}"
        ]
        
        # Add safety thresholds
        safety_metrics = [
            f"Safe vel: <{self.const.SAFE_LANDING_VELOCITY:.1f}",
            f"Safe angle: <{self.const.SAFE_LANDING_ANGLE:.1f}°",
        ]
        
        # Render metrics
        for i, metric in enumerate(metrics):
            color = self.WHITE
            text = self.debug_font.render(metric, True, color)
            self.screen.blit(text, (10, y_offset + i * line_height))
            
        # Render safety thresholds (right side)
        for i, metric in enumerate(safety_metrics):
            color = self.WHITE
            text = self.debug_font.render(metric, True, color)
            self.screen.blit(text, (self.const.SCREEN_WIDTH - 150, y_offset + i * line_height))
        
        # Controls at bottom
        self.screen.blit(self.controls_text, self.controls_rect)
        
        # Game over text if needed
        if self.game_over:
            if self.lander.terminate_reason == 'landed':
                result_text = self.info_font.render('Successful Landing!', True, self.GREEN)
            else:
                result_text = self.info_font.render(
                    f'Failed: {self.lander.terminate_reason}', True, self.RED
                )
            result_rect = result_text.get_rect(center=(self.const.SCREEN_WIDTH/2, self.game_over_rect.top - 40))
            
            self.screen.blit(result_text, result_rect)
            self.screen.blit(self.game_over_text, self.game_over_rect)
        
        # Update display
        pygame.display.flip()
            
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
                        
            if not self.game_over:
                # Get player input
                action = self.handle_input()
                
                # Update lander physics
                self.lander.step(action)
                
                # First check for landing
                if self.terrain.check_landing(self.lander.x, self.lander.y, self.lander.velocity_y, self.lander):
                    self.score += 100
                    self.game_over = True
                    self.lander.terminate('landed')
                
                # Then check for out of bounds
                elif self.lander.x < 0 or self.lander.x > self.const.SCREEN_WIDTH or self.lander.y < 0:
                    self.score -= 50
                    self.game_over = True
                    self.lander.terminate('out_of_bounds')
                
                # Then check for fuel
                elif self.lander.fuel <= 0:
                    self.score -= 50
                    self.game_over = True
                    self.lander.terminate('out_of_fuel')
                
                # Finally check for collisions, but only if we haven't landed
                elif self.terrain.check_collision(self.lander.x, self.lander.y, self.lander):
                    self.score -= 50
                    self.game_over = True
                    self.lander.terminate('crashed')
                
                else:
                    # Small reward for staying alive
                    self.score += 0.1
            
            # Render game state
            self.render()
            
            # Control frame rate
            self.clock.tick(self.const.FPS)
            
    def close(self):
        """Clean up resources"""
        pygame.quit()
        
    def reset(self):
        """Reset the game state with new terrain"""
        # Generate new terrain
        self.terrain = Terrain(self.const.SCREEN_WIDTH, self.const.SCREEN_HEIGHT)
        
        # Calculate spawn position away from landing pad
        landing_pad_x = self.terrain.landing_pad_x
        
        # If landing pad is in left half, spawn on right side, and vice versa
        if landing_pad_x < self.const.SCREEN_WIDTH / 2:
            spawn_x = self.const.SCREEN_WIDTH * 0.8  # Spawn at 80% of screen width
        else:
            spawn_x = self.const.SCREEN_WIDTH * 0.2  # Spawn at 20% of screen width
        
        spawn_y = self.const.SCREEN_HEIGHT * 0.1  # Start near top of screen
        
        # Create new lander
        self.lander = Lander(spawn_x, spawn_y, self.terrain)
        
        # Reset game state
        self.game_over = False
        self.score = 0

if __name__ == "__main__":
    game = HumanLunarLander()
    try:
        game.run()
    finally:
        game.close()