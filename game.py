import pygame
import random
import sys

class FlappyBird:
    def __init__(self, width=400, height=400):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Flappy Bird')
        self.distance_traveled = 0
        
        # Initialize game state
        self.init_game_state()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        
        self.clock = pygame.time.Clock()
    
    def init_game_state(self):
        """Initialize/reset all game state variables"""
        # Bird properties
        self.bird_x = 50
        self.bird_y = self.height // 2
        self.bird_size = 20
        self.velocity = 0
        self.gravity = 0.5
        self.jump_strength = -8
        self.distance_traveled = 0 
        
        # Pipe properties
        self.pipe_width = 50
        self.pipe_gap = 150
        self.pipe_spacing = 200
        self.pipes = []
        self.spawn_pipe()
        
        # Game state
        self.score = 0
        self.game_over = False
        
    def spawn_pipe(self):
        gap_y = random.randint(100, self.height - 100 - self.pipe_gap)
        self.pipes.append({
            'x': self.width,
            'gap_y': gap_y
        })
    
    def get_game_state(self):
        """Return current game state for AI"""
        if not self.pipes:
            return {
                'distance_to_pipe': 0,
                'current_y': self.bird_y,
                'velocity': self.velocity,
                'pipe_y': 0,
                'y_error': 0  # Default error when no pipes
            }
            
        next_pipe = None
        # Find the first pipe that the bird hasn't passed yet
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                next_pipe = pipe
                break
        
        if not next_pipe:
            return {
                'distance_to_pipe': 0,
                'current_y': self.bird_y,
                'velocity': self.velocity,
                'pipe_y': 0,
                'y_error': 0  # Default error when no next pipe
            }
            
        # Calculate center of pipe gap
        pipe_center_y = next_pipe['gap_y'] + (self.pipe_gap / 2)
            
        # Calculate squared error between bird and pipe Y positions
        y_error = (self.bird_y - pipe_center_y) ** 2  # Now using center of gap
            
        return {
            'distance_to_pipe': next_pipe['x'] - self.bird_x,
            'current_y': self.bird_y,
            'velocity': self.velocity,
            'pipe_y': pipe_center_y,  # Now returning center instead of top edge
            'y_error': y_error
        }
    
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_over:
                        self.init_game_state()  # Restart the game
                    else:
                        self.velocity = self.jump_strength
    
    def update(self):
        if self.game_over:
            return
            
        # Update bird position
        self.velocity += self.gravity
        self.bird_y += self.velocity
        
        # Update distance traveled
        self.distance_traveled += 3  # Same speed as pipe movement
        
        # Update pipes
        for pipe in self.pipes:
            pipe['x'] -= 3
        
        # Remove off-screen pipes
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] > -self.pipe_width]
        
        # Spawn new pipes
        if len(self.pipes) == 0 or self.pipes[-1]['x'] < self.width - self.pipe_spacing:
            self.spawn_pipe()
        
        # Check collisions
        for pipe in self.pipes:
            # Check if bird hits the pipe
            if (self.bird_x + self.bird_size > pipe['x'] and 
                self.bird_x < pipe['x'] + self.pipe_width):
                if (self.bird_y < pipe['gap_y'] or 
                    self.bird_y + self.bird_size > pipe['gap_y'] + self.pipe_gap):
                    self.game_over = True
        
        # Check if bird hits the ground or ceiling
        if self.bird_y + self.bird_size > self.height or self.bird_y < 0:
            self.game_over = True
            
        # Update score
        for pipe in self.pipes:
            if (pipe['x'] + self.pipe_width < self.bird_x and 
                pipe.get('scored', False) == False):
                self.score += 1
                pipe['scored'] = True  # Mark this pipe as scored
    
    def draw(self):
        self.screen.fill(self.BLACK)
        
        # Draw bird
        pygame.draw.rect(self.screen, self.BLUE, 
                        (self.bird_x, self.bird_y, self.bird_size, self.bird_size))
        
        # Draw pipes
        for pipe in self.pipes:
            # Draw top pipe
            pygame.draw.rect(self.screen, self.GREEN,
                           (pipe['x'], 0, self.pipe_width, pipe['gap_y']))
            # Draw bottom pipe
            pygame.draw.rect(self.screen, self.GREEN,
                           (pipe['x'], pipe['gap_y'] + self.pipe_gap,
                            self.pipe_width, self.height))
        
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        
        if self.game_over:
            game_over_text = font.render('Game Over! Press SPACE to restart', True, self.WHITE)
            text_rect = game_over_text.get_rect(center=(self.width/2, self.height/2))
            self.screen.blit(game_over_text, text_rect)
        
        pygame.display.flip()
    
    def run(self):
        while True:
            self.clock.tick(60)
            self.handle_input()
            self.update()
            self.draw()

if __name__ == "__main__":
    game = FlappyBird()
    game.run()