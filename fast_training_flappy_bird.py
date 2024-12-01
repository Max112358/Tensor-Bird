from game import FlappyBird

class FastTrainingFlappyBird(FlappyBird):
    def __init__(self, width=400, height=400):
        # Initialize without pygame display
        self.width = width
        self.height = height
        self.distance_traveled = 0
        self.init_game_state()
        
        # Colors (kept for compatibility)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
    
    def draw(self):
        # Override draw method to do nothing
        pass
        
    def update(self):
        """Override update to remove all pygame and timing dependencies"""
        if self.game_over:
            return
            
        # Update bird position
        self.velocity += self.gravity
        self.bird_y += self.velocity
        
        # Update distance traveled - increased for faster training
        self.distance_traveled += 10  # Move faster than visual version
        
        # Update pipes - increased for faster training
        for pipe in self.pipes:
            pipe['x'] -= 10  # Move faster than visual version
        
        # Remove off-screen pipes
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] > -self.pipe_width]
        
        # Spawn new pipes
        if len(self.pipes) == 0 or self.pipes[-1]['x'] < self.width - self.pipe_spacing:
            self.spawn_pipe()
        
        # Check collisions
        for pipe in self.pipes:
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
                pipe['scored'] = True