import pygame
import neat
import random
import os

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 550
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Game Parameters
FPS = 30
FLOOR_Y = 500
PIPE_GAP = 150
PIPE_SPACING = 200
PIPE_VELOCITY = 5
BIRD_JUMP_VELOCITY = -8
GRAVITY = 3

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocity = 0
        self.time = 0
        self.angle = 0
        
    def move(self):
        self.time += 1
        # Calculate displacement using physics equation: d = vt + (1/2)at²
        displacement = self.velocity * self.time + 0.5 * GRAVITY * self.time ** 2
        self.y += displacement
        
        # Update angle based on movement
        if displacement < 0:
            self.angle = min(35, self.angle + 5)
        else:
            self.angle = max(-90, self.angle - 5)
            
    def jump(self):
        self.velocity = BIRD_JUMP_VELOCITY
        self.time = 0

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randrange(150, 400)
        self.top_y = self.height - 400  # Top pipe y position
        self.bottom_y = self.height + PIPE_GAP  # Bottom pipe y position
        
    def move(self):
        self.x -= PIPE_VELOCITY

def check_collision(bird, pipe):
    # Basic rectangle collision detection
    bird_rect = pygame.Rect(bird.x, bird.y, 30, 30)
    top_pipe_rect = pygame.Rect(pipe.x, 0, 50, pipe.height)
    bottom_pipe_rect = pygame.Rect(pipe.x, pipe.bottom_y, 50, SCREEN_HEIGHT)
    
    if (bird_rect.colliderect(top_pipe_rect) or 
        bird_rect.colliderect(bottom_pipe_rect) or 
        bird.y > FLOOR_Y or bird.y < 0):
        return True
    return False

def eval_genomes(genomes, config):
    birds = []
    nets = []
    ge = []
    
    # Create lists of birds, neural networks, and genomes
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(150, 250))
        genome.fitness = 0
        ge.append(genome)
    
    pipes = [Pipe(500)]
    score = 0
    clock = pygame.time.Clock()
    
    run = True
    while run and len(birds) > 0:
        clock.tick(FPS)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        
        # Get index of pipe to focus on
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + 50:
                pipe_ind = 1
        
        # Move birds and make decisions
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1
            
            # Neural network inputs
            output = nets[x].activate((
                bird.y,
                abs(bird.y - pipes[pipe_ind].height),
                abs(bird.y - pipes[pipe_ind].bottom_y)
            ))
            
            if output[0] > 0.5:
                bird.jump()
        
        # Move and manage pipes
        if len(pipes) > 0:
            for pipe in pipes:
                pipe.move()
                
                # Check for collisions
                for x, bird in enumerate(birds):
                    if check_collision(bird, pipe):
                        ge[x].fitness -= 1
                        birds.pop(x)
                        nets.pop(x)
                        ge.pop(x)
            
            # Add new pipe
            if pipes[-1].x < 500 - PIPE_SPACING:
                pipes.append(Pipe(500))
            
            # Remove off-screen pipes
            if pipes[0].x < -50:
                pipes.pop(0)
                score += 1
        
        # Draw everything
        SCREEN.fill((255, 255, 255))
        for bird in birds:
            pygame.draw.rect(SCREEN, (255, 0, 0), (bird.x, bird.y, 30, 30))
        for pipe in pipes:
            pygame.draw.rect(SCREEN, (0, 255, 0), (pipe.x, 0, 50, pipe.height))
            pygame.draw.rect(SCREEN, (0, 255, 0), (pipe.x, pipe.bottom_y, 50, SCREEN_HEIGHT))
        pygame.draw.line(SCREEN, (0, 0, 0), (0, FLOOR_Y), (SCREEN_WIDTH, FLOOR_Y))
        
        pygame.display.update()

def run_neat(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    
    winner = pop.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_neat(config_path)