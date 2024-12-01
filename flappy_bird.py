import pygame
import neat
import random
import os
import colorsys
from background import Background

# Initialize Pygame
pygame.init()

# Constants - Doubled the dimensions
SCREEN_WIDTH = 1600  # Doubled from 800
SCREEN_HEIGHT = 1100  # Doubled from 550
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Colors
SKY_BLUE = (135, 206, 235)

# Game Parameters
FPS = 30
FLOOR_Y = SCREEN_HEIGHT  # Modified to be at bottom of screen
PIPE_GAP = 300  # Doubled from 150 to maintain proportions
PIPE_SPACING = 400  # Doubled from 200 to maintain proportions
PIPE_VELOCITY = 10  # Doubled from 5 to maintain same relative speed
BIRD_JUMP_VELOCITY = -16  # Doubled from -8 to maintain same relative jump height
GRAVITY = 6  # Doubled from 3 to maintain same relative fall speed

# Visual Parameters
PIPE_WIDTH = 140  # Doubled from 70
BIRD_SIZE = 60  # Doubled from 30
VISIBLE_PIPES = 5

# Load and scale pipe images
UP_PIPE_IMG = pygame.image.load('art/purple_pipe.png')
DOWN_PIPE_IMG = pygame.image.load('art/purple_pipe.png')

# Scale pipe images
up_pipe_aspect_ratio = UP_PIPE_IMG.get_height() / UP_PIPE_IMG.get_width()
down_pipe_aspect_ratio = DOWN_PIPE_IMG.get_height() / DOWN_PIPE_IMG.get_width()

UP_PIPE_IMG = pygame.transform.scale(UP_PIPE_IMG, 
                                   (PIPE_WIDTH, int(PIPE_WIDTH * up_pipe_aspect_ratio)))
DOWN_PIPE_IMG = pygame.transform.scale(DOWN_PIPE_IMG, 
                                     (PIPE_WIDTH, int(PIPE_WIDTH * down_pipe_aspect_ratio)))

class Bird:
    def __init__(self, x, y):
        # Load and scale original bird image
        original_image = pygame.image.load('art/bird.png')
        self.original_image = pygame.transform.scale(original_image, (BIRD_SIZE, BIRD_SIZE))
        
        # Create a copy of the scaled image for hue shifting
        self.bird_img = self.original_image.copy()
        
        # Get scaled dimensions
        self.width = BIRD_SIZE
        self.height = BIRD_SIZE
        
        # Randomly generate a hue value
        random_hue = random.random()
        
        # Apply random hue shift
        for x in range(self.width):
            for y in range(self.height):
                r, g, b, a = self.bird_img.get_at((x, y))
                
                if a == 0:  # Skip transparent pixels
                    continue
                    
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                new_h = (h + random_hue) % 1.0
                new_r, new_g, new_b = colorsys.hsv_to_rgb(new_h, s, v)
                
                self.bird_img.set_at((x, y), 
                    (int(new_r*255), int(new_g*255), int(new_b*255), a))
        
        # Initialize position and movement variables
        self.x = x
        self.y = y
        self.velocity = 0
        self.time = 0
        self.angle = 0
        self.max_upward_angle = 45
        self.max_downward_angle = -90
        self.rotation_speed = 8
        
    def move(self):
        self.time += 1
        displacement = self.velocity * self.time + 0.5 * GRAVITY * self.time ** 2
        self.y += displacement
        
        # Calculate target angle based on vertical velocity
        current_velocity = self.velocity + GRAVITY * self.time
        
        if current_velocity < 0:  # Rising
            target_angle = self.max_upward_angle
        else:  # Falling
            fall_ratio = min(current_velocity / 30.0, 1.0)  # Doubled from 15.0
            target_angle = self.max_downward_angle * fall_ratio
        
        # Smoothly rotate towards target angle
        if self.angle < target_angle:
            self.angle = min(target_angle, self.angle + self.rotation_speed)
        elif self.angle > target_angle:
            self.angle = max(target_angle, self.angle - self.rotation_speed)
            
    def jump(self):
        self.velocity = BIRD_JUMP_VELOCITY
        self.time = 0
        
    def draw(self, screen):
        rotated_bird = pygame.transform.rotate(self.bird_img, self.angle)
        new_rect = rotated_bird.get_rect(center=(self.x + self.width//2, 
                                                self.y + self.height//2))
        screen.blit(rotated_bird, new_rect.topleft)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randrange(300, 800)  # Doubled range from (150, 400)
        self.passed = False
        
        # Calculate positions for both pipes
        self.top_y = self.height - DOWN_PIPE_IMG.get_height()
        self.bottom_y = self.height + PIPE_GAP
        
        # Create pipe rectangles for collision detection
        self.top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.height)
        self.bottom_rect = pygame.Rect(self.x, self.bottom_y, PIPE_WIDTH, SCREEN_HEIGHT - self.bottom_y)
        
    def move(self):
        self.x -= PIPE_VELOCITY
        # Update collision rectangles
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x
        
    def draw(self, screen):
        screen.blit(DOWN_PIPE_IMG, (self.x, self.top_y))
        screen.blit(UP_PIPE_IMG, (self.x, self.bottom_y))

def check_collision(bird, pipe):
    bird_rect = pygame.Rect(bird.x, bird.y, BIRD_SIZE, BIRD_SIZE)
    
    if (bird_rect.colliderect(pipe.top_rect) or 
        bird_rect.colliderect(pipe.bottom_rect) or 
        bird.y > FLOOR_Y or bird.y < 0):
        return True
    return False

def draw_game(screen, background, pipes, birds, score):
    # Draw background first
    background.draw(screen)
    
    # Draw pipes and birds
    for pipe in pipes:
        pipe.draw(screen)
    for bird in birds:
        bird.draw(screen)
        
    # Draw score - doubled font size
    font = pygame.font.Font(None, 100)  # Doubled from 50
    score_text = font.render(str(score), True, (255, 255, 255))
    screen.blit(score_text, (SCREEN_WIDTH/2 - score_text.get_width()/2, 100))  # Doubled Y from 50
    
    pygame.display.update()

# Rest of the code remains the same as it uses the constants defined above
def eval_genomes(genomes, config):
    birds = []
    nets = []
    ge = []
    
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(300, 500))  # Doubled starting position from (150, 250)
        genome.fitness = 0
        ge.append(genome)
    
    background = Background(SCREEN_WIDTH, SCREEN_HEIGHT)
    pipes = [Pipe(1000 + i * PIPE_SPACING) for i in range(VISIBLE_PIPES)]  # Doubled starting X from 500
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
        
        background.move()
        
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
                pipe_ind = 1
        
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1
            
            output = nets[x].activate((
                bird.y,
                abs(bird.y - pipes[pipe_ind].height),
                abs(bird.y - pipes[pipe_ind].bottom_y)
            ))
            
            if output[0] > 0.5:
                bird.jump()
        
        for pipe in pipes:
            pipe.move()
            
            for x, bird in enumerate(birds):
                if check_collision(bird, pipe):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                elif not pipe.passed and bird.x > pipe.x + PIPE_WIDTH:
                    pipe.passed = True
                    score += 1
        
        while len(pipes) > 0 and pipes[0].x < -PIPE_WIDTH:
            pipes.pop(0)
            pipes.append(Pipe(pipes[-1].x + PIPE_SPACING))
        
        draw_game(SCREEN, background, pipes, birds, score)

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