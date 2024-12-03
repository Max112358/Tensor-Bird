# test_main.py
import pygame
import neat
import os
import sys
from constants import *
from bird import Bird
from pipe import Pipe 
from background import Background
from game_utils import check_collision, draw_game
from death_marker import DeathMarker

def create_extreme_pipe(x, is_high):
    pipe = Pipe(x)
    if is_high:
        # Place gap as high as possible while maintaining minimum margins
        pipe.gap_y = PIPE_TOP_MARGIN
    else:
        # Place gap as low as possible while maintaining minimum margins
        pipe.gap_y = SCREEN_HEIGHT - PIPE_GAP - PIPE_BOTTOM_MARGIN
    
    # Recalculate pipe positions based on new gap_y
    pipe.top_y = pipe.gap_y - pipe.DOWN_PIPE_IMG.get_height()
    pipe.bottom_y = pipe.gap_y + PIPE_GAP
    pipe.height = pipe.gap_y
    
    # Update collision rectangles
    pipe.top_rect = pygame.Rect(pipe.x, 0, PIPE_WIDTH, pipe.gap_y)
    pipe.bottom_rect = pygame.Rect(pipe.x, pipe.bottom_y, PIPE_WIDTH, 
                                 SCREEN_HEIGHT - pipe.bottom_y)
    return pipe

def eval_genomes(genomes, config):
    try:
        # Initialize lists to track active birds and networks
        birds = []
        nets = []
        ge = []
        death_markers = []
        best_genome = None
        best_fitness = -float('inf')
        
        # Create neural networks for each genome
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            birds.append(Bird(BIRD_START_X, BIRD_START_Y))
            genome.fitness = 0
            ge.append(genome)
        
        # Initialize game objects
        background = Background(SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Create alternating extreme height pipes
        pipes = []
        for i in range(VISIBLE_PIPES):
            is_high = i % 2 == 0  # Alternate between high and low
            x = FIRST_PIPE_X + i * PIPE_SPACING
            pipes.append(create_extreme_pipe(x, is_high))
        
        score = 0
        clock = pygame.time.Clock()
        
        run = True
        while run and len(birds) > 0:
            clock.tick(FPS)
            
            # Handle quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    current_best = max(ge, key=lambda x: x.fitness)
                    print('\nBest genome at exit:\n{!s}'.format(current_best))
                    print(f'Final fitness: {current_best.fitness}')
                    pygame.quit()
                    sys.exit()
            
            # Move background for parallax effect
            background.move()
            
            # Move and clean up death markers
            for marker in death_markers[:]:
                marker.move()
                if marker.is_offscreen():
                    death_markers.remove(marker)
            
            # Determine which pipes to focus on (current and next)
            pipe_ind = 0
            next_pipe_ind = 1
            if len(birds) > 0:
                if len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
                    pipe_ind = 1
                    next_pipe_ind = 2
            
            # Update all birds
            for x, bird in enumerate(birds):
                bird.move()
                ge[x].fitness += 0.1
                
                # Get the next pipe if available
                next_pipe = pipes[next_pipe_ind] if next_pipe_ind < len(pipes) else pipes[pipe_ind]
                
                # Neural network inputs
                output = nets[x].activate((
                    bird.y,
                    bird.velocity,
                    abs(bird.y - pipes[pipe_ind].height),
                    abs(bird.y - pipes[pipe_ind].bottom_y),
                    pipes[pipe_ind].x - bird.x,
                    abs(bird.y - next_pipe.height),
                    abs(bird.y - next_pipe.bottom_y),
                    next_pipe.x - bird.x
                ))
                
                if output[0] > 0.5:
                    bird.jump()
                
                if ge[x].fitness > best_fitness:
                    best_fitness = ge[x].fitness
                    best_genome = ge[x]
            
            # Update and check all pipes
            for pipe in pipes:
                pipe.move()
                
                # Check each bird for collisions
                x = 0
                while x < len(birds):
                    collision, death_pos = check_collision(birds[x], pipe)
                    if collision:
                        if death_pos:
                            death_markers.append(DeathMarker(*death_pos))
                        ge[x].fitness -= 1
                        birds.pop(x)
                        nets.pop(x)
                        ge.pop(x)
                    elif not pipe.passed and birds[x].x > pipe.x + PIPE_WIDTH:
                        pipe.passed = True
                        score += 1
                        # Extra reward for passing extreme pipes
                        ge[x].fitness += 8
                    x += 1
            
            # Remove and add new pipes as needed
            while len(pipes) > 0 and pipes[0].x < -PIPE_WIDTH:
                pipes.pop(0)
                # Add new pipe with opposite height of the last pipe
                is_high = not (pipes[-1].gap_y <= PIPE_TOP_MARGIN + 10)  # Check if last pipe was low
                pipes.append(create_extreme_pipe(pipes[-1].x + PIPE_SPACING, is_high))
            
            # Draw the current game state
            draw_game(SCREEN, background, pipes, birds, score, death_markers)
        
        return best_genome
        
    except pygame.error:
        sys.exit()

def run_neat(config_path):
    try:
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
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except SystemExit:
        print("\nTraining terminated")

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption("EXTREME " + GAME_TITLE)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_neat(config_path)