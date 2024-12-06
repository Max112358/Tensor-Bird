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
from inputs import get_pipe_inputs

def eval_genomes(genomes, config):
    try:
        birds = []
        nets = []
        ge = []
        death_markers = []
        best_genome = None
        best_fitness = -float('inf')
        
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            birds.append(Bird(BIRD_START_X, BIRD_START_Y))
            genome.fitness = 0
            ge.append(genome)
        
        background = Background(SCREEN_WIDTH, SCREEN_HEIGHT)
        pipes = [Pipe(FIRST_PIPE_X + i * PIPE_SPACING) for i in range(VISIBLE_PIPES)]
        score = 0
        clock = pygame.time.Clock()
        
        run = True
        while run and len(birds) > 0:
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    current_best = max(ge, key=lambda x: x.fitness)
                    print('\nBest genome at exit:\n{!s}'.format(current_best))
                    print(f'Final fitness: {current_best.fitness}')
                    pygame.quit()
                    sys.exit()
            
            background.move()
            
            for marker in death_markers[:]:
                marker.move()
                if marker.is_offscreen():
                    death_markers.remove(marker)
            
            # Determine which pipes to focus on
            pipe_ind = 0
            next_pipe_ind = 1
            if len(birds) > 0:
                if len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
                    pipe_ind = 1
                    next_pipe_ind = 2
            
            # Check if any bird has reached fitness threshold
            threshold_reached = False
            for genome in ge:
                if genome.fitness >= config.fitness_threshold:
                    threshold_reached = True
                    best_genome = genome
                    best_fitness = genome.fitness
                    print(f"\nFitness threshold {config.fitness_threshold} reached!")
                    print(f"Best fitness achieved: {best_fitness}")
                    run = False
                    break
                    
            if threshold_reached:
                break
            
            # Update all birds
            for x, bird in enumerate(birds):
                bird.move()
                ge[x].fitness += 0.1
                
                # Get the next pipe if available
                next_pipe = pipes[next_pipe_ind] if next_pipe_ind < len(pipes) else pipes[pipe_ind]
                
                # Get neural network inputs using the inputs module
                output = nets[x].activate(get_pipe_inputs(bird, pipes[pipe_ind], next_pipe))
                
                if output[0] > 0.5:
                    bird.jump()
                
                if ge[x].fitness > best_fitness:
                    best_fitness = ge[x].fitness
                    best_genome = ge[x]
            
            # Update and check all pipes
            for pipe in pipes:
                pipe.move()
                
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
                        ge[x].fitness += 5
                    x += 1
            
            while len(pipes) > 0 and pipes[0].x < -PIPE_WIDTH:
                pipes.pop(0)
                pipes.append(Pipe(pipes[-1].x + PIPE_SPACING))
            
            draw_game(SCREEN, background, pipes, birds, score, death_markers)
        
        return best_genome
        
    except pygame.error:
        sys.exit()

def run_neat(config_path, checkpoint_file=None):
    try:
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading from checkpoint: {checkpoint_file}")
            pop = neat.Checkpointer.restore_checkpoint(checkpoint_file)
            start_gen = pop.generation
            
            # Reconstruct population from species
            all_members = {}
            for species in pop.species.species.values():
                all_members.update(species.members)
            pop.population = all_members
            
        else:
            if checkpoint_file:
                print(f"Checkpoint file {checkpoint_file} not found. Starting fresh.")
            pop = neat.Population(config)
            start_gen = 0
        
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        checkpointer = neat.Checkpointer(5, filename_prefix='neat-checkpoint-')
        pop.add_reporter(checkpointer)
        
        remaining_gens = 50 - start_gen
        winner = pop.run(eval_genomes, remaining_gens)
        print('\nBest genome:\n{!s}'.format(winner))
        
    except KeyboardInterrupt:
        print("\nSaving checkpoint before exiting...")
        current_gen = pop.generation
        checkpointer.save_checkpoint(config, pop, pop.species, current_gen)
        print(f"Checkpoint saved as neat-checkpoint-{current_gen}")
    except SystemExit:
        print("\nTraining terminated")

if __name__ == "__main__":
    pygame.init()
    pygame.display.set_caption(GAME_TITLE)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    
    # Parse command line arguments
    checkpoint_file = None
    if len(sys.argv) > 2 and sys.argv[1] == '-load':
        checkpoint_file = sys.argv[2]
    
    run_neat(config_path, checkpoint_file)