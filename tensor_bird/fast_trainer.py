import neat
import sys
import os
from datetime import datetime
from bird import Bird
from pipe import Pipe
from inputs import get_pipe_inputs
from constants import *

class DebugStats:
    def __init__(self):
        self.frames = 0
        self.pipes_cleared = 0
        self.total_distance = 0
        self.current_fitnesses = []
        self.best_genome = None
        self.best_fitness = -float('inf')
        
    def reset(self):
        self.__init__()

def save_checkpoint(config, population, species, generation):
    """Save the current population as a checkpoint file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'neat-checkpoint-{generation}-{timestamp}'
    
    # Create a Checkpointer instance
    checkpointer = neat.Checkpointer(generation_interval=1, filename_prefix=filename)
    checkpointer.save_checkpoint(config, population, species, generation)
    
    # Find the best performing genome
    best_genome = None
    best_fitness = float('-inf')
    
    for sid, s in species.species.items():
        for gid, genome in s.members.items():
            if genome.fitness is not None and genome.fitness > best_fitness:
                best_fitness = genome.fitness
                best_genome = genome
    
    if best_genome:
        print('\nBest genome:\n{!s}'.format(best_genome))
    
    print(f"\nCheckpoint saved as: {filename}")
    return filename

def fast_eval_genomes(genomes, config):
    birds = []
    nets = []
    ge = []
    stats = DebugStats()
    
    FITNESS_THRESHOLD = 60000.0  # Adjust this value as needed
    
    print("\n=== Starting New Evaluation ===")
    print(f"Number of genomes: {len(genomes)}")
    print(f"Fitness threshold: {FITNESS_THRESHOLD}")
    
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(BIRD_START_X, BIRD_START_Y))
        genome.fitness = 0
        ge.append(genome)
        stats.current_fitnesses.append(0)
    
    pipes = [Pipe(FIRST_PIPE_X + i * PIPE_SPACING) for i in range(VISIBLE_PIPES)]
    
    print("\nInitial state:")
    print(f"Birds: {len(birds)}")
    print(f"Initial pipe positions: {[int(p.x) for p in pipes]}")
    
    while len(birds) > 0:
        stats.frames += 1
        
        if stats.frames % 100 == 0:
            print(f"\nFrame {stats.frames}:")
            print(f"Birds alive: {len(birds)}")
            print(f"Best fitness: {max(stats.current_fitnesses) if stats.current_fitnesses else 0}")
            print(f"Pipes cleared: {stats.pipes_cleared}")
        
        # Check if any bird has reached the fitness threshold
        current_best_fitness = max(stats.current_fitnesses)
        if current_best_fitness >= FITNESS_THRESHOLD:
            print(f"\nFitness threshold {FITNESS_THRESHOLD} reached!")
            print(f"Final fitness: {current_best_fitness}")
            return True  # Signal to stop evolution
        
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
                pipe_ind = 1
        
        for x, bird in enumerate(birds):
            ge[x].fitness += 0.1
            stats.current_fitnesses[x] = ge[x].fitness
            
            if ge[x].fitness > stats.best_fitness:
                stats.best_fitness = ge[x].fitness
                stats.best_genome = ge[x]
                #print(f"\nNew best fitness: {stats.best_fitness:.2f}")
            
            prev_x = bird.x
            bird.move()
            distance_moved = bird.x - BIRD_START_X
            stats.total_distance = max(stats.total_distance, distance_moved)
            
            pipe = pipes[pipe_ind]
            next_pipe = pipes[pipe_ind + 1] if pipe_ind + 1 < len(pipes) else None
            inputs = get_pipe_inputs(bird, pipe, next_pipe)
            output = nets[x].activate(inputs)
            
            if output[0] > 0.5:
                bird.jump()
        
        for pipe in pipes:
            pipe.move()
            
            x = 0
            while x < len(birds):
                collision = (
                    birds[x].y < 0 or
                    birds[x].y + birds[x].height > FLOOR_Y or
                    birds[x].x < pipe.x + PIPE_WIDTH and
                    birds[x].x + birds[x].width > pipe.x and
                    (birds[x].y < pipe.gap_y or
                     birds[x].y + birds[x].height > pipe.gap_y + PIPE_GAP)
                )
                
                if collision:
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                    stats.current_fitnesses.pop(x)
                elif not pipe.passed and birds[x].x > pipe.x + PIPE_WIDTH:
                    pipe.passed = True
                    stats.pipes_cleared += 1
                    stats.current_fitnesses[x] = ge[x].fitness
                    #print(f"\nPipe cleared! New fitness: {ge[x].fitness:.2f}")
                x += 1
        
        while len(pipes) > 0 and pipes[0].x < -PIPE_WIDTH:
            pipes.pop(0)
            pipes.append(Pipe(pipes[-1].x + PIPE_SPACING))
    
    return False  # Signal to continue evolution

def run_fast_training(config_path, generations=50):
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
    
    # Custom evaluation loop to handle fitness threshold
    generation = 0
    while generation < generations:
        print(f"\n===== Generation {generation} =====")
        
        # Evaluate genomes
        fitness_threshold_reached = False
        for _, genome in pop.population.items():
            genome.fitness = 0
        
        fitness_threshold_reached = fast_eval_genomes(list(pop.population.items()), config)
        
        if fitness_threshold_reached:
            # Save checkpoint before exiting
            checkpoint_file = save_checkpoint(config, pop, pop.species, generation)
            print(f"\nFitness threshold reached! Checkpoint saved.")
            print(f"You can now load this bird using:")
            print(f"python main.py -load {checkpoint_file}")
            sys.exit(0)
            
        # Create next generation
        pop.population = pop.reproduction.reproduce(config, pop.species, pop.config.pop_size, generation)
        pop.species.speciate(config, pop.population, generation)
        generation += 1

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run_fast_training(config_path)