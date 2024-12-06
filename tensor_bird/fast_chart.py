import neat
import sys
import os
from datetime import datetime
from bird import Bird
from pipe import Pipe
from inputs import get_pipe_inputs
from constants import *
import random
import json

def run_training_session():
    """Run a single training session and return generations needed to reach fitness"""
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
    )
    
    pop = neat.Population(config)
    
    max_fitness_reached = 0
    generation = 0
    
    while generation < 50:  # Cap at 50 generations
        generation += 1
        
        # Reset fitness for all genomes
        for _, genome in pop.population.items():
            genome.fitness = 0
            
        # Evaluate current generation
        birds = []
        nets = []
        ge = []
        
        for genome_id, genome in pop.population.items():
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append(net)
            birds.append(Bird(BIRD_START_X, BIRD_START_Y))
            ge.append(genome)
            
        pipes = [Pipe(FIRST_PIPE_X + i * PIPE_SPACING) for i in range(VISIBLE_PIPES)]
        
        # Run until all birds die
        while len(birds) > 0:
            pipe_ind = 0
            if len(birds) > 0 and len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
                pipe_ind = 1
                
            # Update birds
            for x, bird in enumerate(birds):
                ge[x].fitness += 0.1
                bird.move()
                
                # Get neural network decision
                pipe = pipes[pipe_ind]
                next_pipe = pipes[pipe_ind + 1] if pipe_ind + 1 < len(pipes) else None
                output = nets[x].activate(get_pipe_inputs(bird, pipe, next_pipe))
                
                if output[0] > 0.5:
                    bird.jump()
                    
                max_fitness_reached = max(max_fitness_reached, ge[x].fitness)
                
                # Check if we've reached target fitness
                if ge[x].fitness >= 5000:
                    return generation, ge[x].fitness
            
            # Update pipes and check collisions
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
                    elif not pipe.passed and birds[x].x > pipe.x + PIPE_WIDTH:
                        pipe.passed = True
                        ge[x].fitness += 5
                    x += 1
                    
            # Remove and add pipes as needed
            while len(pipes) > 0 and pipes[0].x < -PIPE_WIDTH:
                pipes.pop(0)
                pipes.append(Pipe(pipes[-1].x + PIPE_SPACING))
                
        # Create next generation
        pop.population = pop.reproduction.reproduce(config, pop.species, pop.config.pop_size, generation)
        pop.species.speciate(config, pop.population, generation)
        
    return generation, max_fitness_reached

def collect_training_data(num_sessions=50):
    """Run multiple training sessions and collect data"""
    results = []
    for session in range(num_sessions):
        print(f"Running session {session + 1}/{num_sessions}")
        generations, fitness = run_training_session()
        results.append({"generations": generations, "fitness": fitness})
        
    # Save results to file
    with open('training_results.json', 'w') as f:
        json.dump(results, f)
    
    return results

if __name__ == "__main__":
    results = collect_training_data()
    print("\nTraining complete! Results saved to training_results.json")