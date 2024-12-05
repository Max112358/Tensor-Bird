import pygame
import neat
import os
import sys
import csv
from datetime import datetime
from itertools import product
import numpy as np
from constants import *
from bird import Bird
from pipe import Pipe 
from background import Background
from game_utils import check_collision, draw_game
from death_marker import DeathMarker

class HyperparameterTest:
    def __init__(self):
        self.target_fitness = 200
        self.max_generations = 10
        self.results = []
        self.best_performance = {
            'generations': float('inf'),
            'fitness': 0,
            'params': None
        }
        
        # Parameters with balanced min/max ranges
        self.param_ranges = {
            'bias_init_stdev': np.arange(0.1, 1.5, 0.2),    # 7 values
            'bias_range': np.arange(1.0, 10.0, 2.0),        # 5 values - will be used for both +/-
            'bias_mutate_power': np.arange(0.05, 0.2, 0.05)   # 4 values
        }
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_file = f'bias_param_results_{self.timestamp}.csv'
        self.setup_results_file()

    def setup_results_file(self):
        headers = [
            'bias_init_stdev', 
            'bias_range',      # Single column for range since min/max are linked
            'bias_mutate_power',
            'generations_to_target',
            'max_fitness_achieved',
            'timestamp'
        ]
        with open(self.results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def test_parameter_set(self, params):
        config_path = os.path.join(os.path.dirname(__file__), "config-feedforward.txt")
        config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Set the min/max values symmetrically based on the range parameter
        config.genome_config.bias_init_stdev = params['bias_init_stdev']
        config.genome_config.bias_max_value = params['bias_range']
        config.genome_config.bias_min_value = -params['bias_range']
        config.genome_config.bias_mutate_power = params['bias_mutate_power']
        
        pop = neat.Population(config)
        
        # Create the initial species
        pop.species.speciate(config, pop.population, 0)
        
        best_fitness = 0
        for generation in range(self.max_generations):
            gen_best = 0
            for _, genome in pop.population.items():
                genome.fitness = 0
            
            fitness, target_reached = self.eval_genomes(list(pop.population.items()), config)
            best_fitness = max(best_fitness, fitness)
            
            if target_reached:
                return generation + 1, best_fitness
            
            if generation < self.max_generations - 1:
                species_set = pop.species
                pop.population = pop.reproduction.reproduce(config, species_set,
                                                         pop.config.pop_size, generation)
                pop.species.speciate(config, pop.population, generation + 1)
        
        return self.max_generations, best_fitness

    def print_best_performance(self):
        print("\n=== BEST PERFORMANCE SO FAR ===")
        print(f"Generations: {self.best_performance['generations']}")
        print(f"Max Fitness: {self.best_performance['fitness']:.2f}")
        print("Parameters:")
        for param, value in self.best_performance['params'].items():
            print(f"  {param}: {value:.3f}")
        print("============================\n")

    def update_best_performance(self, generations, fitness, params):
        if generations < self.best_performance['generations'] or \
           (generations == self.best_performance['generations'] and 
            fitness > self.best_performance['fitness']):
            self.best_performance['generations'] = generations
            self.best_performance['fitness'] = fitness
            self.best_performance['params'] = params.copy()
            return True
        return False

    def eval_genomes(self, genomes, config):
        birds = []
        nets = []
        ge = []
        death_markers = []
        max_fitness = 0
        
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
        
        while len(birds) > 0:
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            background.move()
            
            for marker in death_markers[:]:
                marker.move()
                if marker.is_offscreen():
                    death_markers.remove(marker)
            
            pipe_ind = 0
            next_pipe_ind = 1
            if len(birds) > 0:
                if len(pipes) > 1 and birds[0].x > pipes[0].x + PIPE_WIDTH:
                    pipe_ind = 1
                    next_pipe_ind = 2
            
            next_pipe = pipes[next_pipe_ind] if next_pipe_ind < len(pipes) else pipes[pipe_ind]
            
            for x, bird in enumerate(birds):
                bird.move()
                # Only fitness measure is survival time
                ge[x].fitness += 0.1
                
                output = nets[x].activate((
                    bird.y / SCREEN_HEIGHT,
                    bird.velocity / MAX_FALL_SPEED,
                    abs(bird.y - pipes[pipe_ind].height) / SCREEN_HEIGHT,
                    abs(bird.y - pipes[pipe_ind].bottom_y) / SCREEN_HEIGHT,
                    (pipes[pipe_ind].x - bird.x) / SCREEN_WIDTH,
                    abs(bird.y - next_pipe.height) / SCREEN_HEIGHT,
                    abs(bird.y - next_pipe.bottom_y) / SCREEN_HEIGHT,
                    (next_pipe.x - bird.x) / SCREEN_WIDTH
                ))
                
                if output[0] > 0.5:
                    bird.jump()
                
                max_fitness = max(max_fitness, ge[x].fitness)
                
                if ge[x].fitness >= self.target_fitness:
                    return max_fitness, True
            
            for pipe in pipes:
                pipe.move()
                
                x = 0
                while x < len(birds):
                    collision, death_pos = check_collision(birds[x], pipe)
                    if collision:
                        if death_pos:
                            death_markers.append(DeathMarker(*death_pos))
                        # Remove bird when it dies (no penalty)
                        birds.pop(x)
                        nets.pop(x)
                        ge.pop(x)
                    elif not pipe.passed and birds[x].x > pipe.x + PIPE_WIDTH:
                        pipe.passed = True
                        score += 1
                    x += 1
            
            while len(pipes) > 0 and pipes[0].x < -PIPE_WIDTH:
                pipes.pop(0)
                pipes.append(Pipe(pipes[-1].x + PIPE_SPACING))
            
            draw_game(SCREEN, background, pipes, birds, score, death_markers)
            
        return max_fitness, False

    def run_tests(self):
        try:
            pygame.init()
            pygame.display.set_caption("Hyperparameter Testing")
            
            total_combinations = np.prod([len(range) for range in self.param_ranges.values()])
            print(f"Total parameter combinations to test: {total_combinations}")
            
            current_test = 0
            for params in product(*self.param_ranges.values()):
                current_test += 1
                param_dict = dict(zip(self.param_ranges.keys(), params))
                
                print(f"\nTesting combination {current_test}/{total_combinations}")
                print("Parameters:", param_dict)
                
                generations, max_fitness = self.test_parameter_set(param_dict)
                
                if self.update_best_performance(generations, max_fitness, param_dict):
                    print("\n🌟 NEW BEST PERFORMANCE! 🌟")
                    self.print_best_performance()
                
                result = list(params) + [generations, max_fitness, datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                
                with open(self.results_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(result)
                
                print(f"Generations to target: {generations}")
                print(f"Max fitness achieved: {max_fitness:.2f}")
                
        except KeyboardInterrupt:
            print("\nTesting interrupted by user")
            print("\nFinal best performance:")
            self.print_best_performance()
        except Exception as e:
            print(f"\nError during testing: {str(e)}")
            raise  # Re-raise the exception to see the full traceback
        finally:
            pygame.quit()

if __name__ == "__main__":
    tester = HyperparameterTest()
    tester.run_tests()