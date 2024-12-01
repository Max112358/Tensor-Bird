from genetics import EnhancedGeneticTracker, EnhancedFitnessCalculator, AdvancedSpeciation, MutationOptimizer
import neat
import numpy as np

class EnhancedFlappyTrainer:
    def __init__(self, config_path):
        # Initialize genetic components
        self.genetic_tracker = EnhancedGeneticTracker()
        self.fitness_calculator = EnhancedFitnessCalculator()
        self.speciation = AdvancedSpeciation()
        self.mutation_optimizer = MutationOptimizer()
        
        # Load NEAT config
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
        
        # Initialize population with enhanced tracking
        self.population = neat.Population(self.config)
        
    def evaluate_bird(self, bird, game_state):
        """Track bird stats for enhanced fitness calculation"""
        return {
            'height_history': [bird.bird_y],
            'velocity_history': [bird.velocity],
            'time_alive': game_state['game_time'],
            'pipes_cleared': game_state['score']
        }
        
    def run_generation(self, genomes, config):
        """Enhanced main game loop with genetic optimizations"""
        birds_list = []
        models_list = []
        genomes_list = []
        bird_stats = defaultdict(lambda: {'height_history': [], 'velocity_history': []})
        
        # Initialize birds and neural networks
        for genome_id, genome in genomes:
            birds_list.append(Bird(bird_starting_x_position, bird_starting_y_position))
            genome.fitness = 0
            genomes_list.append(genome)
            model = neat.nn.FeedForwardNetwork.create(genome, config)
            models_list.append(model)
        
        # Initialize game objects
        floor = Floor(floor_starting_y_position)
        pipes_list = [Pipe(pipe_starting_x_position + i * pipe_horizontal_gap) 
                     for i in range(pipe_max_num)]
        
        game_time = 0
        score = 0
        running = True
        
        while running and len(birds_list) > 0:
            game_time += 1/FPS
            
            # Update game objects
            floor.move()
            for pipe in pipes_list:
                pipe.move()
            
            # Process each bird
            for idx, (bird, model, genome) in enumerate(zip(birds_list, models_list, genomes_list)):
                # Track bird stats
                bird_stats[idx]['height_history'].append(bird.bird_y)
                bird_stats[idx]['velocity_history'].append(bird.velocity)
                
                # Get next pipe
                pipe_idx = get_index(pipes_list, [bird])
                next_pipe = pipes_list[pipe_idx]
                
                # Calculate neural network inputs
                inputs = (
                    bird.x - next_pipe.x,  # delta_x
                    bird.y - next_pipe.top_pipe_height,  # delta_y_top
                    bird.y - next_pipe.bottom_pipe_topleft  # delta_y_bottom
                )
                
                # Get network output
                output = model.activate(inputs)
                
                # Update bird
                if output[0] > prob_threshold_to_jump:
                    bird.jump()
                bird.move()
                
                # Check collision
                if collide(bird, next_pipe, floor, screen):
                    # Calculate final fitness
                    game_state = {
                        'distance_traveled': bird.x - bird_starting_x_position,
                        'game_time': game_time,
                        'score': score,
                    }
                    
                    stats = {
                        'height_history': np.array(bird_stats[idx]['height_history']),
                        'velocity_history': np.array(bird_stats[idx]['velocity_history']),
                        'time_alive': game_time,
                        'pipes_cleared': score
                    }
                    
                    genome.fitness = self.fitness_calculator.calculate_fitness(game_state, stats)
                    
                    # Remove bird
                    birds_list.pop(idx)
                    models_list.pop(idx)
                    genomes_list.pop(idx)
                    continue
            
            # Update score
            score = len([p for p in pipes_list if p.x + p.IMG_WIDTH < birds_list[0].x if p.get('scored', False)])
            
            # Draw game state
            draw_game(screen, birds_list, pipes_list, floor, score, generation, game_time)
            
        # Update species
        species = self.speciation.speciate(genomes_list)
        
        # Track generation stats
        best_fitness = max(genome.fitness for _, genome in genomes)
        avg_fitness = sum(genome.fitness for _, genome in genomes) / len(genomes)
        self.genetic_tracker.track_generation(generation, best_fitness, avg_fitness, len(species))
        
        # Optimize mutation rates
        new_mutation_rate = self.mutation_optimizer.adjust_mutation_rates(
            self.genetic_tracker.generation_stats['best_fitness']
        )
        
        # Apply new mutation rate to config
        self.config.genome_config.mutation_rate = new_mutation_rate

    def run(self, num_generations=50):
        """Run the enhanced training process"""
        self.population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        self.population.add_reporter(stats)
        
        winner = self.population.run(self.run_generation, num_generations)
        
        return winner, stats

def main():
    trainer = EnhancedFlappyTrainer('config-feedforward.txt')
    winner, stats = trainer.run(50)
    
    # Visualize results
    node_names = {-1:'delta_x', -2: 'delta_y_top', -3:'delta_y_bottom', 0:'Jump or Not'}
    draw_net(trainer.config, winner, True, node_names=node_names)
    plot_stats(stats, ylog=False, view=True)
    plot_species(stats, view=True)
    
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    main()