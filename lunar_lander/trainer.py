import neat
import numpy as np
from environment import MultiLanderEnv
import time
import os
from game_init import get_constants
import pickle
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from input_handler import InputHandler
from best_genome_logger import BestGenomeLogger



class LanderTrainer:
    fittest_genomes = []  # Will store top genomes by fitness
    MAX_STORED_GENOMES = 7

    def __init__(self, num_landers: int = 20, checkpoint_interval: int = 5, fast_mode: bool = False, inject_genomes: bool = False):
        """
        Initialize the trainer with genome logging
        """
        # Initialize genome logger first
        self.const = get_constants()
        self.genome_logger = BestGenomeLogger()
        self.logger = self.genome_logger.logger
        self.logger.info("Initializing LanderTrainer")
        
        try:
            # Initialize components
            self.env = MultiLanderEnv(num_landers=num_landers, fast_mode=fast_mode)
            self.logger.debug("Environment initialized")
            
            self.input_handler = InputHandler()
            self.logger.debug("Input handler initialized")
            
            # Training state
            self.generation = 0
            self.checkpoint_interval = checkpoint_interval
            self.best_fitness = float('-inf')
            self.generation_stats = []
            self.fast_mode = fast_mode
            self.population = None
            self.best_genome_io = None  # To store inputs and outputs of the best genome
            self.inject_genomes = inject_genomes  # Store the injection flag
            
            # Create required directories
            os.makedirs('checkpoints', exist_ok=True)
            os.makedirs('outputs', exist_ok=True)
            
            # Initialize checkpointer
            self.checkpointer = neat.Checkpointer(
                generation_interval=checkpoint_interval,
                time_interval_seconds=None,
                filename_prefix='checkpoints/neat-checkpoint-'
            )
            
            # Patch NEAT with genome logging
            self.genome_logger.patch_neat()
            self.logger.info("LanderTrainer initialization complete")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during trainer initialization: {str(e)}", exc_info=True)
            else:
                print(f"Failed to initialize trainer: {str(e)}")
            raise

    def _update_top_genomes(self, genome):
        """Update the list of top genomes if the given genome qualifies"""
        if not hasattr(genome, 'fitness') or genome.fitness is None:
            return

        # Create a copy of the genome for storage
        genome_copy = pickle.loads(pickle.dumps(genome))
        
        # If we haven't reached max capacity, just append
        if len(self.fittest_genomes) < self.MAX_STORED_GENOMES:
            if not any(g.fitness == genome.fitness for g in self.fittest_genomes):  # Avoid duplicates
                self.fittest_genomes.append(genome_copy)
                self.fittest_genomes.sort(key=lambda x: x.fitness, reverse=True)
                print(f"\n[Top Genomes] Added new genome with fitness {genome.fitness:.2f}")
                print(f"Current top 5 fitnesses: {[f'{g.fitness:.2f}' for g in self.fittest_genomes[:5]]}")
        # If the new genome is better than the worst stored genome
        elif genome.fitness > self.fittest_genomes[-1].fitness:
            if not any(g.fitness == genome.fitness for g in self.fittest_genomes):  # Avoid duplicates
                old_worst = self.fittest_genomes[-1].fitness
                self.fittest_genomes[-1] = genome_copy
                self.fittest_genomes.sort(key=lambda x: x.fitness, reverse=True)
                print(f"\n[Top Genomes] Replaced genome (fitness: {old_worst:.2f}) with better genome (fitness: {genome.fitness:.2f})")
                print(f"Current top 5 fitnesses: {[f'{g.fitness:.2f}' for g in self.fittest_genomes[:5]]}")

    def save_checkpoint(self) -> None:
        """Manually trigger a checkpoint save using NEAT's checkpointer"""
        if self.checkpointer and self.population:
            # Force an immediate checkpoint save
            self.checkpointer.save_checkpoint(
                self.population.config,
                self.population.population,
                self.population.species,
                self.generation
            )
            self.logger.info(f"Manually saved checkpoint at generation {self.generation}")

    def eval_genomes(self, genomes, config) -> None:
        """
        Evaluate each genome.
        
        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT config object
        """
        # Initialize evaluation stats
        gen_stats = {
            'max_fitness': float('-inf'),
            'avg_fitness': 0,
            'successful_landings': 0,
            'total_fitness': 0
        }
        
        # Create neural networks for each genome
        networks = []
        genome_list = []
        
        # Create copies of top genomes with new IDs if injection is enabled
        if self.inject_genomes:
            max_genome_id = max(gid for gid, _ in genomes)
            next_genome_id = max_genome_id + 1
            
            injected_genomes = []
            for top_genome in self.fittest_genomes:
                # Create a deep copy of the genome
                genome_copy = pickle.loads(pickle.dumps(top_genome))
                # Reset the genome's fitness
                genome_copy.fitness = None
                # Add to the injection list with a new ID
                injected_genomes.append((next_genome_id, genome_copy))
                next_genome_id += 1
            
            # Combine original genomes with injected ones
            all_genomes = genomes + injected_genomes
        else:
            all_genomes = genomes
        
        # Process all genomes (original + injected)
        for genome_id, genome in all_genomes:
            genome.fitness = None  # Reset fitness for fresh evaluation
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            networks.append(net)
            genome_list.append(genome)
        
        # Evaluate genomes in batches
        for i in range(0, len(networks), self.env.num_landers):
            batch_networks = networks[i:i + self.env.num_landers]
            batch_genomes = genome_list[i:i + self.env.num_landers]
            
            # Pad batch if needed
            while len(batch_networks) < self.env.num_landers:
                batch_networks.append(None)
                batch_genomes.append(None)
            
            states = self.env.reset()  # Fresh environment for each batch
            done = False
            step = 0
            total_landers = len(states)
            episode_landings = 0
            
            while not done:
                # Get actions for each lander
                actions = []
                for idx, (network, lander) in enumerate(zip(batch_networks, self.env.landers)):
                    if network is None:
                        actions.append(0)  # No-op for padding networks
                        continue
                        
                    try:
                        state = self.input_handler.get_state(lander, self.env.terrain)
                        output = network.activate(state)
                        
                        # Convert network output to action
                        action = 0
                        if len(output) >= 3:
                            # Convert from [-1,1] to [0,1] range if using tanh activation
                            normalized_outputs = [(x + 1) / 2 for x in output]
                            threshold = 0.5
                            
                            # Check thrusters with threshold
                            if normalized_outputs[0] > threshold:  # Left thruster
                                action = 1
                            elif normalized_outputs[2] > threshold:  # Right thruster
                                action = 3
                            elif normalized_outputs[1] > threshold:  # Main thruster
                                action = 2
                            
                        actions.append(action)
                    except Exception as e:
                        print(f"Error activating network: {e}")
                        actions.append(0)
                
                try:
                    # Step environment
                    states, rewards, dones, info = self.env.step(actions)
                    step += 1
                    
                    completed = self.env.get_completed_landers()
                    active_count = self.env.get_active_landers()
                    
                    # Print progress
                    print(f"\rBatch {i//self.env.num_landers + 1} Step {step}: Active: {active_count}/{total_landers} | " + 
                        f"Landed: {completed.get('landed', 0)} | " +
                        f"Crashed: {completed.get('crashed', 0)} | " +
                        f"Out of Bounds: {completed.get('out_of_bounds', 0)} | " +
                        f"Out of Fuel: {completed.get('out_of_fuel', 0)}", end='   ')
                    
                    if info.get('quit', False):
                        print("\nWindow closed, ending training")
                        self.env.close()
                        raise KeyboardInterrupt
                    
                    # Handle rewards for each genome
                    for idx, (genome, reward) in enumerate(zip(batch_genomes, rewards)):
                        if genome is not None:
                            # Initialize fitness if needed
                            if not hasattr(genome, 'fitness') or genome.fitness is None:
                                genome.fitness = 0
                            
                            # Set the fitness to the current total reward
                            # This represents the accumulated reward over the full episode
                            genome.fitness = reward
                            
                            
                            if genome.fitness > self.const.LANDING_REWARD:
                                episode_landings += 1
                                self._update_top_genomes(genome)
                                
                            
                    
                    if all(dones) or not self.env.is_running():
                        done = True
                    
                    if not self.fast_mode:
                        time.sleep(1/60)
                        
                except Exception as e:
                    print(f"\nError during environment step: {e}")
                    done = True
                
            print()  # New line after step updates
        
        # Calculate generation statistics
        gen_stats = {
            'max_fitness': float('-inf'),
            'avg_fitness': 0,
            'total_fitness': 0,
            'successful_landings': 0
        }

        # Use a single pass to calculate all statistics
        valid_genomes = [g for g in genome_list if g is not None and hasattr(g, 'fitness') and g.fitness is not None]
        if valid_genomes:
            fitnesses = [g.fitness for g in valid_genomes]
            gen_stats['max_fitness'] = max(fitnesses)
            gen_stats['total_fitness'] = sum(fitnesses)
            gen_stats['avg_fitness'] = gen_stats['total_fitness'] / len(valid_genomes)
            gen_stats['successful_landings'] = sum(1 for f in fitnesses if f >= self.const.LANDING_REWARD)

            # Update best fitness and genome if we found a better one
            if gen_stats['max_fitness'] > self.best_fitness:
                self.best_fitness = gen_stats['max_fitness']
                # Find the best genome
                self.best_genome = next(g for g in valid_genomes if g.fitness == gen_stats['max_fitness'])
                self.best_genome_id = self.best_genome.key

        # Store generation statistics
        self.generation_stats.append(gen_stats)

        # Update top genomes list after evaluation
        for genome in valid_genomes:
            self._update_top_genomes(genome)
            if genome.fitness >= self.const.LANDING_REWARD:
                # Manually trigger checkpoint save on successful landing
                self.save_checkpoint()

        # Print generation summary
        print(f"\nGeneration {self.generation} completed:")
        print(f"Max Fitness: {gen_stats['max_fitness']:.2f}")
        print(f"Avg Fitness: {gen_stats['avg_fitness']:.2f}")
        print(f"Successful Landings: {gen_stats['successful_landings']}")
        
        self.generation += 1
    
    def run(self, config_path: str, n_generations: int = 50, 
            checkpoint_file: str = None) -> Tuple[Optional[neat.genome.DefaultGenome], 
                                                neat.statistics.StatisticsReporter]:
        """Run the training process with genome logging"""
        try:
            # Load configuration
            config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                config_path
            )
            
            if checkpoint_file:
                self.logger.info(f"Restoring from checkpoint: {checkpoint_file}")
                self.population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
                self.logger.info(f"Restored to generation {self.population.generation}")
            else:
                self.logger.info("Creating new population")
                self.population = neat.Population(config)
            
            # Initialize best genome tracking
            self.best_genome = None
            self.best_genome_id = None
            
            # Add reporters
            self.population.add_reporter(neat.StdOutReporter(True))
            stats = neat.StatisticsReporter()
            self.population.add_reporter(stats)
            
            # Add checkpointer
            self.checkpointer = neat.Checkpointer(
                generation_interval=self.checkpoint_interval,
                time_interval_seconds=None,
                filename_prefix='checkpoints/neat-checkpoint-'
            )
            self.population.add_reporter(self.checkpointer)
            
            # Run for specified number of generations
            remaining_generations = n_generations - self.generation
            winner = self.population.run(self.eval_genomes, remaining_generations)
            
            if winner:
                self.logger.info("\nWinner found!")
                self.logger.info(f"Winner fitness: {winner.fitness}")
                
                # Save the winner
                winner_path = os.path.join('outputs', 'winner.pkl')
                self.logger.info(f"Saving winner to {winner_path}")
                with open(winner_path, 'wb') as f:
                    pickle.dump(winner, f)
                    
                # Get best overall genome from logger
                best_genome, best_fitness = self.genome_logger.get_best_genome()
                if best_genome is not None:
                    self.logger.info(f"\nBest Overall Performance:")
                    self.logger.info(f"Fitness: {best_fitness}")
                    
                    # Save best genome
                    best_path = os.path.join('outputs', 'best_genome.pkl')
                    self.logger.info(f"Saving best genome to {best_path}")
                    with open(best_path, 'wb') as f:
                        pickle.dump(best_genome, f)
            else:
                self.logger.info("\nNo winner found")
                
            # Save final statistics
            self._save_training_stats()
            return winner, stats
            
        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user")
            self._save_training_stats()
            return None, stats
            
        except Exception as e:
            self.logger.error(f"Training stopped due to error: {str(e)}", exc_info=True)
            self._save_training_stats()
            raise
        
        finally:
            if hasattr(self, 'env'):
                self.env.close()
    
    def _save_best_genome(self, genome: neat.genome.DefaultGenome) -> None:
        """Save the best performing genome"""
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(genome, f)
    
    def _save_training_stats(self) -> None:
        """Save training statistics and generate plots"""
        # Don't try to plot if we have no complete generations
        if self.generation == 0 or not self.generation_stats:
            print("No training statistics to plot")
            return
        
        # Save raw statistics
        stats_dict = {
            'generation': list(range(self.generation)),
            'max_fitness': [stats['max_fitness'] for stats in self.generation_stats],
            'avg_fitness': [stats['avg_fitness'] for stats in self.generation_stats],
            'successful_landings': [stats['successful_landings'] for stats in self.generation_stats]
        }
        
        with open('training_stats.pkl', 'wb') as f:
            pickle.dump(stats_dict, f)
        
        # Generate and save plots
        plt.figure(figsize=(12, 8))
        
        # Fitness plot
        plt.subplot(2, 1, 1)
        plt.plot(stats_dict['generation'], stats_dict['max_fitness'], label='Max Fitness')
        plt.plot(stats_dict['generation'], stats_dict['avg_fitness'], label='Avg Fitness')
        plt.title('Fitness over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
        plt.grid(True)
        
        # Successful landings plot
        plt.subplot(2, 1, 2)
        plt.plot(stats_dict['generation'], stats_dict['successful_landings'])
        plt.title('Successful Landings per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Number of Successful Landings')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()
    
    def load_best_genome(self, config_path: str) -> Tuple[Optional[neat.genome.DefaultGenome], neat.config.Config]:
        """
        Load the best genome and its configuration
        
        Args:
            config_path: Path to NEAT configuration file
            
        Returns:
            Tuple of (genome, config) or (None, config) if no genome is found
        """
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        try:
            with open('best_genome.pkl', 'rb') as f:
                genome = pickle.load(f)
            return genome, config
        except FileNotFoundError:
            print("No saved genome found")
            return None, config
    
    def close(self) -> None:
        """Clean up resources"""
        self.env.close()
