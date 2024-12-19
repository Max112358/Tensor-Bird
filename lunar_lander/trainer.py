import neat
import numpy as np
from environment import MultiLanderEnv
import time
import os
import pickle
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from input_handler import InputHandler
from best_genome_logger import BestGenomeLogger



class LanderTrainer:
    def __init__(self, num_landers: int = 20, checkpoint_interval: int = 5, fast_mode: bool = False):
        """
        Initialize the trainer with genome logging
        """
        # Initialize genome logger first
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
            
            # Create required directories
            os.makedirs('checkpoints', exist_ok=True)
            os.makedirs('outputs', exist_ok=True)
            
            # Patch NEAT with genome logging
            self.genome_logger.patch_neat()
            self.logger.info("LanderTrainer initialization complete")
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error during trainer initialization: {str(e)}", exc_info=True)
            else:
                print(f"Failed to initialize trainer: {str(e)}")
            raise
        
    def eval_genomes(self, genomes, config) -> None:
        print("\nInitial fitness values:")
        for genome_id, genome in genomes:
            genome.fitness = None  # Reset fitness even for elite genomes
        """Evaluate each genome"""
        # Initialize generation statistics
        gen_stats = {
            'max_fitness': float('-inf'),
            'avg_fitness': 0,
            'successful_landings': 0,
            'total_fitness': 0
        }
        
        # Create neural networks for each genome
        networks = []
        genome_list = []
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            networks.append(net)
            genome_list.append(genome)
            genome.fitness = 0
        
        # Evaluate genomes in batches of num_landers size
        for i in range(0, len(networks), self.env.num_landers):
            batch_networks = networks[i:i + self.env.num_landers]
            batch_genomes = genome_list[i:i + self.env.num_landers]
            
            # Pad with None if we don't have enough genomes to fill the batch
            while len(batch_networks) < self.env.num_landers:
                batch_networks.append(None)
                batch_genomes.append(None)
            
            states = self.env.reset()
            done = False
            step = 0
            total_landers = len(states)
            episode_landings = 0
            
            while not done:
                # Get actions for each lander using its own network
                actions = []
                for idx, (network, lander) in enumerate(zip(batch_networks, self.env.landers)):
                    if network is None:
                        actions.append(0)  # No-op for padding networks
                    else:
                        try:
                            # Use input handler to get normalized state
                            state = self.input_handler.get_state(lander, self.env.terrain)
                            
                            # Get raw outputs from network
                            output = network.activate(state)
                            
                            # Default to no thrust
                            action = 0
                            
                            '''
                            # Check thrusters in priority order (main > side thrusters)
                            if len(output) >= 3:  # Ensure we have all three outputs
                                if output[1] > 0.5:  # Main thruster has priority
                                    action = 2
                                elif output[0] > 0.5:  # Left thruster
                                    action = 1
                                elif output[2] > 0.5:  # Right thruster
                                    action = 3
                            '''
                                    
                            if len(output) >= 3:
                                # Convert from [-1,1] to [0,1] range if using tanh activation
                                normalized_outputs = [(x + 1) / 2 for x in output]
                                
                                # Use a lower threshold (e.g., 0.3)
                                threshold = 0.3
                                
                                # Check thrusters with lower threshold
                                if normalized_outputs[1] > threshold:  # Main thruster
                                    action = 2
                                elif normalized_outputs[0] > threshold:  # Left thruster
                                    action = 1
                                elif normalized_outputs[2] > threshold:  # Right thruster
                                    action = 3
                            
                            actions.append(action)
                        except Exception as e:
                            print(f"Error activating network: {e}")
                            actions.append(0)  # Fallback to no thrust
                
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
                    
                    # Update genome fitness for active networks
                    for genome, reward in zip(batch_genomes, rewards):
                        if genome is not None:
                            genome.fitness += reward
                    
                    # Track successful landings
                    if completed.get('landed', 0) > episode_landings:
                        episode_landings = completed.get('landed', 0)
                        gen_stats['successful_landings'] += episode_landings
                    
                    if all(dones) or not self.env.is_running():
                        done = True
                    
                    if not self.fast_mode:
                        time.sleep(1/60)
                        
                except Exception as e:
                    print(f"\nError during environment step: {e}")
                    done = True
                
            print()  # New line after step updates
        
        print("\nGeneration max fitness calculation:")
        
        # Update generation statistics
        for genome in genome_list:
            if genome is not None:
                gen_stats['max_fitness'] = max(gen_stats['max_fitness'], genome.fitness)
                gen_stats['total_fitness'] += genome.fitness
        
        # Calculate generation averages
        gen_stats['avg_fitness'] = gen_stats['total_fitness'] / len(genome_list)
        
        # Store generation statistics
        self.generation_stats.append(gen_stats)
        
        # Update best fitness
        if gen_stats['max_fitness'] > self.best_fitness:
            self.best_fitness = gen_stats['max_fitness']
        
        # Print generation summary
        print(f"\nGeneration {self.generation} completed:")
        print(f"Max Fitness: {gen_stats['max_fitness']:.2f}")
        print(f"Avg Fitness: {gen_stats['avg_fitness']:.2f}")
        print(f"Successful Landings: {gen_stats['successful_landings']}")
        
        self.generation += 1
        
        # Save checkpoint if needed
        if self.generation % self.checkpoint_interval == 0:
            self.save_checkpoint()
    
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
                try:
                    self.generation = int(checkpoint_file.split('-')[-1])
                    self.logger.info(f"Restored to generation {self.generation}")
                except ValueError:
                    self.logger.warning("Could not determine generation from checkpoint filename")
            else:
                self.logger.info("Creating new population")
                self.population = neat.Population(config)
            
            # Add reporters
            stats = neat.StatisticsReporter()
            self.population.add_reporter(stats)
            
            # Add checkpointer
            checkpointer = neat.Checkpointer(
                generation_interval=self.checkpoint_interval,
                time_interval_seconds=None,
                filename_prefix='checkpoints/neat-checkpoint-'
            )
            self.population.add_reporter(checkpointer)
            
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

    def save_checkpoint(self) -> None:
        """Save a checkpoint of the current training state"""
        filename = f'checkpoints/neat-checkpoint-{self.generation}'
        with open(filename, 'wb') as f:
            pickle.dump({
                'generation': self.generation,
                'best_fitness': self.best_fitness,
                'generation_stats': self.generation_stats
            }, f)
            
    def load_checkpoint(self, filename: str) -> Dict[str, Any]:
        """Load a training checkpoint"""
        with open(filename, 'rb') as f:
            checkpoint = pickle.load(f)
            self.generation = checkpoint['generation']
            self.best_fitness = checkpoint['best_fitness']
            self.generation_stats = checkpoint['generation_stats']
            return checkpoint