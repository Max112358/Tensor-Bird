import neat
import numpy as np
from environment import MultiLanderEnv
import time
import os
import pickle
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from constants import MAX_STEPS_PER_EPISODE, MIN_STEPS_BEFORE_DONE

class LanderTrainer:
    def __init__(self, num_landers: int = 20, checkpoint_interval: int = 5, fast_mode: bool = False):
        """
        Initialize the trainer
        
        Args:
            num_landers: Number of landers to train simultaneously
            checkpoint_interval: How often to save checkpoints (in generations)
            fast_mode: Whether to run in fast mode without rendering
        """
        self.env = MultiLanderEnv(num_landers=num_landers, fast_mode=fast_mode)
        self.generation = 0
        self.checkpoint_interval = checkpoint_interval
        self.best_fitness = float('-inf')
        self.generation_stats = []
        self.fast_mode = fast_mode
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        
    def eval_genomes(self, genomes, config) -> None:
        """
        Evaluate all genomes in one batch
        
        Args:
            genomes: List of (genome_id, genome) tuples
            config: NEAT configuration
        """
        # Keep track of active networks and their genomes
        nets = []
        for genome_id, genome in genomes:
            genome.fitness = -100.0  # Start with penalty
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append((genome, net))
            
        # Initialize environment
        states = self.env.reset()
        
        # Training loop variables
        step = 0
        running = True
        generation_stats = {
            'max_fitness': float('-inf'),
            'avg_fitness': 0,
            'successful_landings': 0
        }
        
        # Main training loop
        while running and step < MAX_STEPS_PER_EPISODE:
            # Get actions from networks
            actions = []
            for i, (state, (genome, net)) in enumerate(zip(states, nets)):
                output = net.activate(state)
                action = np.argmax(output)
                actions.append(action)
            
            # Step environment
            states, rewards, dones, info = self.env.step(actions)
            step += 1
            
            # Check if window was closed
            if info.get('quit', False):
                print("\nWindow closed, ending training")
                running = False
                self.env.close()
                raise KeyboardInterrupt
            
            # Update fitness scores
            for i, ((genome, _), reward, lander_info) in enumerate(zip(nets, rewards, info['landers'])):
                genome.fitness += reward  # Always update fitness, not just for non-zero rewards
                
                # Track best fitness
                if genome.fitness > self.best_fitness:
                    self.best_fitness = genome.fitness
                    self._save_best_genome(genome)
                
                # Count successful landings
                if lander_info.get('reason') == 'landed':
                    generation_stats['successful_landings'] += 1
            
            # Control frame rate if not in fast mode
            if not self.fast_mode:
                time.sleep(1/60)  # Cap at 60 FPS
            
            # Check if episode should end
            if info['all_done'] or not self.env.is_running():
                break
        
        # Update generation statistics
        fitnesses = [genome.fitness for genome, _ in nets]
        generation_stats['max_fitness'] = max(fitnesses)
        generation_stats['avg_fitness'] = sum(fitnesses) / len(fitnesses)
        self.generation_stats.append(generation_stats)
        
        # Print generation summary
        print(f"\nGeneration {self.generation} completed:")
        print(f"Steps: {step}")
        print(f"Max Fitness: {generation_stats['max_fitness']:.2f}")
        print(f"Avg Fitness: {generation_stats['avg_fitness']:.2f}")
        print(f"Successful Landings: {generation_stats['successful_landings']}")
        
        # Print termination reasons
        completed = self.env.get_completed_landers()
        for reason, count in completed.items():
            print(f"{reason}: {count}")
        
        self.generation += 1
    
    def run(self, config_path: str, n_generations: int = 50) -> Tuple[Optional[neat.genome.DefaultGenome], neat.statistics.StatisticsReporter]:
        """
        Run the training process
        
        Args:
            config_path: Path to NEAT configuration file
            n_generations: Number of generations to train
            
        Returns:
            Tuple of (winner genome, statistics reporter)
        """
        # Load configuration
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        
        # Create population and add reporters
        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        
        try:
            # Run for specified number of generations
            winner = pop.run(self.eval_genomes, n_generations)
            
            # Save final statistics
            self._save_training_stats()
            
            return winner, stats
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self._save_training_stats()
            return None, stats
            
        except Exception as e:
            print(f"\nTraining stopped due to error: {e}")
            self._save_training_stats()
            raise
    
    def _save_best_genome(self, genome: neat.genome.DefaultGenome) -> None:
        """Save the best performing genome"""
        with open('best_genome.pkl', 'wb') as f:
            pickle.dump(genome, f)
    
    def _save_training_stats(self) -> None:
        """Save training statistics and generate plots"""
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