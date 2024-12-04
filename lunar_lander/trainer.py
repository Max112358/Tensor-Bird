import neat
import numpy as np
from environment import MultiLanderEnv
import time

class LanderTrainer:
    def __init__(self, num_landers: int = 20):
        self.env = MultiLanderEnv(num_landers=num_landers)
        self.generation = 0
        
    def eval_genomes(self, genomes, config):
        # Keep track of active networks and their genomes
        nets = []
        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            nets.append((genome, net))
            
        # Initialize environment
        states = self.env.reset()
        
        # Training parameters
        max_steps = 2000  # Increase maximum steps per episode
        min_steps = 200   # Minimum steps before checking termination
        step = 0
        
        # Main training loop
        while step < max_steps:
            # Get actions from networks
            actions = []
            for i, (state, (genome, net)) in enumerate(zip(states, nets)):
                output = net.activate(state)
                action = np.argmax(output)
                actions.append(action)
            
            # Step environment
            states, rewards, dones, info = self.env.step(actions)
            step += 1
            
            # Update fitness
            for i, ((genome, _), reward) in enumerate(zip(nets, rewards)):
                if reward != 0:  # Only update non-zero rewards
                    genome.fitness += reward

            # Render at a controlled framerate
            self.env.render()
            time.sleep(1/60)  # Cap at 60 FPS
            
            # Only check termination after minimum steps
            if step > min_steps:
                active_count = sum(1 for info in info['landers'] if info['active'])
                if active_count == 0:
                    break
        
        print(f"Generation {self.generation} completed in {step} steps")
        self.generation += 1
    
    def run(self, config_path: str, n_generations: int = 50):
        config = neat.Config(
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
        
        winner = pop.run(self.eval_genomes, n_generations)
        return winner, stats
    
    def close(self):
        self.env.close()