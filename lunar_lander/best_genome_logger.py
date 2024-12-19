import neat
import logging
import os
from datetime import datetime

class BestGenomeLogger:
    def __init__(self, log_dir: str = "genome_logs"):
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"best_genome_{timestamp}.log")
        
        # Configure logging
        self.logger = logging.getLogger('genome_logger')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Track best genome across generations
        self.best_genome = None
        self.best_fitness = float('-inf')
        
    def patch_neat(self):
        """Patch NEAT library with genome logging"""
        # Store original evaluation method
        original_eval_genomes = neat.Population.run
        logger = self.logger  # Store reference to logger
        genome_logger = self  # Store reference to self
        
        def eval_genomes_wrapper(self, fitness_function, n):
            """Wrap the evaluation function to track best genome"""
            def wrapped_fitness(genomes, config):
                # Call original fitness function
                fitness_function(genomes, config)
                
                # Find best genome in this generation
                best_genome = None
                best_fitness = float('-inf')
                
                for genome_id, genome in genomes:
                    if genome.fitness is not None and genome.fitness > best_fitness:
                        best_fitness = genome.fitness
                        best_genome = genome
                
                if best_genome is not None:
                    logger.info(f"\nBest Genome in Generation {self.generation}:")
                    logger.info(f"Fitness: {best_fitness}")
                    logger.info("Network Structure:")
                    logger.info("Nodes:")
                    for node_id, node in best_genome.nodes.items():
                        logger.info(f"  Node {node_id}: {node}")
                    logger.info("Connections:")
                    for conn_id, conn in best_genome.connections.items():
                        logger.info(f"  {conn_id}: weight={conn.weight}, enabled={conn.enabled}")
                    
                    # Track global best in the logger instance
                    if best_fitness > genome_logger.best_fitness:
                        genome_logger.best_fitness = best_fitness
                        genome_logger.best_genome = best_genome
                        logger.info("\nNew Best Genome Found!")
                        logger.info(f"New Best Fitness: {best_fitness}")
            
            # Call original run method with wrapped fitness function
            return original_eval_genomes(self, wrapped_fitness, n)
        
        # Apply patch
        neat.Population.run = eval_genomes_wrapper
        self.logger.info("NEAT library patched with genome logging")
        
    def log_generation_complete(self, generation, stats):
        """Log statistics at the end of each generation"""
        self.logger.info(f"\nGeneration {generation} Complete")
        if stats:
            mean = stats.get_fitness_mean()[-1] if stats.get_fitness_mean() else None
            self.logger.info(f"Mean Fitness: {mean}")
            self.logger.info(f"Best Fitness Ever: {self.best_fitness}")
            
    def get_best_genome(self):
        """Return the best genome found so far"""
        return self.best_genome, self.best_fitness