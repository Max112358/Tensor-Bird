from collections import defaultdict
import numpy as np
import random

class EnhancedGeneticTracker:
    def __init__(self):
        self.generation_stats = defaultdict(list)
        self.species_stats = defaultdict(list)
        self.innovation_tracker = {}
        self.elite_genomes = []
        self.max_elite_size = 5
        
    def track_generation(self, generation, best_fitness, avg_fitness, species_count):
        """Track generation-level statistics"""
        self.generation_stats['generation'].append(generation)
        self.generation_stats['best_fitness'].append(best_fitness)
        self.generation_stats['avg_fitness'].append(avg_fitness)
        self.generation_stats['species_count'].append(species_count)

    def update_elite_genomes(self, genome, fitness):
        """Keep track of best performing genomes"""
        if len(self.elite_genomes) < self.max_elite_size:
            self.elite_genomes.append((genome, fitness))
            self.elite_genomes.sort(key=lambda x: x[1], reverse=True)
        elif fitness > self.elite_genomes[-1][1]:
            self.elite_genomes[-1] = (genome, fitness)
            self.elite_genomes.sort(key=lambda x: x[1], reverse=True)

class EnhancedFitnessCalculator:
    def __init__(self):
        # Weights for different components of fitness
        self.distance_weight = 1.0
        self.height_control_weight = 0.5
        self.smooth_flight_weight = 0.3
        self.pipe_clear_bonus = 50
        self.survival_time_weight = 0.1
        
    def calculate_fitness(self, game_state, bird_stats):
        """Calculate a more nuanced fitness score"""
        base_fitness = game_state['distance_traveled'] * self.distance_weight
        
        # Reward height control
        height_variance = np.var(bird_stats['height_history'])
        height_control = max(0, (1000 - height_variance)) * self.height_control_weight
        
        # Reward smooth flight
        velocity_variance = np.var(bird_stats['velocity_history'])
        smooth_flight = max(0, (100 - velocity_variance)) * self.smooth_flight_weight
        
        # Bonus for clearing pipes
        pipe_bonus = game_state['pipes_cleared'] * self.pipe_clear_bonus
        
        # Time survived
        survival_bonus = game_state['time_alive'] * self.survival_time_weight
        
        return base_fitness + height_control + smooth_flight + pipe_bonus + survival_bonus
        
class AdvancedSpeciation:
    def __init__(self, compatibility_threshold=3.0):
        self.threshold = compatibility_threshold
        self.species = []
        self.representative_genomes = []
        
    def calculate_compatibility(self, genome1, genome2):
        """Calculate compatibility distance between two genomes"""
        disjoint_coeff = 1.0
        weight_coeff = 0.4
        
        # Get matching and disjoint genes
        genes1 = set(genome1.genes)
        genes2 = set(genome2.genes)
        matching = genes1 & genes2
        disjoint = (genes1 ^ genes2)
        
        # Calculate average weight difference for matching genes
        weight_diff = 0
        if matching:
            weight_diff = sum(abs(genome1.genes[gene].weight - genome2.genes[gene].weight) 
                            for gene in matching) / len(matching)
        
        # Calculate compatibility distance
        compatibility = (disjoint_coeff * len(disjoint) / max(len(genes1), len(genes2)) +
                       weight_coeff * weight_diff)
                       
        return compatibility
        
    def speciate(self, population):
        """Divide population into species"""
        self.species = []
        unassigned = population.copy()
        
        while unassigned:
            representative = unassigned.pop(0)
            species = [representative]
            
            remaining = unassigned.copy()
            unassigned = []
            
            for genome in remaining:
                if self.calculate_compatibility(representative, genome) < self.threshold:
                    species.append(genome)
                else:
                    unassigned.append(genome)
                    
            if len(species) > 0:
                self.species.append(species)
                self.representative_genomes.append(representative)
        
        return self.species

class MutationOptimizer:
    def __init__(self):
        self.base_mutation_rate = 0.3
        self.min_mutation_rate = 0.1
        self.adaptation_rate = 0.95
        
    def adjust_mutation_rates(self, species_fitness_history):
        """Dynamically adjust mutation rates based on fitness improvements"""
        if len(species_fitness_history) < 2:
            return self.base_mutation_rate
            
        recent_improvement = (species_fitness_history[-1] - species_fitness_history[-2]) 
        recent_improvement /= max(species_fitness_history[-2], 1)
        
        if recent_improvement <= 0:
            # Increase mutation rate if fitness is stagnating
            new_rate = min(self.base_mutation_rate * (1 / self.adaptation_rate), 1.0)
        else:
            # Decrease mutation rate if fitness is improving
            new_rate = max(self.base_mutation_rate * self.adaptation_rate, self.min_mutation_rate)
            
        self.base_mutation_rate = new_rate
        return new_rate