# freeway_trainer.py
import pygame
import neat
import configparser
import numpy as np
import pickle
from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, GAME_TITLE,
    BASE_REWARD_PER_FRAME, COLLISION_PENALTY, MAX_VELOCITY,
    NUM_LANES
)
from player_car import PlayerCar
from traffic_manager import TrafficManager
from game_visualizer import GameVisualizer
from ai_input_processor import get_car_inputs, get_input_size, get_output_size
import os
import sys

class FreewayTrainer:
    def __init__(self, config_path, checkpoint_file=None):
        """Initialize the training environment"""
        pygame.init()
        pygame.display.set_caption(f"{GAME_TITLE} - AI Training")
        
        # Calculate input size dynamically
        input_size = get_input_size()
        output_size = get_output_size()
        
        # Read and modify the config file
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Update the DefaultGenome section with calculated sizes
        config['DefaultGenome']['num_inputs'] = str(input_size)
        config['DefaultGenome']['num_outputs'] = str(output_size)
        
        # Write the modified config to a temporary file
        temp_config_path = 'temp_config.txt'
        with open(temp_config_path, 'w') as temp_config:
            config.write(temp_config)
        
        # Initialize NEAT with modified config
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            temp_config_path
        )
        
        # Clean up temporary config file
        os.remove(temp_config_path)
        
        # Load checkpoint or create new population
        if checkpoint_file and os.path.exists(checkpoint_file):
            print(f"Loading checkpoint: {checkpoint_file}")
            self.population = neat.Checkpointer.restore_checkpoint(checkpoint_file)
        else:
            self.population = neat.Population(self.config)
        
        # Add reporters
        self.population.add_reporter(neat.StdOutReporter(True))
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(self.stats)
        self.population.add_reporter(
            neat.Checkpointer(5, filename_prefix='freeway-checkpoint-')
        )
        
        # Initialize game components
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.visualizer = GameVisualizer(self.screen)
        
    def create_ai_cars(self, genomes, config):
        """Create AI cars with proper boundary initialization"""
        ai_cars = []
        
        road_height = self.visualizer.road_bottom - self.visualizer.road_top
        lane_height = road_height / NUM_LANES
        
        for i, (genome_id, genome) in enumerate(genomes):
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            
            # Calculate spawn position
            lane = i % NUM_LANES
            start_y = self.visualizer.road_top + (lane * lane_height) + (lane_height / 2)
            start_x = SCREEN_WIDTH * 0.2
            
            # Create car with proper position
            car = PlayerCar(start_x, start_y, brain=net)
            
            # Set proper boundaries immediately
            car.top_boundary = self.visualizer.road_top
            car.bottom_boundary = self.visualizer.road_bottom
            car.left_boundary = 0
            car.right_boundary = SCREEN_WIDTH
            
            # Store additional info
            car.genome = genome
            car.genome_id = genome_id
            car.current_lane = lane
            
            # Initialize genome fitness to 0
            genome.fitness = 0.0  # Explicitly set to float
            
            ai_cars.append(car)
            
            # Debug output
            print(f"\nCar {genome_id} created:")
            print(f"Position: ({car.x}, {car.y})")
            print(f"Lane: {lane}")
            print(f"Boundaries: top={car.top_boundary}, bottom={car.bottom_boundary}")
        
        return ai_cars
    
    def eval_genomes(self, genomes, config):
        """Evaluate all genomes in the current generation"""
        # Create AI cars for all genomes
        ai_cars = self.create_ai_cars(genomes, config)
        
        # Initialize game state
        traffic_manager = TrafficManager(
            self.visualizer.road_top,
            self.visualizer.road_bottom
        )
        traffic_manager.spawn_initial_traffic()
        
        # Training parameters
        max_frames = 2000
        frame_count = 0
        
        # Track maximum fitness and distance for visualization
        max_fitness = 0
        max_distance = 0
        best_car = None
        
        # Game loop
        while frame_count < max_frames and any(car.is_active for car in ai_cars):
            frame_count += 1
            
            # Get delta time
            dt = self.visualizer.update_fps() / 1000.0
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Update all active AI cars
            active_cars = [car for car in ai_cars if car.is_active]
            
            # If all cars are inactive, break the loop
            if not active_cars:
                break
                
            for car in active_cars:
                # Update car
                car.update(traffic_manager, dt)
                
                # Check for collisions
                if traffic_manager.check_collision(car):
                    car.handle_collision()
                    car.is_active = False
                else:
                    # Update genome fitness
                    speed_multiplier = car.velocity / MAX_VELOCITY
                    fitness_gain = BASE_REWARD_PER_FRAME * speed_multiplier * dt
                    car.genome.fitness += fitness_gain
                    
                    # Track best performing car
                    if car.genome.fitness > max_fitness:
                        max_fitness = car.genome.fitness
                        max_distance = car.total_distance
                        best_car = car
            
            # Update traffic manager with active cars
            if active_cars:
                # Use the best performing car as the reference for traffic
                traffic_manager.update(dt, best_car if best_car else active_cars[0])
            
            # Visualize current state
            self.visualizer.draw_frame(
                ai_cars,
                traffic_manager,
                max_fitness,
                max_distance
            )
        
        # Ensure all genomes have their final fitness values
        for car in ai_cars:
            if car.genome.fitness is None:
                car.genome.fitness = 0
            
        print(f"Generation complete. Max fitness: {max_fitness:.2f}")
    
    def run(self, generations=50):
        """Run the training process"""
        try:
            winner = self.population.run(self.eval_genomes, generations)
            
            # Save the winner
            with open('winner.pkl', 'wb') as f:
                pickle.dump(winner, f)
            
            print('\nBest genome:\n{!s}'.format(winner))
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            pygame.quit()

if __name__ == '__main__':
    # Setup paths
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    
    # Get checkpoint file if specified
    checkpoint_file = None
    if len(sys.argv) > 2 and sys.argv[1] == '-checkpoint':
        checkpoint_file = sys.argv[2]
    
    # Create and run trainer
    trainer = FreewayTrainer(config_path, checkpoint_file)
    trainer.run(generations=50)