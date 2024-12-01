from fast_training_flappy_bird import FastTrainingFlappyBird
from ai import create_flappy_model, save_model
from collections import deque
import numpy as np
import gc
import tensorflow as tf
from tensorflow import keras
import random

class FastLearningFlappyBird:
    def __init__(self, model_path='saved_model', training_speed=10):
        self.model_path = model_path
        
        try:
            self.model = create_flappy_model(model_path)
            if self.model is None:
                raise ValueError("Model creation failed")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
            
        self.training_speed = training_speed
        self.game = FastTrainingFlappyBird()
        
        # Enhanced state tracking
        self.games_played = 0
        self.last_loss = float('inf')
        self.best_distance = 0
        self.average_distance = 0
        
        # Performance tracking
        self.recent_games_buffer = []
        self.buffer_size = 50
        self.successful_runs = []
        
        # Experience replay buffers
        self.replay_buffer = deque(maxlen=50000)  # Store last 50k experiences
        self.success_buffer = deque(maxlen=10000)  # Store experiences from good runs
        self.min_replay_size = 1000  # Min experiences before training
        self.batch_size = 32
        self.training_epochs = 20  # Train more on each batch
        self.success_threshold = 1000  # Distance that defines a "good" run
        
        # Dynamic exploration
        self.min_exploration_rate = 0.05
        self.max_exploration_rate = 0.3
        self.exploration_decay = 0.9995
        
        print("Initializing with random training data...")
        self.initialize_random_data(200)
    
    def get_state_input(self, game_state):
        """Get the normalized 5-feature state input for the model"""
        return [
            game_state['distance_to_pipe'] / 500,  
            game_state['current_y'] / 400,         
            game_state['velocity'] / 10,          
            game_state['pipe_y'] / 400,          
            game_state['y_error'] / 10000         
        ]
    
    def initialize_random_data(self, num_games):
        """Initialize the model with random gameplay data"""
        print("Generating initial random training data...")
        
        for _ in range(num_games):
            self.game.init_game_state()
            
            while not self.game.game_over:
                game_state = self.game.get_game_state()
                if game_state:
                    state_input = self.get_state_input(game_state)
                    action = 1 if np.random.random() < 0.1 else 0
                    self.save_experience(state_input, action)
                    if action:
                        self.game.velocity = self.game.jump_strength
                self.game.update()
    
    def save_experience(self, state, action):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action))
        
        # If this was part of a successful run, also save to success buffer
        if self.game.distance_traveled >= self.success_threshold:
            self.success_buffer.append((state, action))
    
    def predict_action(self, game_state):
        """Get prediction from model with controlled exploration"""
        state_input = self.get_state_input(game_state)
        inputs = np.array([state_input])
        
        prediction = self.model.predict(inputs, verbose=0)
        
        # Dynamic exploration rate
        exploration_rate = max(
            self.min_exploration_rate,
            self.max_exploration_rate * (self.exploration_decay ** self.games_played)
        )
        
        # Safety checks
        if game_state['velocity'] < -8:  # Moving up too fast
            return False
        if game_state['current_y'] < 30:  # Too close to ceiling
            return False
        if game_state['current_y'] > 350:  # Too close to ground
            return True
        
        # Enhanced decision making
        if np.random.random() < exploration_rate:
            # Smarter random decisions based on state
            if game_state['current_y'] > game_state['pipe_y'] + 100:
                return True  # More likely to jump if below pipe
            elif game_state['current_y'] < game_state['pipe_y'] - 100:
                return False  # Less likely to jump if above pipe
            return np.random.random() > 0.7
        
        # Use softmax probabilities for smoother decision making
        jump_prob = prediction[0][1]
        return np.random.random() < jump_prob
    
    def train_model(self):
        """Train on mixed batch of regular and successful experiences"""
        if len(self.replay_buffer) < self.min_replay_size:
            return
            
        # Create training batch with mix of regular and successful experiences
        training_batch = []
        
        # Add regular experiences
        training_batch.extend(random.sample(self.replay_buffer, 
                                         min(len(self.replay_buffer), self.batch_size)))
        
        # Add successful experiences if available
        if self.success_buffer:
            training_batch.extend(random.sample(self.success_buffer,
                                             min(len(self.success_buffer), self.batch_size // 4)))
        
        # Prepare data for training
        states = np.array([exp[0] for exp in training_batch])
        actions = np.array([exp[1] for exp in training_batch])
        y_categorical = tf.keras.utils.to_categorical(actions, num_classes=2)
        
        # Train for multiple epochs on this batch
        history = self.model.fit(
            states, y_categorical,
            epochs=self.training_epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        
        new_loss = history.history['loss'][-1]
        
        if new_loss < self.last_loss:
            improvement = (self.last_loss - new_loss) / self.last_loss * 100
            print(f"Model improved! Loss decreased by {improvement:.2f}% from {self.last_loss:.4f} to {new_loss:.4f}")
            self.model.save(self.model_path)
        self.last_loss = new_loss
    
    def run_fast_training(self, num_games=1000, display_interval=50):
        """Run training at maximum speed"""
        try:
            for game_num in range(num_games):
                if game_num % 100 == 0:
                    gc.collect()  # Periodic memory cleanup
                
                self.game.init_game_state()
                
                # Run single game
                while not self.game.game_over:
                    game_state = self.game.get_game_state()
                    if game_state:
                        should_jump = self.predict_action(game_state)
                        state_input = self.get_state_input(game_state)
                        self.save_experience(state_input, 1 if should_jump else 0)
                        if should_jump:
                            self.game.velocity = self.game.jump_strength
                    self.game.update()
                
                # Update stats
                self.games_played += 1
                final_distance = self.game.distance_traveled
                
                if final_distance > self.best_distance:
                    self.best_distance = final_distance
                    print(f"New best distance: {self.best_distance:.1f}")
                    save_model(self.model, self.model_path)
                
                self.average_distance = (self.average_distance * (self.games_played - 1) + 
                                    final_distance) / self.games_played
                
                # Train more frequently with replay buffer
                if game_num % (self.training_speed // 2) == 0:
                    self.train_model()
                
                # Display progress periodically
                if game_num % display_interval == 0:
                    print(f"Game {game_num}/{num_games}: "
                          f"Distance = {final_distance:.1f}, "
                          f"Avg = {self.average_distance:.1f}, "
                          f"Best = {self.best_distance:.1f}, "
                          f"Loss = {self.last_loss:.4f}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        finally:
            save_model(self.model, self.model_path)
            print("Final model saved")