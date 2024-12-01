import numpy as np
import csv
import os

class GameStateHandler:
    def __init__(self, filename="game_states.txt"):
        self.filename = filename
        # Added y_error to header
        self.header = ['distance_to_pipe', 'current_y', 'velocity', 'pipe_y', 'y_error', 'action']
        
    def save_frame(self, game_state, action, append=True):
        """
        Save a single frame of game state to file
        action should be 1 for jump, 0 for no jump
        """
        mode = 'a' if append else 'w'
        file_exists = os.path.exists(self.filename)
        
        with open(self.filename, mode, newline='') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists and mode == 'w':
                writer.writerow(self.header)
            
            # Write game state and action
            writer.writerow([
                game_state['distance_to_pipe'],
                game_state['current_y'],
                game_state['velocity'],
                game_state['pipe_y'],
                game_state['y_error'],  # Include y_error in saved data
                action
            ])
    
    def save_game_session(self, states_and_actions):
        """
        Save multiple frames from a complete game session
        states_and_actions should be list of (game_state, action) tuples
        """
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            for state, action in states_and_actions:
                writer.writerow([
                    state['distance_to_pipe'],
                    state['current_y'],
                    state['velocity'],
                    state['pipe_y'],
                    state['y_error'],  # Include y_error in saved data
                    action
                ])
    
    def load_training_data(self):
        """
        Load all game states and actions for training
        Returns X (states) and y (actions) as numpy arrays
        """
        states = []
        actions = []
        
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                # Convert strings to floats for state data
                state = [float(x) for x in row[:-1]]  # Now includes y_error
                action = int(row[-1])
                
                states.append(state)
                actions.append(action)
        
        return np.array(states), np.array(actions)

# Example usage
if __name__ == "__main__":
    handler = GameStateHandler()
    
    # Example: Saving individual frames
    game_state = {
        'distance_to_pipe': 200,
        'current_y': 150,
        'velocity': -5,
        'pipe_y': 250
    }
    handler.save_frame(game_state, action=1)  # Save with jump action
    
    # Example: Saving multiple frames at once
    game_session = [
        (game_state, 1),
        (game_state, 0),
        # ... more states and actions
    ]
    handler.save_game_session(game_session)
    
    # Example: Loading data for training
    X_train, y_train = handler.load_training_data()
    print(f"Loaded {len(X_train)} training examples")