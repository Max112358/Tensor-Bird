# checkpoint_manager.py
import pickle
from dataclasses import dataclass
from typing import List, Tuple
import pygame
import copy
import os

@dataclass
class GameState:
    """Stores complete game state at a checkpoint"""
    bird_state: Tuple[float, float, float]  # x, y, velocity of the best bird
    pipe_states: List[dict]  # Complete state of each pipe
    score: int
    generation: int
    best_fitness: float

class CheckpointManager:
    def __init__(self, checkpoint_interval=5, max_checkpoints=3):
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
        self.last_checkpoint_score = 0
        
    def should_save_checkpoint(self, score: int) -> bool:
        return score > 0 and score % self.checkpoint_interval == 0 and score > self.last_checkpoint_score
        
    def save_checkpoint(self, birds, pipes, score, generation, best_fitness):
        """Save complete game state using best bird"""
        # Save state of best performing bird
        best_bird = birds[0]  # Assuming first bird is representative
        bird_state = (best_bird.x, best_bird.y, best_bird.velocity)
        
        # Save complete pipe state including gaps and passed status
        pipe_states = []
        for pipe in pipes:
            pipe_state = {
                'x': pipe.x,
                'gap_y': pipe.gap_y,
                'passed': pipe.passed,
                'top_y': pipe.top_y,
                'bottom_y': pipe.bottom_y,
                'height': pipe.height,
                'top_rect': {
                    'x': pipe.top_rect.x,
                    'y': pipe.top_rect.y,
                    'width': pipe.top_rect.width,
                    'height': pipe.top_rect.height
                },
                'bottom_rect': {
                    'x': pipe.bottom_rect.x,
                    'y': pipe.bottom_rect.y,
                    'width': pipe.bottom_rect.width,
                    'height': pipe.bottom_rect.height
                }
            }
            pipe_states.append(pipe_state)
            
        state = GameState(
            bird_state=bird_state,
            pipe_states=pipe_states,
            score=score,
            generation=generation,
            best_fitness=best_fitness
        )
        
        self.checkpoints.append(state)
        self.last_checkpoint_score = score
        
        # Remove oldest checkpoint if we exceed max
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)
            
    def get_restore_point(self, backup_count=2):
        """Get a checkpoint from backup_count checkpoints ago"""
        if len(self.checkpoints) >= backup_count:
            return self.checkpoints[-backup_count]
        return None
        
    def restore_game_state(self, state: GameState, bird_class, pipe_class, num_birds=20):
        """Restore a complete game state from checkpoint"""
        # Create multiple birds at the saved position
        birds = []
        x, y, vel = state.bird_state
        for _ in range(num_birds):
            bird = bird_class(x, y)
            bird.velocity = vel
            birds.append(bird)
            
        # Restore complete pipe states
        pipes = []
        for pipe_state in state.pipe_states:
            pipe = pipe_class(pipe_state['x'])
            pipe.gap_y = pipe_state['gap_y']
            pipe.passed = pipe_state['passed']
            pipe.top_y = pipe_state['top_y']
            pipe.bottom_y = pipe_state['bottom_y']
            pipe.height = pipe_state['height']
            
            # Properly restore collision rectangles
            top_rect = pipe_state['top_rect']
            bottom_rect = pipe_state['bottom_rect']
            
            pipe.top_rect = pygame.Rect(
                top_rect['x'], 
                top_rect['y'], 
                top_rect['width'], 
                top_rect['height']
            )
            
            pipe.bottom_rect = pygame.Rect(
                bottom_rect['x'], 
                bottom_rect['y'], 
                bottom_rect['width'], 
                bottom_rect['height']
            )
            
            pipes.append(pipe)
            
        return birds, pipes, state.score

def save_checkpoint_to_file(checkpoint_manager, filename):
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint_manager, f)

def load_checkpoint_from_file(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None