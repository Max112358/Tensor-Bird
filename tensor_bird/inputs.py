from constants import (
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    PIPE_WIDTH,
    PIPE_GAP
)

def get_pipe_inputs(bird, current_pipe, next_pipe=None):
    """
    Calculate normalized neural network inputs for the bird.
    
    Args:
        bird: Bird object containing position and velocity information
        current_pipe: The nearest Pipe object ahead of the bird
        next_pipe: The second nearest Pipe object ahead of the bird (optional)
    
    Returns:
        tuple: (
            normalized_bird_height,
            normalized_distance_to_pipe_x,
            normalized_distance_to_current_gap_center,
            normalized_distance_to_next_gap_center
        )
    """
    # Calculate center points of pipe gaps
    current_gap_center = current_pipe.gap_y + (PIPE_GAP / 2)
    
    inputs = [
        bird.y / SCREEN_HEIGHT,  # Bird height
        (current_pipe.x + PIPE_WIDTH - bird.x) / SCREEN_WIDTH,  # Distance to pipe's right edge
        (bird.y - current_gap_center) / SCREEN_HEIGHT,  # Distance to current pipe gap center
    ]
    
    if next_pipe:
        next_gap_center = next_pipe.gap_y + (PIPE_GAP / 2)
        inputs.append((bird.y - next_gap_center) / SCREEN_HEIGHT)  # Distance to next pipe gap center
    
    return tuple(inputs)