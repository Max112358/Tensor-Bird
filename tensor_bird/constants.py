# constants.py
import os
# Base/reference resolution
BASE_WIDTH = 1600
BASE_HEIGHT = 900
# Initialize pygame before getting screen info
import pygame
pygame.init()
# Get current screen info
screen_info = pygame.display.Info()
MARGIN = 120  # Margin from screen edges
# Calculate available space
available_width = screen_info.current_w - MARGIN
available_height = screen_info.current_h - MARGIN
# Use whichever is smaller: base size or available space
SCREEN_WIDTH = min(BASE_WIDTH, available_width)
SCREEN_HEIGHT = min(BASE_HEIGHT, available_height)

# Window title
GAME_TITLE = "Tensor Bird"

# Screen setup - center the window
os_x_pos = (screen_info.current_w - SCREEN_WIDTH) // 2
os_y_pos = (screen_info.current_h - SCREEN_HEIGHT) // 2
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{os_x_pos},{os_y_pos}"
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))


# Colors
SKY_BLUE = (135, 206, 235)

# Game settings
FPS = 120

# Sizes relative to screen
BIRD_SIZE = int(SCREEN_HEIGHT * 0.055)  # ~60px at 1100 height
PIPE_WIDTH = int(SCREEN_WIDTH * 0.0875)  # ~140px at 1600 width
PIPE_GAP = int(SCREEN_HEIGHT * 0.273)    # ~300px at 1100 height
PIPE_SPACING = int(SCREEN_WIDTH * 0.25)   # ~400px at 1600 width

# Safe margins for pipes (as percentage of screen height)
PIPE_TOP_MARGIN = int(SCREEN_HEIGHT * 0.1)    # 10% from top
PIPE_BOTTOM_MARGIN = int(SCREEN_HEIGHT * 0.1)  # 10% from bottom

# Pipe height range
PIPE_MIN_HEIGHT = PIPE_TOP_MARGIN + PIPE_GAP
PIPE_MAX_HEIGHT = SCREEN_HEIGHT - PIPE_BOTTOM_MARGIN - PIPE_GAP

# Positions
FLOOR_Y = SCREEN_HEIGHT
BIRD_START_X = SCREEN_WIDTH * 0.2        # 20% from left edge
BIRD_START_Y = SCREEN_HEIGHT * 0.45      # Slightly above middle
FIRST_PIPE_X = SCREEN_WIDTH * 0.625      # 62.5% across screen

# Physics
PIPE_VELOCITY = SCREEN_WIDTH * 0.00156    # Scales with screen width
BIRD_JUMP_VELOCITY = SCREEN_HEIGHT * -0.00636  # Scales with screen height
GRAVITY = SCREEN_HEIGHT * 0.000227        # Scales with screen height
MAX_FALL_SPEED = SCREEN_HEIGHT * 0.00727  # Scales with screen height

# Game parameters
VISIBLE_PIPES = 5  # Number of pipes visible at once

""" print("Screen dimensions:", SCREEN_WIDTH, SCREEN_HEIGHT)
print("Bird properties:")
print(f"- Size: {BIRD_SIZE}")
print(f"- Jump velocity: {BIRD_JUMP_VELOCITY}")
print(f"- Max fall speed: {MAX_FALL_SPEED}")
print(f"- Gravity: {GRAVITY}")
print("Pipe properties:")
print(f"- Width: {PIPE_WIDTH}")
print(f"- Gap: {PIPE_GAP}")
print(f"- Spacing: {PIPE_SPACING}")
print(f"- Velocity: {PIPE_VELOCITY}") """