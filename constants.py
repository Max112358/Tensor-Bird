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

# Calculate scale factors
SCALE_X = SCREEN_WIDTH / BASE_WIDTH
SCALE_Y = SCREEN_HEIGHT / BASE_HEIGHT
SCALE = min(SCALE_X, SCALE_Y)  # Use minimum to maintain aspect ratio

# Screen setup - center the window
os_x_pos = (screen_info.current_w - SCREEN_WIDTH) // 2
os_y_pos = (screen_info.current_h - SCREEN_HEIGHT) // 2
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{os_x_pos},{os_y_pos}"
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Colors
SKY_BLUE = (135, 206, 235)

# Game settings
FPS = 120
GAME_TITLE = "Flappy Bird AI"

# Sizes relative to base resolution, then scaled
BIRD_SIZE = int(60 * SCALE)  # Base: 60px
PIPE_WIDTH = int(140 * SCALE)  # Base: 140px
PIPE_GAP = int(300 * SCALE)    # Base: 300px
PIPE_SPACING = int(400 * SCALE)   # Base: 400px

# Safe margins for pipes (relative to original values)
PIPE_TOP_MARGIN = int(100 * SCALE)    # Base: 100px from top
PIPE_BOTTOM_MARGIN = int(100 * SCALE)  # Base: 100px from bottom

# Positions - scaled relative to screen dimensions
FLOOR_Y = SCREEN_HEIGHT
BIRD_START_X = int(SCREEN_WIDTH * 0.2)        # 20% from left edge
BIRD_START_Y = int(SCREEN_HEIGHT * 0.45)      # Slightly above middle
FIRST_PIPE_X = int(SCREEN_WIDTH * 0.625)      # 62.5% across screen

# Physics - using original values scaled
PIPE_VELOCITY = int(2.5 * SCALE)  # Slower base speed
BIRD_JUMP_VELOCITY = float(-4 * SCALE)  # Less powerful jump
GRAVITY = float(0.15 * SCALE)  # Reduced gravity
MAX_FALL_SPEED = float(4 * SCALE)  # Lower terminal velocity

# Pipe height range - allow full range of screen minus margins
PIPE_MIN_HEIGHT = PIPE_TOP_MARGIN
PIPE_MAX_HEIGHT = SCREEN_HEIGHT - PIPE_BOTTOM_MARGIN - PIPE_GAP

# Game parameters
VISIBLE_PIPES = 5  # Number of pipes visible at once