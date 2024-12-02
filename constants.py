# constants.py
import pygame

# Screen setup
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1100
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