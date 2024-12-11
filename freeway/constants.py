# constants.py
# Base/reference resolution with widescreen aspect ratio for better forward visibility
import pygame
pygame.init()

# Get current screen info
screen_info = pygame.display.Info()
MARGIN = 100  # Margin from screen edges

# Base resolution targets (widescreen for better forward visibility)
BASE_WIDTH = 1920
BASE_HEIGHT = 1080

# Calculate available space
available_width = screen_info.current_w - MARGIN
available_height = screen_info.current_h - MARGIN

# Use whichever is smaller: base size or available space
SCREEN_WIDTH = min(BASE_WIDTH, available_width)
SCREEN_HEIGHT = min(BASE_HEIGHT, available_height)

# Window title
GAME_TITLE = "AutoDrive"

# Center the window
os_x_pos = (screen_info.current_w - SCREEN_WIDTH) // 2
os_y_pos = (screen_info.current_h - SCREEN_HEIGHT) // 2
import os
os.environ['SDL_VIDEO_WINDOW_POS'] = f"{os_x_pos},{os_y_pos}"
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Colors
ROAD_COLOR = (40, 40, 40)        # Dark grey
LANE_COLOR = (255, 255, 255)     # White
PLAYER_COLOR = (0, 255, 0)       # Green
NPC_COLOR = (255, 0, 0)         # Red
SHOULDER_COLOR = (150, 150, 150) # Light grey

# Game settings
FPS = 60

# Road configuration
NUM_LANES = 4
LANE_WIDTH = SCREEN_HEIGHT // (NUM_LANES + 1)  # Extra space for shoulders
SHOULDER_WIDTH = LANE_WIDTH

# Car dimensions (scaled to lane width)
CAR_LENGTH = LANE_WIDTH * 0.8
CAR_WIDTH = LANE_WIDTH * 0.6

# Traffic settings
TRAFFIC_DENSITY = 0.3  # Percentage of max possible cars
MIN_CAR_SPACING = CAR_LENGTH * 2  # Minimum space between cars
NUM_CARS_VISIBLE_AHEAD = 3  # Per lane
SPAWN_DISTANCE = SCREEN_WIDTH + CAR_LENGTH  # Where new cars spawn
DESPAWN_DISTANCE = -CAR_LENGTH  # Where to remove cars that are too far behind

# Physics (all speeds in pixels per frame)
MAX_VELOCITY = SCREEN_WIDTH * 0.035        # Maximum player velocity
MIN_VELOCITY = SCREEN_WIDTH * 0.015       # Minimum player velocity
MAX_ACCELERATION = MAX_VELOCITY * 0.1     # Maximum acceleration per frame
MAX_DECELERATION = MAX_VELOCITY * 0.2     # Maximum deceleration per frame
LANE_CHANGE_SPEED = LANE_WIDTH * 0.1      # Speed of lane change maneuver

# NPC behavior
MIN_NPC_VELOCITY = MAX_VELOCITY * 0.4     # Slowest NPCs
MAX_NPC_VELOCITY = MAX_VELOCITY * 0.9     # Fastest NPCs
MAX_NPC_ACCELERATION = MAX_ACCELERATION * 0.5
MAX_NPC_DECELERATION = MAX_DECELERATION * 0.5
NPC_LANE_CHANGE_PROBABILITY = 0.001       # Chance per frame to initiate lane change
MIN_NPC_FOLLOWING_DISTANCE = CAR_LENGTH * 2

# Reward settings
BASE_REWARD_PER_FRAME = 0.1
COLLISION_PENALTY = -1000
OFFROAD_PENALTY = -500
REWARD_SPEED_MULTIPLIER = 1.0  # This will multiply by current_speed / MAX_VELOCITY

# Input preprocessing
MAX_VISION_DISTANCE = SCREEN_WIDTH * 0.8  # How far ahead AI can see
DISTANCE_BUCKETS = 10  # Number of distance divisions for input normalization

# Debug visualization
SHOW_DEBUG_INFO = True
DEBUG_FONT_SIZE = int(SCREEN_HEIGHT * 0.02)
DEBUG_COLOR = (255, 255, 0)  # Yellow