# game_init.py
import pygame
from dataclasses import dataclass

@dataclass
class GameConstants:
    """Container for all game constants, both scaled and unscaled"""
    # Required parameters (no defaults)
    SCREEN_WIDTH: int
    SCREEN_HEIGHT: int
    PIXELS_PER_METER: float
    GROUND_HEIGHT: int
    LANDING_PAD_WIDTH: int
    TERRAIN_ROUGHNESS: int
    LANDER_WIDTH: int
    LANDER_HEIGHT: int
    LEG_LENGTH: int
    GRAVITY: float
    MAIN_ENGINE_POWER: float
    SIDE_ENGINE_POWER: float
    SAFE_LANDING_VELOCITY: float
    LANDING_PAD_TOLERANCE: float
    POS_NORMALIZATION: float
    VEL_NORMALIZATION: float
    
    # Optional parameters (with defaults)
    ANGULAR_DAMPING: float = 0.5
    LINEAR_DAMPING: float = 0.1
    SAFE_LANDING_ANGLE: float = 18.0
    INITIAL_FUEL: float = 200.0
    MAIN_ENGINE_FUEL_COST: float = 1.0
    SIDE_ENGINE_FUEL_COST: float = 0.5
    LANDING_REWARD: float = 100.0
    CRASH_PENALTY: float = -100.0
    OUT_OF_BOUNDS_PENALTY: float = -100.0
    OUT_OF_FUEL_PENALTY: float = -100.0
    SAFETY_VIOLATION_PENALTY: float = -10000.0
    ANGLE_NORMALIZATION: float = 3.14159
    FPS: int = 60
    DT: float = 1.0 / 60

def initialize_game(width: int, height: int) -> GameConstants:
    """Initialize pygame and create scaled game constants"""
    pygame.init()
    
    # Calculate scale factors
    relative_unit = min(width, height)
    height_scale = height / 600  # Use 600px as baseline height
    
    return GameConstants(
        # Screen/window
        SCREEN_WIDTH=width,
        SCREEN_HEIGHT=height,
        PIXELS_PER_METER=relative_unit / 30,
        
        # Terrain (as percentages of screen)
        GROUND_HEIGHT=int(height * 0.917),
        LANDING_PAD_WIDTH=int(width * 0.125),
        TERRAIN_ROUGHNESS=int(height * 0.033),
        
        # Lander
        LANDER_WIDTH=int(width * 0.025),
        LANDER_HEIGHT=int(height * 0.05),
        LEG_LENGTH=int(height * 0.017),
        
        # Physics (scaled by height)
        GRAVITY=40.0 * height_scale,
        MAIN_ENGINE_POWER=1500.0 * height_scale,
        SIDE_ENGINE_POWER=250.0 * height_scale,
        
        # Game parameters
        SAFE_LANDING_VELOCITY=80.0 * height_scale,
        LANDING_PAD_TOLERANCE=int(height * 0.008),
        
        # Normalization factors
        POS_NORMALIZATION=relative_unit / 2,
        VEL_NORMALIZATION=80.0 * height_scale * 2  # 2x safe landing velocity
    )

# Global constants instance
CONST = None

def get_constants() -> GameConstants:
    """Get the global constants instance"""
    if CONST is None:
        raise RuntimeError("Game not initialized. Call initialize_game() first.")
    return CONST

def init() -> GameConstants:
    """Initialize the game with default resolution"""
    global CONST
    CONST = initialize_game(800, 600)
    return CONST