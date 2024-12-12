# constants.py
import pygame

# Base constants that don't require scaling
ANGULAR_DAMPING = 0.5
LINEAR_DAMPING = 0.1
SAFE_LANDING_ANGLE = 18.0  # degrees
INITIAL_FUEL = 200.0
MAIN_ENGINE_FUEL_COST = 1.0
SIDE_ENGINE_FUEL_COST = 0.5
LANDING_REWARD = 100.0
CRASH_PENALTY = -100.0
OUT_OF_BOUNDS_PENALTY = -100.0
OUT_OF_FUEL_PENALTY = -100.0
SAFETY_VIOLATION_PENALTY = -10000.0
ANGLE_NORMALIZATION = 3.14159
FPS = 60
DT = 1.0 / FPS

# Base values for scaled constants (as percentages or ratios)
BASE_LANDING_PAD_WIDTH_PCT = 0.125  # 12.5% of viewport width
BASE_TERRAIN_ROUGHNESS_PCT = 0.033  # 3.3% of viewport height
BASE_GROUND_HEIGHT_PCT = 0.917  # 91.7% of viewport height
BASE_LANDER_WIDTH_PCT = 0.025  # 2.5% of viewport width
BASE_LANDER_HEIGHT_PCT = 0.05  # 5% of viewport height
BASE_LEG_LENGTH_PCT = 0.017  # 1.7% of viewport height

# Physics base values (will be scaled by height ratio)
BASE_GRAVITY = 40.0
BASE_MAIN_ENGINE_POWER = 1500.0
BASE_SIDE_ENGINE_POWER = 250.0
BASE_SAFE_LANDING_VELOCITY = 80.0
BASE_LANDING_PAD_TOLERANCE = 0.008  # 0.8% of viewport height

class GameScaling:
    """Handles dynamic resolution scaling for the game"""
    _instance = None
    
    def __init__(self, viewport_width: int, viewport_height: int):
        # Store viewport dimensions
        self.width = viewport_width
        self.height = viewport_height
        self.base_height = 600  # Original base height for scaling
        
        # Calculate scale factors
        self.height_scale = viewport_height / self.base_height
        self.relative_unit = min(viewport_width, viewport_height)
        
        # Calculate all scaled values
        self._calculate_dimensions()
    
    def _calculate_dimensions(self):
        """Calculate all scaled dimensions and constants"""
        # Terrain dimensions
        self.landing_pad_width = int(self.width * BASE_LANDING_PAD_WIDTH_PCT)
        self.terrain_roughness = int(self.height * BASE_TERRAIN_ROUGHNESS_PCT)
        self.ground_height = int(self.height * BASE_GROUND_HEIGHT_PCT)
        
        # Lander dimensions
        self.lander_width = int(self.width * BASE_LANDER_WIDTH_PCT)
        self.lander_height = int(self.height * BASE_LANDER_HEIGHT_PCT)
        self.leg_length = int(self.height * BASE_LEG_LENGTH_PCT)
        
        # Physics values
        self.gravity = BASE_GRAVITY * self.height_scale
        self.main_engine_power = BASE_MAIN_ENGINE_POWER * self.height_scale
        self.side_engine_power = BASE_SIDE_ENGINE_POWER * self.height_scale
        self.safe_landing_velocity = BASE_SAFE_LANDING_VELOCITY * self.height_scale
        self.landing_pad_tolerance = int(self.height * BASE_LANDING_PAD_TOLERANCE)
        
        # Conversion factor
        self.pixels_per_meter = self.relative_unit / 30
    
    @classmethod
    def initialize(cls, viewport_width: int, viewport_height: int):
        """Initialize the global scaling instance"""
        if cls._instance is None:
            cls._instance = cls(viewport_width, viewport_height)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        """Get the global scaling instance"""
        if cls._instance is None:
            raise RuntimeError("GameScaling not initialized. Call initialize() first.")
        return cls._instance
    
    def normalize_position(self, x: float, y: float) -> tuple[float, float]:
        """Convert viewport coordinates to normalized [-1, 1] range"""
        norm_x = (x / self.width) * 2 - 1
        norm_y = (y / self.height) * 2 - 1
        return norm_x, norm_y
    
    def normalize_velocity(self, vx: float, vy: float) -> tuple[float, float]:
        """Normalize velocities relative to safe landing velocity"""
        norm_vx = vx / (self.safe_landing_velocity * 2)
        norm_vy = vy / (self.safe_landing_velocity * 2)
        return norm_vx, norm_vy

# Helper function to get scaled values
def get_scaling():
    """Get the current scaling instance"""
    return GameScaling.get_instance()