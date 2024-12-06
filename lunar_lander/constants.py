# Screen/window constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

PIXELS_PER_METER = 20  # Arbitrary scale factor


# Terrain constants
GROUND_HEIGHT = 550  # Height of the ground from top of screen
LANDING_PAD_WIDTH = 100
TERRAIN_ROUGHNESS = 20  # Max height variation in terrain

# Training constants
MAX_STEPS_PER_EPISODE = 2000
MIN_STEPS_BEFORE_DONE = 200

# Rewards
LANDING_REWARD = 100.0
CRASH_PENALTY = -100.0
OUT_OF_BOUNDS_PENALTY = -100.0
OUT_OF_FUEL_PENALTY = -100.0

# Safe landing parameters
SAFE_LANDING_VELOCITY = 80.0  # Maximum safe landing velocity
SAFE_LANDING_ANGLE = 15.0  # Maximum safe landing velocity
LANDING_PAD_TOLERANCE = 5    # How close to pad height counts as landing

# Physics constants 
GRAVITY = 40.0  # Normal gravity in m/s^2
MAIN_ENGINE_POWER = 1500.0  # Main thruster force in Newtons
SIDE_ENGINE_POWER = 250.0   # Side thruster force in Newtons
ANGULAR_DAMPING = 0.5    # Increased rotational drag for better control
LINEAR_DAMPING = 0.1     # Increased linear drag for better control

# Physics timestep
FPS = 60
DT = 1.0 / FPS

# Lander constants
LANDER_WIDTH = 20
LANDER_HEIGHT = 30
LEG_LENGTH = 10
INITIAL_FUEL = 200.0
MAIN_ENGINE_FUEL_COST = 1.0
SIDE_ENGINE_FUEL_COST = 0.5

# Normalization factors for AI
POS_NORMALIZATION = 400.0
VEL_NORMALIZATION = 10.0
ANGLE_NORMALIZATION = 3.14159