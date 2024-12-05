from rocket_physics import RocketPhysics, PhysicsConfig, PhysicsState
import numpy as np
import math
import terrain as Terrain
from constants import *

class Lander:
    def __init__(self, x: float, y: float, terrain: Terrain):
        # Initialize physics configuration with much larger force values
        physics_config = PhysicsConfig(
            mass=100.0,  # kg
            width=LANDER_WIDTH / PIXELS_PER_METER,  # Convert pixels to meters
            height=LANDER_HEIGHT / PIXELS_PER_METER,
            gravity=GRAVITY / PIXELS_PER_METER,  # Convert to m/s^2
            main_engine_force=MAIN_ENGINE_POWER,  # Remove pixel conversion for forces
            side_engine_force=SIDE_ENGINE_POWER,  # Remove pixel conversion for forces
            linear_drag=LINEAR_DAMPING,
            angular_drag=ANGULAR_DAMPING,
            dt=DT
        )
        
        # Initialize physics engine
        self.physics = RocketPhysics(physics_config)
        
        # Set initial position
        self.physics.state = PhysicsState(
            position=np.array([x / PIXELS_PER_METER, y / PIXELS_PER_METER]),
            velocity=np.zeros(2),
            angle=0.0,
            angular_velocity=0.0
        )
        
        # Landing gear state
        self.left_leg_contact = False
        self.right_leg_contact = False
        
        # Fuel system
        self.fuel = INITIAL_FUEL
        
        # Dimensions (kept in pixels for rendering)
        self.width = LANDER_WIDTH
        self.height = LANDER_HEIGHT
        self.leg_length = LEG_LENGTH
        
        # Active state
        self.active = True
        self.terminated = False
        self.terminate_reason = None
        
        #the level
        self.terrain = terrain
        
        # thruster state tracking
        self.thrusters = {
            'main': False,
            'left': False,
            'right': False
        }
        
    @property
    def x(self) -> float:
        """Get x position in pixels"""
        return self.physics.state.position[0] * PIXELS_PER_METER
        
    @property
    def y(self) -> float:
        """Get y position in pixels"""
        return self.physics.state.position[1] * PIXELS_PER_METER
        
    @property
    def angle(self) -> float:
        """Get angle in radians"""
        return self.physics.state.angle
        
    @property
    def velocity_x(self) -> float:
        """Get x velocity in pixels/sec"""
        return self.physics.state.velocity[0] * PIXELS_PER_METER
        
    @property
    def velocity_y(self) -> float:
        """Get y velocity in pixels/sec"""
        return self.physics.state.velocity[1] * PIXELS_PER_METER
        
    @property
    def angular_velocity(self) -> float:
        """Get angular velocity in radians/sec"""
        return self.physics.state.angular_velocity

    def get_vertices(self) -> list:
        # ... (rest of get_vertices remains the same)
        pass

    def get_leg_positions(self) -> tuple:
        # ... (rest of get_leg_positions remains the same)
        pass

    def terminate(self, reason: str):
        self.active = False
        self.terminated = True
        self.terminate_reason = reason
        
    def step(self, action: int) -> np.ndarray:
        """Update lander physics based on action
        Actions: 0=noop, 1=left engine, 2=main engine, 3=right engine"""
        
        if not self.active or self.fuel <= 0:
            return self.get_state()
        
        # Reset thruster states
        self.thrusters = {
            'main': False,
            'left': False,
            'right': False
        }
        
        # Update thrusters based on action
        if action == 1:  # Left engine
            self.thrusters['left'] = True
            self.fuel -= SIDE_ENGINE_FUEL_COST
        elif action == 2:  # Main engine
            self.thrusters['main'] = True
            self.fuel -= MAIN_ENGINE_FUEL_COST
        elif action == 3:  # Right engine
            self.thrusters['right'] = True
            self.fuel -= SIDE_ENGINE_FUEL_COST
            
        self.fuel = max(0, self.fuel)
        
        # Update physics
        self.physics.step(self.thrusters)
        
        return self.get_state()
        
    def get_color(self) -> tuple:
        """Return the appropriate color based on lander state"""
        if not self.active:
            return (0, 0, 255)  # Blue for dead/landed
        elif any(self.thrusters.values()):
            return (255, 0, 0)  # Red for any thruster firing
        else:
            return (255, 255, 255)  # White for alive but drifting
    
    def get_state(self) -> np.ndarray:
        """Return the normalized state vector"""
        landing_pad_center_x = self.terrain.landing_pad_x
        landing_pad_center_y = self.terrain.ground_height
        
        distance_to_pad_x = abs(self.x - landing_pad_center_x) / self.terrain.width
        distance_to_pad_y = abs(self.y - landing_pad_center_y) / self.terrain.height
        
        return np.array([
            self.x / POS_NORMALIZATION - 1,  # Normalized x position
            self.y / POS_NORMALIZATION - 1,  # Normalized y position
            self.velocity_x / VEL_NORMALIZATION,  # Normalized x velocity
            self.velocity_y / VEL_NORMALIZATION,  # Normalized y velocity
            self.angle / ANGLE_NORMALIZATION,  # Normalized angle
            self.angular_velocity,  # Angular velocity
            distance_to_pad_x,  # Normalized distance to landing pad center (x)
            distance_to_pad_y,  # Normalized distance to landing pad center (y)
            self.fuel / INITIAL_FUEL  # Normalized fuel level
        ])
        
        
    def get_vertices(self) -> list:
        """Get vertices for rendering, with rotation applied"""
        # Define vertices relative to center (in pixels)
        half_width = self.width / 2
        half_height = self.height / 2
        vertices = [
            (-half_width, -half_height),  # Top left
            (half_width, -half_height),   # Top right
            (half_width, half_height),    # Bottom right
            (-half_width, half_height)    # Bottom left
        ]
        
        # Apply rotation and translation
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        
        rotated_vertices = []
        for vx, vy in vertices:
            # Rotate point
            rotated_x = vx * cos_angle - vy * sin_angle
            rotated_y = vx * sin_angle + vy * cos_angle
            
            # Translate to lander position (explicitly convert to integers for pygame)
            final_x = int(rotated_x + self.x)
            final_y = int(rotated_y + self.y)
            
            rotated_vertices.append((final_x, final_y))
        
        return rotated_vertices

    def get_leg_positions(self) -> tuple:
        """Get leg endpoints for rendering, with rotation applied"""
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
        
        # Left leg
        left_base_x = -self.width/2
        left_base_y = self.height/2
        
        # Rotate base point
        left_rotated_x = left_base_x * cos_angle - left_base_y * sin_angle
        left_rotated_y = left_base_x * sin_angle + left_base_y * cos_angle
        
        # Translate to lander position (convert to integers)
        left_start = (int(left_rotated_x + self.x), int(left_rotated_y + self.y))
        
        # Calculate leg end point with angle
        left_end_x = left_base_x - self.leg_length * 0.7  # Angle the legs outward
        left_end_y = left_base_y + self.leg_length
        
        # Rotate and translate end point
        left_end_rotated_x = left_end_x * cos_angle - left_end_y * sin_angle
        left_end_rotated_y = left_end_x * sin_angle + left_end_y * cos_angle
        left_end = (int(left_end_rotated_x + self.x), int(left_end_rotated_y + self.y))
        
        # Right leg (mirror of left leg)
        right_base_x = self.width/2
        right_base_y = self.height/2
        
        # Rotate base point
        right_rotated_x = right_base_x * cos_angle - right_base_y * sin_angle
        right_rotated_y = right_base_x * sin_angle + right_base_y * cos_angle
        
        # Translate to lander position (convert to integers)
        right_start = (int(right_rotated_x + self.x), int(right_rotated_y + self.y))
        
        # Calculate leg end point with angle
        right_end_x = right_base_x + self.leg_length * 0.7  # Angle the legs outward
        right_end_y = right_base_y + self.leg_length
        
        # Rotate and translate end point
        right_end_rotated_x = right_end_x * cos_angle - right_end_y * sin_angle
        right_end_rotated_y = right_end_x * sin_angle + right_end_y * cos_angle
        right_end = (int(right_end_rotated_x + self.x), int(right_end_rotated_y + self.y))
        
        return ((left_start, left_end), (right_start, right_end))