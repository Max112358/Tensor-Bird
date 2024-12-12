from rocket_physics import RocketPhysics, PhysicsConfig, PhysicsState
import numpy as np
import math
import terrain as Terrain
from game_init import get_constants

class Lander:
    def __init__(self, x: float, y: float, terrain: Terrain):
        # Get global constants
        const = get_constants()
        
        # Initialize physics configuration
        physics_config = PhysicsConfig(
            mass=100.0,  # kg
            width=const.LANDER_WIDTH / const.PIXELS_PER_METER,  # Convert pixels to meters
            height=const.LANDER_HEIGHT / const.PIXELS_PER_METER,
            gravity=const.GRAVITY / const.PIXELS_PER_METER,  # Convert to m/s^2
            main_engine_force=const.MAIN_ENGINE_POWER,
            side_engine_force=const.SIDE_ENGINE_POWER,
            linear_drag=const.LINEAR_DAMPING,
            angular_drag=const.ANGULAR_DAMPING,
            dt=const.DT
        )
        
        # Initialize physics engine
        self.physics = RocketPhysics(physics_config)
        
        # Set initial position
        self.physics.state = PhysicsState(
            position=np.array([x / const.PIXELS_PER_METER, y / const.PIXELS_PER_METER]),
            velocity=np.zeros(2),
            angle=0.0,
            angular_velocity=0.0
        )
        
        # Landing gear state
        self.left_leg_contact = False
        self.right_leg_contact = False
        
        # Fuel system
        self.fuel = const.INITIAL_FUEL
        
        # Dimensions (kept in pixels for rendering)
        self.width = const.LANDER_WIDTH
        self.height = const.LANDER_HEIGHT
        self.leg_length = const.LEG_LENGTH
        
        # Active state
        self.active = True
        self.terminated = False
        self.terminate_reason = None
        
        # The level
        self.terrain = terrain
        
        # Thruster state tracking
        self.thrusters = {
            'main': False,
            'left': False,
            'right': False
        }
        
    @property
    def x(self) -> float:
        """Get x position in pixels"""
        const = get_constants()
        return self.physics.state.position[0] * const.PIXELS_PER_METER
        
    @property
    def y(self) -> float:
        """Get y position in pixels"""
        const = get_constants()
        return self.physics.state.position[1] * const.PIXELS_PER_METER
        
    @property
    def angle(self) -> float:
        """Get angle in radians"""
        return self.physics.state.angle
        
    @property
    def velocity_x(self) -> float:
        """Get x velocity in pixels/sec"""
        const = get_constants()
        return self.physics.state.velocity[0] * const.PIXELS_PER_METER
        
    @property
    def velocity_y(self) -> float:
        """Get y velocity in pixels/sec"""
        const = get_constants()
        return self.physics.state.velocity[1] * const.PIXELS_PER_METER
        
    @property
    def angular_velocity(self) -> float:
        """Get angular velocity in radians/sec"""
        return self.physics.state.angular_velocity

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
            
            # Translate to lander position
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
        
        # Translate to lander position
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
        
        # Translate to lander position
        right_start = (int(right_rotated_x + self.x), int(right_rotated_y + self.y))
        
        # Calculate leg end point with angle
        right_end_x = right_base_x + self.leg_length * 0.7  # Angle the legs outward
        right_end_y = right_base_y + self.leg_length
        
        # Rotate and translate end point
        right_end_rotated_x = right_end_x * cos_angle - right_end_y * sin_angle
        right_end_rotated_y = right_end_x * sin_angle + right_end_y * cos_angle
        right_end = (int(right_end_rotated_x + self.x), int(right_end_rotated_y + self.y))
        
        return ((left_start, left_end), (right_start, right_end))

    def terminate(self, reason: str):
        """Set lander to terminated state"""
        self.active = False
        self.terminated = True
        self.terminate_reason = reason
        
    def step(self, action: int) -> np.ndarray:
        """Update lander physics based on action"""
        const = get_constants()
        
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
            self.fuel -= const.SIDE_ENGINE_FUEL_COST
        elif action == 2:  # Main engine
            self.thrusters['main'] = True
            self.fuel -= const.MAIN_ENGINE_FUEL_COST
        elif action == 3:  # Right engine
            self.thrusters['right'] = True
            self.fuel -= const.SIDE_ENGINE_FUEL_COST
            
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
        """Return the normalized state vector with updated normalization logic"""
        const = get_constants()
        
        landing_pad_center_x = self.terrain.landing_pad_x
        landing_pad_center_y = self.terrain.ground_height
        
        # Calculate normalized distances to landing pad (already properly normalized)
        distance_to_pad_x = abs(self.x - landing_pad_center_x) / self.terrain.width
        distance_to_pad_y = abs(self.y - landing_pad_center_y) / self.terrain.height
        
        # Leave x velocity untouched
        raw_vel_x = self.velocity_x
        
        # Normalize y velocity relative to safe landing velocity
        norm_vel_y = self.velocity_y / const.SAFE_LANDING_VELOCITY
        
        # Normalize angle relative to safe landing angle (convert from radians to degrees first)
        angle_degrees = math.degrees(self.angle)
        norm_angle = angle_degrees / const.SAFE_LANDING_ANGLE
        
        # Angular velocity remains unchanged
        angular_vel = self.angular_velocity
        
        return np.array([
            raw_vel_x,  # x velocity (unchanged)
            norm_vel_y,  # y velocity (relative to safe landing velocity)
            norm_angle,  # angle (relative to safe landing angle)
            angular_vel,  # angular velocity (unchanged)
            distance_to_pad_x,  # normalized x distance to pad [0, 1]
            distance_to_pad_y,  # normalized y distance to pad [0, 1]
        ])