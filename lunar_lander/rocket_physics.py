import numpy as np
from dataclasses import dataclass
from typing import NamedTuple
import math

@dataclass
class PhysicsConfig:
    """Configuration parameters for rocket physics"""
    mass: float  # kg
    width: float  # meters
    height: float  # meters
    gravity: float  # m/s^2
    main_engine_force: float  # Newtons
    side_engine_force: float  # Newtons
    linear_drag: float  # Drag coefficient
    angular_drag: float  # Angular drag coefficient
    dt: float  # Physics timestep

class Force(NamedTuple):
    """Force vector with application point"""
    vector: np.ndarray  # Force vector (x, y)
    position: np.ndarray  # Application point relative to COM (x, y)

class PhysicsState(NamedTuple):
    """Complete physics state of the rocket"""
    position: np.ndarray  # (x, y) in world space
    velocity: np.ndarray  # (vx, vy) in world space
    angle: float  # radians
    angular_velocity: float  # radians/sec

class RocketPhysics:
    def __init__(self, config: PhysicsConfig):
        self.config = config
        
        # Calculate moment of inertia for a rectangular body
        self.moment_of_inertia = (self.config.mass / 12.0) * (
            self.config.width ** 2 + self.config.height ** 2
        )
        
        # Define thruster positions relative to center of mass
        half_width = self.config.width / 2
        half_height = self.config.height / 2
        
        self.thruster_positions = {
            'main': np.array([0.0, half_height]),
            'left': np.array([-half_width, 0.0]),
            'right': np.array([half_width, 0.0])
        }
        
        # Initialize state
        self.state = PhysicsState(
            position=np.zeros(2),
            velocity=np.zeros(2),
            angle=0.0,
            angular_velocity=0.0
        )
    
    def _to_global_coords(self, local_vector: np.ndarray) -> np.ndarray:
        """Convert a vector from local to global coordinates"""
        cos_angle = math.cos(self.state.angle)
        sin_angle = math.sin(self.state.angle)
        rotation = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        return rotation @ local_vector
    
    def _apply_force(self, force: Force) -> tuple[np.ndarray, float]:
        """Apply a force at a point, returns (linear_acceleration, torque)"""
        # Convert force and position to global coordinates
        force_global = self._to_global_coords(force.vector)
        pos_global = self._to_global_coords(force.position)
        
        # Calculate linear acceleration
        linear_acc = force_global / self.config.mass
        
        # Calculate torque (2D cross product)
        torque = pos_global[0] * force_global[1] - pos_global[1] * force_global[0]
        
        return linear_acc, torque
    
    def step(self, thrusters: dict[str, bool]) -> PhysicsState:
        """Update physics state based on active thrusters
        
        Args:
            thrusters: Dict of thruster states ('main', 'left', 'right')
        """
        # Start with acceleration from gravity
        total_acceleration = np.array([0.0, self.config.gravity])
        total_torque = 0.0
        
        # Apply thruster forces
        if thrusters.get('main'):
            force = Force(
                vector=np.array([0.0, -self.config.main_engine_force]),
                position=self.thruster_positions['main']
            )
            acc, torque = self._apply_force(force)
            total_acceleration += acc
            total_torque += torque
            
        if thrusters.get('left'):
            force = Force(
                vector=np.array([0.0, self.config.side_engine_force]),
                position=self.thruster_positions['left']
            )
            acc, torque = self._apply_force(force)
            total_acceleration += acc
            total_torque += torque
            
        if thrusters.get('right'):
            force = Force(
                vector=np.array([0.0, self.config.side_engine_force]),
                position=self.thruster_positions['right']
            )
            acc, torque = self._apply_force(force)
            total_acceleration += acc
            total_torque += torque
        
        # Apply drag
        linear_damping = math.exp(-self.config.linear_drag * self.config.dt)
        angular_damping = math.exp(-self.config.angular_drag * self.config.dt)
        
        # Update velocities
        new_velocity = (
            self.state.velocity * linear_damping + 
            total_acceleration * self.config.dt
        )
        
        new_angular_velocity = (
            self.state.angular_velocity * angular_damping +
            (total_torque / self.moment_of_inertia) * self.config.dt
        )
        
        # Update positions
        new_position = (
            self.state.position + 
            self.state.velocity * self.config.dt +
            0.5 * total_acceleration * self.config.dt ** 2
        )
        
        new_angle = (
            self.state.angle + 
            self.state.angular_velocity * self.config.dt +
            0.5 * (total_torque / self.moment_of_inertia) * self.config.dt ** 2
        )
        
        # Normalize angle to [-π, π]
        new_angle = math.atan2(math.sin(new_angle), math.cos(new_angle))
        
        # Update state
        self.state = PhysicsState(
            position=new_position,
            velocity=new_velocity,
            angle=new_angle,
            angular_velocity=new_angular_velocity
        )
        
        return self.state