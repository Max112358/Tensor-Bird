import math

def convert_degrees_to_custom_angle(degrees: float) -> float:
    """
    Convert degrees to custom angle setup:
    - 0° (up) = 0.0
    - 90° (right) = 0.5
    - 180° (down) = ±1.0
    - 270° (left) = -0.5
    """
    degrees = degrees % 360
    if degrees <= 90:  # Up to right (0° to 90°)
        return degrees / 180  # 0.0 to 0.5
    elif degrees <= 180:  # Right to down (90° to 180°)
        return 0.5 + (degrees - 90) / 90 * 0.5  # 0.5 to 1.0
    elif degrees <= 270:  # Down to left (180° to 270°)
        return (1 - (degrees - 180) / 180) * -1  # -0.5 to -1.0
    else:  # Left to up (270° to 360°)
        return -0.5 + (degrees - 270) / 90 * 0.5  # -0.5 to 0.0
