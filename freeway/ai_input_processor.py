import numpy as np
from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, NUM_LANES,
    LANE_WIDTH, MAX_VELOCITY, MAX_VISION_DISTANCE,
    NUM_CARS_VISIBLE_AHEAD, DISTANCE_BUCKETS
)

def get_lane_info(y_position, road_top, road_bottom):
    """
    Get normalized lane position and distances to lane edges.
    
    Args:
        y_position (float): Current y coordinate
        road_top (float): Y coordinate of road top
        road_bottom (float): Y coordinate of road bottom
        
    Returns:
        tuple: (normalized_lane_position, normalized_dist_to_top, normalized_dist_to_bottom)
    """
    # Normalize y position relative to road height
    road_height = road_bottom - road_top
    normalized_y = (y_position - road_top) / road_height
    
    # Calculate distances to lane edges
    current_lane = int(normalized_y * NUM_LANES)
    lane_top = road_top + (current_lane * LANE_WIDTH)
    lane_bottom = lane_top + LANE_WIDTH
    
    dist_to_top = (y_position - lane_top) / LANE_WIDTH
    dist_to_bottom = (lane_bottom - y_position) / LANE_WIDTH
    
    return normalized_y, dist_to_top, dist_to_bottom

def get_car_inputs(player_car, traffic_manager):
    """
    Process all inputs for AI decision making.
    
    Args:
        player_car: PlayerCar instance
        traffic_manager: TrafficManager instance
        
    Returns:
        list: Normalized inputs for neural network
    """
    inputs = []
    
    # 1. Player state inputs
    # Normalize velocity
    inputs.append(player_car.velocity / MAX_VELOCITY)
    
    # Normalize position
    inputs.append(player_car.x / SCREEN_WIDTH)
    lane_pos, dist_top, dist_bottom = get_lane_info(
        player_car.y, 
        traffic_manager.road_top, 
        traffic_manager.road_bottom
    )
    inputs.extend([lane_pos, dist_top, dist_bottom])
    
    # 2. Traffic information
    # Create distance buckets for each lane
    lane_buckets = np.zeros((NUM_LANES, DISTANCE_BUCKETS))
    lane_speeds = np.zeros(NUM_LANES)  # Track average speed per lane
    cars_in_lane = np.zeros(NUM_LANES)  # Count cars per lane
    
    # Get nearby cars
    nearby_cars = traffic_manager.get_nearby_cars(
        player_car.x, 
        player_car.y, 
        MAX_VISION_DISTANCE
    )
    
    for car in nearby_cars:
        # Get car's lane
        car_lane_pos, _, _ = get_lane_info(
            car.y,
            traffic_manager.road_top,
            traffic_manager.road_bottom
        )
        lane = int(car_lane_pos * NUM_LANES)
        if not 0 <= lane < NUM_LANES:
            continue
            
        # Calculate relative position and velocity
        rel_x = (car.x - player_car.x) / MAX_VISION_DISTANCE
        rel_speed = (car.velocity - player_car.velocity) / MAX_VELOCITY
        
        # Only process cars ahead of player
        if rel_x > 0:
            # Find appropriate distance bucket
            bucket = int(rel_x * DISTANCE_BUCKETS)
            if bucket < DISTANCE_BUCKETS:
                lane_buckets[lane][bucket] = 1
                lane_speeds[lane] += rel_speed
                cars_in_lane[lane] += 1
    
    # Normalize lane speeds by number of cars
    for lane in range(NUM_LANES):
        if cars_in_lane[lane] > 0:
            lane_speeds[lane] /= cars_in_lane[lane]
    
    # Add lane occupancy information
    inputs.extend(lane_buckets.flatten())
    
    # Add normalized lane speeds
    inputs.extend(lane_speeds)
    
    # 3. Immediate proximity info (emergency inputs)
    for lane in range(NUM_LANES):
        # Check car directly ahead in each lane
        car = traffic_manager.get_closest_car_in_lane(lane, player_car.x, ahead=True)
        if car:
            # Normalize distance and relative speed
            distance = (car.x - player_car.x) / MAX_VISION_DISTANCE
            rel_speed = (car.velocity - player_car.velocity) / MAX_VELOCITY
            inputs.extend([distance, rel_speed])
        else:
            inputs.extend([1.0, 0.0])  # No car ahead
            
        # Check car directly behind in each lane
        car = traffic_manager.get_closest_car_in_lane(lane, player_car.x, ahead=False)
        if car:
            # Normalize distance and relative speed
            distance = (player_car.x - car.x) / MAX_VISION_DISTANCE
            rel_speed = (car.velocity - player_car.velocity) / MAX_VELOCITY
            inputs.extend([distance, rel_speed])
        else:
            inputs.extend([1.0, 0.0])  # No car behind
    
    return inputs

def get_input_size():
    """Calculate total number of inputs"""
    size = (
        5 +  # Player state (velocity, x, lane_pos, dist_top, dist_bottom)
        (NUM_LANES * DISTANCE_BUCKETS) +  # Lane occupancy buckets
        NUM_LANES +  # Lane speeds
        (NUM_LANES * 4)  # Immediate proximity info (2 values for ahead, 2 for behind)
    )
    return size

def get_output_size():
    """Get number of outputs needed"""
    return 2  # [acceleration, lane_change]