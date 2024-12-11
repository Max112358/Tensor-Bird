import pygame
import sys
import time
from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, GAME_TITLE,
    BASE_REWARD_PER_FRAME
)
from player_car import PlayerCar
from traffic_manager import TrafficManager
from game_visualizer import GameVisualizer

def main():
    """Main game loop for human players"""
    pygame.init()
    pygame.display.set_caption(GAME_TITLE)
    
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    visualizer = GameVisualizer(screen)
    
    def reset_game():
        """Reset the game state"""
        traffic_manager = TrafficManager(
            visualizer.road_top,
            visualizer.road_bottom
        )
        traffic_manager.spawn_initial_traffic()
        
        # Start player in middle lane
        player_y = (visualizer.road_top + visualizer.road_bottom) / 2
        player_car = PlayerCar(SCREEN_WIDTH * 0.2, player_y)
        
        return traffic_manager, player_car, 0  # 0 is initial score
    
    traffic_manager, player_car, score = reset_game()
    game_over = False
    death_time = 0
    DEATH_LOCKOUT = 1.5  # Time to wait before allowing restart
    
    # Main game loop
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # Convert to seconds
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and game_over:
                    if current_time - death_time >= DEATH_LOCKOUT:
                        traffic_manager, player_car, score = reset_game()
                        game_over = False
        
        if not game_over:
            # Handle continuous keyboard input
            keys = pygame.key.get_pressed()
            
            # Acceleration/Braking
            acceleration = 0
            if keys[pygame.K_UP]:
                acceleration = 1.0
            elif keys[pygame.K_DOWN]:
                acceleration = -1.0
            player_car.accelerate(acceleration)
            
            # Lane changes
            if keys[pygame.K_LEFT]:
                current_lane = traffic_manager._get_lane(player_car.y)
                if current_lane > 0:
                    target_y = traffic_manager._get_lane_y(current_lane - 1)
                    player_car.move_toward_y(target_y, player_car.width * 0.1)
                    
            elif keys[pygame.K_RIGHT]:
                current_lane = traffic_manager._get_lane(player_car.y)
                if current_lane < traffic_manager.num_lanes - 1:
                    target_y = traffic_manager._get_lane_y(current_lane + 1)
                    player_car.move_toward_y(target_y, player_car.width * 0.1)
            
            # Update game state
            player_car.update()
            traffic_manager.update(dt, player_car)
            
            # Check for collisions
            if traffic_manager.check_collision(player_car):
                game_over = True
                death_time = current_time
            else:
                # Update score based on speed
                speed_multiplier = player_car.velocity / player_car.MAX_VELOCITY
                score += BASE_REWARD_PER_FRAME * speed_multiplier * dt
        
        # Draw everything
        visualizer.draw_frame(
            player_car,
            traffic_manager,
            score,
            player_car.total_distance,
            game_over
        )
    
    pygame.quit()

if __name__ == "__main__":
    main()