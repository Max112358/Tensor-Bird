import pygame
from constants import (
    SCREEN_WIDTH, SCREEN_HEIGHT, NUM_LANES,
    ROAD_COLOR, LANE_COLOR, SHOULDER_COLOR,
    DEBUG_COLOR, DEBUG_FONT_SIZE, SHOW_DEBUG_INFO,
    LANE_WIDTH, SHOULDER_WIDTH, MAX_VELOCITY, FPS
)

class GameVisualizer:
    def __init__(self, screen):
        """Initialize the game visualizer"""
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.current_fps = 0
        
        # Set up fonts
        self.score_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.1))
        self.debug_font = pygame.font.Font(None, DEBUG_FONT_SIZE)
        self.info_font = pygame.font.Font(None, int(DEBUG_FONT_SIZE * 0.8))
        
        # Pre-calculate road dimensions
        self.road_top = SHOULDER_WIDTH
        self.road_bottom = SCREEN_HEIGHT - SHOULDER_WIDTH
        self.road_height = self.road_bottom - self.road_top
        
        # Initialize lane markers
        self.lane_marker_length = 50
        self.lane_marker_gap = 30
        self.lane_marker_width = 3
        self.lane_marker_speed = 2
        self.lane_offset = 0
        
        # Add visual elements for better lane distinction
        self.lane_colors = [
            (max(35, min(45 + i * 2, 55)), 
             max(35, min(45 + i * 2, 55)), 
             max(35, min(45 + i * 2, 55))) 
            for i in range(NUM_LANES)
        ]
        
        # Create lane labels
        self.lane_labels = []
        for i in range(NUM_LANES):
            label = self.debug_font.render(f"Lane {i+1}", True, LANE_COLOR)
            self.lane_labels.append(label)
        
        # Current state
        self.is_paused = False
        
    def _draw_road(self):
        """Draw the road, shoulders, and lane markers"""
        # Clear screen and draw shoulders
        self.screen.fill(SHOULDER_COLOR)
        
        # Draw each lane with slightly different shading
        lane_height = (self.road_bottom - self.road_top) / NUM_LANES
        for lane in range(NUM_LANES):
            lane_rect = pygame.Rect(
                0,
                self.road_top + (lane * lane_height),
                SCREEN_WIDTH,
                lane_height
            )
            pygame.draw.rect(self.screen, self.lane_colors[lane], lane_rect)
            
            # Draw lane label
            label = self.lane_labels[lane]
            label_rect = label.get_rect(
                left=10,
                centery=self.road_top + (lane * lane_height) + (lane_height / 2)
            )
            pygame.draw.rect(self.screen, (0, 0, 0, 128), label_rect)
            self.screen.blit(label, label_rect)
        
        # Draw dashed lane markers
        for lane in range(1, NUM_LANES):
            y = self.road_top + (lane * lane_height)
            marker_x = -self.lane_offset
            while marker_x < SCREEN_WIDTH:
                pygame.draw.line(
                    self.screen,
                    LANE_COLOR,
                    (marker_x, y),
                    (marker_x + self.lane_marker_length, y),
                    self.lane_marker_width
                )
                marker_x += self.lane_marker_length + self.lane_marker_gap
        
        # Draw solid edge lines
        edge_width = 4
        pygame.draw.line(self.screen, LANE_COLOR, (0, self.road_top), 
                        (SCREEN_WIDTH, self.road_top), edge_width)
        pygame.draw.line(self.screen, LANE_COLOR, (0, self.road_bottom), 
                        (SCREEN_WIDTH, self.road_bottom), edge_width)
        
        # Update animation offset
        self.lane_offset = (self.lane_offset + self.lane_marker_speed) % (
            self.lane_marker_length + self.lane_marker_gap)
        
    def _draw_score(self, score, distance):
        """Draw score and distance information"""
        score_text = f"Score: {int(score)}"
        score_surface = self.score_font.render(score_text, True, (255, 255, 255))
        score_rect = score_surface.get_rect(topleft=(20, 20))
        
        bg_rect = score_rect.copy()
        bg_rect.inflate_ip(20, 10)
        pygame.draw.rect(self.screen, (0, 0, 0, 128), bg_rect)
        
        self.screen.blit(score_surface, score_rect)
        
        distance_text = f"Distance: {int(distance)}m"
        distance_surface = self.debug_font.render(distance_text, True, (200, 200, 200))
        distance_rect = distance_surface.get_rect(
            topleft=(20, score_rect.bottom + 10)
        )
        self.screen.blit(distance_surface, distance_rect)
        
    def update_fps(self):
        """Update FPS counter and return delta time"""
        return self.clock.tick(FPS)
        
    def _draw_debug_info(self, cars, traffic_manager):
        """Draw debug information for multiple cars"""
        if not SHOW_DEBUG_INFO:
            return
            
        # Calculate actual FPS
        fps = round(1000 / max(1, self.current_fps))
        
        # Count active cars
        active_cars = [car for car in cars if car.is_active]
        
        # Get best performing car
        best_car = None
        if active_cars:
            best_car = max(active_cars, key=lambda car: car.genome.fitness)
        
        debug_info = [
            f"FPS: {fps}",
            f"Cars Alive: {len(active_cars)}/{len(cars)}",
            f"NPC Cars: {len(traffic_manager.cars)}",
        ]
        
        if best_car:
            debug_info.extend([
                f"Best Speed: {best_car.velocity:.1f}",
                f"Best Distance: {best_car.total_distance:.1f}",
                f"Best Fitness: {best_car.genome.fitness:.1f}"
            ])
        
        # Calculate maximum text width
        max_width = 0
        rendered_texts = []
        for info in debug_info:
            text_surface = self.debug_font.render(info, True, DEBUG_COLOR)
            rendered_texts.append(text_surface)
            max_width = max(max_width, text_surface.get_width())
        
        # Draw background
        padding = 10
        total_height = len(debug_info) * 25
        background_rect = pygame.Rect(
            SCREEN_WIDTH - max_width - padding * 2,
            padding,
            max_width + padding * 2,
            total_height + padding
        )
        pygame.draw.rect(self.screen, (0, 0, 0, 128), background_rect)
        
        # Draw debug info
        y = padding * 1.5
        for text_surface in rendered_texts:
            text_rect = text_surface.get_rect(
                right=SCREEN_WIDTH - padding,
                top=y
            )
            self.screen.blit(text_surface, text_rect)
            y += 25
            
    def _draw_speed_indicator(self, cars):
        """Draw speedometer for best performing car"""
        active_cars = [car for car in cars if car.is_active]
        if not active_cars:
            return
            
        # Get best performing car
        best_car = max(active_cars, key=lambda car: car.genome.fitness)
        
        # Draw speedometer background
        speed_rect = pygame.Rect(10, SCREEN_HEIGHT-60, 200, 50)
        pygame.draw.rect(self.screen, (40, 40, 40), speed_rect)
        
        # Draw speed bar
        speed_ratio = best_car.velocity / MAX_VELOCITY
        bar_width = 190 * speed_ratio
        bar_color = self._get_speed_color(speed_ratio)
        speed_bar_rect = pygame.Rect(15, SCREEN_HEIGHT-55, bar_width, 40)
        pygame.draw.rect(self.screen, bar_color, speed_bar_rect)
        
        # Draw speed text
        speed_text = f"{int(best_car.velocity * 3.6)} km/h"
        speed_surface = self.info_font.render(speed_text, True, (255, 255, 255))
        self.screen.blit(speed_surface, (220, SCREEN_HEIGHT-45))
        
    def _get_speed_color(self, speed_ratio):
        """Get color for speed indicator"""
        if speed_ratio < 0.5:
            return (0, 255, 0)  # Green
        elif speed_ratio < 0.8:
            return (255, 255, 0)  # Yellow
        else:
            return (255, 0, 0)  # Red
            
    def draw_game_over(self, final_score, final_distance):
        """Draw game over screen"""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.fill((0, 0, 0))
        overlay.set_alpha(128)
        self.screen.blit(overlay, (0, 0))
        
        game_over_text = self.score_font.render("GAME OVER", True, (255, 0, 0))
        score_text = self.debug_font.render(
            f"Final Score: {int(final_score)}", True, (255, 255, 255))
        distance_text = self.debug_font.render(
            f"Distance: {int(final_distance)}m", True, (255, 255, 255))
        
        # Center all text
        screen_center_x = SCREEN_WIDTH // 2
        screen_center_y = SCREEN_HEIGHT // 2
        
        self.screen.blit(game_over_text, 
            game_over_text.get_rect(center=(screen_center_x, screen_center_y - 60)))
        self.screen.blit(score_text, 
            score_text.get_rect(center=(screen_center_x, screen_center_y)))
        self.screen.blit(distance_text, 
            distance_text.get_rect(center=(screen_center_x, screen_center_y + 40)))
        
    def draw_frame(self, cars, traffic_manager, score, distance, game_over=False):
        """Draw a complete frame of the game"""
        self._draw_road()
        traffic_manager.draw(self.screen)
        
        # Draw all active cars
        for car in cars:
            if car.is_active:
                car.draw(self.screen)
        
        self._draw_score(score, distance)
        self._draw_speed_indicator(cars)
        self._draw_debug_info(cars, traffic_manager)
        
        if game_over:
            self.draw_game_over(score, distance)
        
        pygame.display.flip()