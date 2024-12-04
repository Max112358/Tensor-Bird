# game_utils.py
import pygame
from constants import BIRD_SIZE, FLOOR_Y, SCREEN_WIDTH

def check_collision(bird, pipe):
    bird_rect = pygame.Rect(bird.x, bird.y, BIRD_SIZE, BIRD_SIZE)
    
    # Check pipe collisions first
    pipe_collision = (bird_rect.colliderect(pipe.top_rect) or 
                     bird_rect.colliderect(pipe.bottom_rect))
    
    # Check ceiling and floor collisions
    # Use bird_rect.bottom to ensure bird dies when its bottom edge hits floor
    ceiling_collision = bird.y < 0
    floor_collision = bird_rect.bottom > FLOOR_Y
    
    if pipe_collision or ceiling_collision or floor_collision:
        # Calculate center position of bird for death marker
        center_x = bird.x + BIRD_SIZE // 2
        center_y = bird.y + BIRD_SIZE // 2
        return True, (center_x, center_y)
    return False, None

def draw_game(screen, background, pipes, birds, score, death_markers=None):
    background.draw(screen)
    
    for pipe in pipes:
        pipe.draw(screen)
        
    if death_markers:
        for marker in death_markers:
            marker.draw(screen)
            
    for bird in birds:
        bird.draw(screen)
        
    font = pygame.font.Font(None, 100)
    score_text = font.render(str(score), True, (255, 255, 255))
    screen.blit(score_text, (SCREEN_WIDTH/2 - score_text.get_width()/2, 100))
    
    pygame.display.update()