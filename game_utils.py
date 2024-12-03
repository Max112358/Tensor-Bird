# game_utils.py
import pygame
from constants import BIRD_SIZE, FLOOR_Y, SCREEN_WIDTH, SCALE

def check_collision(bird, pipe):
    bird_rect = pygame.Rect(bird.x, bird.y, BIRD_SIZE, BIRD_SIZE)
    
    pipe_collision = (bird_rect.colliderect(pipe.top_rect) or 
                     bird_rect.colliderect(pipe.bottom_rect))
    
    ceiling_collision = bird.y < 0
    floor_collision = bird_rect.bottom > FLOOR_Y
    
    if pipe_collision or ceiling_collision or floor_collision:
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
    
    # Scale font size with screen
    font = pygame.font.Font(None, int(100 * SCALE))
    score_text = font.render(str(score), True, (255, 255, 255))
    screen.blit(score_text, (SCREEN_WIDTH/2 - score_text.get_width()/2, 100 * SCALE))
    
    pygame.display.update()