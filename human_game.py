# human_game.py
import pygame
import sys
from bird import Bird
from pipe import Pipe
from background import Background
from game_utils import check_collision
from death_marker import DeathMarker
from constants import *
import time

def main():
    pygame.init()
    pygame.display.set_caption(GAME_TITLE)

    clock = pygame.time.Clock()
    
    def reset_game():
        bird = Bird(SCREEN_WIDTH * 0.2, SCREEN_HEIGHT * 0.4)
        background = Background(SCREEN_WIDTH, SCREEN_HEIGHT)
        first_pipe_x = SCREEN_WIDTH * 0.75
        pipes = [Pipe(first_pipe_x + i * PIPE_SPACING) for i in range(VISIBLE_PIPES)]
        return bird, background, pipes, [], 0  # Added empty death_markers list
    
    bird, background, pipes, death_markers, score = reset_game()
    game_over = False
    death_time = 0
    DEATH_LOCKOUT = 1.5
    
    score_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.091))
    game_over_font = pygame.font.Font(None, int(SCREEN_HEIGHT * 0.067))
    debug_font = pygame.font.Font(None, 36)
    game_over_text = game_over_font.render('Game Over! Press SPACE to restart', True, (255, 255, 255))
    game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))

    while True:
        clock.tick(FPS)
        current_time = time.time()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if game_over:
                        if current_time - death_time >= DEATH_LOCKOUT:
                            bird, background, pipes, death_markers, score = reset_game()
                            game_over = False
                    else:
                        bird.jump()

        if not game_over:
            bird.move()
            background.move()
            
            # Move and clean up death markers
            for marker in death_markers[:]:
                marker.move()
                if marker.is_offscreen():
                    death_markers.remove(marker)
            
            for pipe in pipes:
                pipe.move()
                
                collision, death_pos = check_collision(bird, pipe)
                if collision:
                    #print(f"Game Over! Death position: {death_pos}")
                    #print(f"Bird position: ({bird.x}, {bird.y})")
                    #print(f"Pipe position: x={pipe.x}, top_rect={pipe.top_rect}, bottom_rect={pipe.bottom_rect}")
                    if death_pos:
                        death_markers.append(DeathMarker(*death_pos))
                    game_over = True
                    death_time = current_time
                    break
                
                elif not pipe.passed and bird.x > pipe.x + PIPE_WIDTH:
                    pipe.passed = True
                    score += 1
            
            while len(pipes) > 0 and pipes[0].x < -PIPE_WIDTH:
                pipes.pop(0)
                pipes.append(Pipe(pipes[-1].x + PIPE_SPACING))

        # Draw everything
        SCREEN.fill(SKY_BLUE)
        background.draw(SCREEN)
        
        for pipe in pipes:
            pipe.draw(SCREEN)
            
        bird.draw(SCREEN)
        
        # Draw death markers after bird so they appear on top
        for marker in death_markers:
            marker.draw(SCREEN)
        
        # Draw score
        score_text = score_font.render(str(score), True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT * 0.091))
        SCREEN.blit(score_text, score_rect)
        
        '''
        # Draw debug info
        if not game_over:
            debug_text = debug_font.render(f"Bird: ({int(bird.x)}, {int(bird.y)}) Vel: {bird.velocity:.1f}", True, (255, 255, 255))
            SCREEN.blit(debug_text, (10, 10))
        '''
            
        if game_over:
            if current_time - death_time < DEATH_LOCKOUT:
                lockout_text = game_over_font.render('Game Over!', True, (255, 255, 255))
            else:
                lockout_text = game_over_text
            SCREEN.blit(lockout_text, game_over_rect)
            
        pygame.display.flip()

if __name__ == "__main__":
    main()