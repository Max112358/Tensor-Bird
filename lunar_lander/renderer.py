import pygame
from lander import Lander
from terrain import Terrain

class Renderer:
    def __init__(self, width: int, height: int):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Multi Lunar Lander")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        
    def render(self, landers: list[Lander], terrain: Terrain):
        self.screen.fill(self.BLACK)
        
        # Draw terrain
        pygame.draw.lines(self.screen, self.WHITE, False, terrain.points, 2)
        
        # Draw landing pad
        pad_left = terrain.landing_pad_x - terrain.landing_pad_width/2
        pygame.draw.line(
            self.screen, 
            self.RED,
            (pad_left, terrain.ground_height),
            (pad_left + terrain.landing_pad_width, terrain.ground_height),
            4
        )
        
        # Draw landers
        for lander in landers:
            # Draw main body
            pygame.draw.polygon(self.screen, self.WHITE, lander.get_vertices())
            
            # Draw legs
            left_leg, right_leg = lander.get_leg_positions()
            pygame.draw.line(self.screen, self.WHITE, *left_leg)
            pygame.draw.line(self.screen, self.WHITE, *right_leg)
        
        pygame.display.flip()
        
    def close(self):
        pygame.quit()