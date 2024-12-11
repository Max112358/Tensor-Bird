import pygame
from lander import Lander
from terrain import Terrain

class Renderer:
    def __init__(self, width: int, height: int):
        pygame.init()
        
        # Set window flags for better background behavior
        flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SHOWN
        
        # Create window with flags
        self.screen = pygame.display.set_mode((width, height), flags)
        pygame.display.set_caption("Multi Lunar Lander")
        
        # Initialize font for lander numbers
        self.number_font = pygame.font.Font(None, 36)  # Added font for numbers
        
        # Configure window for background rendering while maintaining taskbar visibility
        info = pygame.display.get_wm_info()
        if 'window' in info:  # Check if we can get the window handle
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):  # Windows specific
                    hwnd = info['window']
                    # Set window style to allow background rendering while keeping taskbar visibility
                    GWL_EXSTYLE = -20
                    WS_EX_APPWINDOW = 0x40000
                    WS_EX_COMPOSITED = 0x02000000
                    # Combine flags to get both behaviors
                    style = WS_EX_APPWINDOW | WS_EX_COMPOSITED
                    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            except Exception:
                pass  # Fail silently if we can't set window styles
        
        # Track window states
        self.has_focus = True
        self.running = True
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        
    def render(self, landers: list[Lander], terrain: Terrain) -> bool:
        """Returns False if the window should close, True otherwise"""
        # Handle window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.ACTIVEEVENT and event.state == 2:
                self.has_focus = event.gain
            elif event.type in (pygame.VIDEOEXPOSE, pygame.WINDOWENTER):
                pygame.display.update()
                
        if not self.running:
            return False
                
        # Always render regardless of focus state
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
        
        # Draw landers with appropriate colors and number labels
        for i, lander in enumerate(landers):
            color = lander.get_color()
            
            # Draw main body
            pygame.draw.polygon(self.screen, color, lander.get_vertices())
            
            # Draw legs
            left_leg, right_leg = lander.get_leg_positions()
            pygame.draw.line(self.screen, color, *left_leg)
            pygame.draw.line(self.screen, color, *right_leg)
            
            # Draw lander number
            number_text = self.number_font.render(str(i + 1), True, self.BLACK)
            number_rect = number_text.get_rect()
            number_rect.center = (int(lander.x), int(lander.y))
            self.screen.blit(number_text, number_rect)
        
        # Update display with vsync if possible
        pygame.display.flip()
        return True
        
    def close(self):
        pygame.quit()