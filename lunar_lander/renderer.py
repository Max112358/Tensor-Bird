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
        self.number_font = pygame.font.Font(None, 36)
        
        # Configure window for background rendering while maintaining taskbar visibility
        info = pygame.display.get_wm_info()
        if 'window' in info:  # Check if we can get the window handle
            try:
                import ctypes
                if hasattr(ctypes, 'windll'):  # Windows specific
                    hwnd = info['window']
                    GWL_EXSTYLE = -20
                    WS_EX_APPWINDOW = 0x40000
                    WS_EX_COMPOSITED = 0x02000000
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
        self.GREEN = (0, 255, 0)
        
    def _get_lander_color(self, lander: Lander) -> tuple:
        """Determine lander color based on its state"""
        if not lander.active:
            if lander.terminate_reason == 'landed':
                return self.GREEN  # Successfully landed
            return self.BLUE  # Other terminated states (crashed/out of bounds/out of fuel)
        elif any(lander.thrusters.values()):
            return self.RED  # Thrusters firing
        return self.WHITE  # Active but drifting
        
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
            # Get color based on lander state
            color = self._get_lander_color(lander)
            
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