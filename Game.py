import pygame
import numpy as np

INIT_X = 300
INIT_Y = 520
SHAPE_X = 20
SHAPE_Y = 50
GRAVITY = 0.5
FRICTION = 0.3
COLOR = (255,0,0)
COLOR_GRAY = (200,200,200)
VELOCITY = 0.6
MAX_SPEED = 6
MULTIPLIER = 0.8
CAMERA_LINE = 350
COMBO_TIMEOUT = 3000

class Harold(pygame.Rect):
    def __init__(self, x, y, width, height, image=None):
        pygame.Rect.__init__(self, x, y, width, height)
        self.image = image
        self.airborne = False
        self.velocity_y = 0
        self.velocity_x = 0
        self.charge = 0
    def move(self, gravity, friction, max_speed, cap_y=550, cap_x=500):
        self.y += self.velocity_y
        self.velocity_y += gravity
        
        if np.abs(self.velocity_x) < friction:
            self.velocity_x = 0
        else:
            self.velocity_x -= friction * np.sign(self.velocity_x)

        if self.velocity_x > max_speed:
            self.velocity_x = max_speed

        if self.velocity_x < -max_speed:
            self.velocity_x = -max_speed

        if self.y > cap_y:
            self.y = cap_y
            self.airborne = False

        if self.x > cap_x - 20:
            self.x = cap_x - 20
            if abs(self.velocity_x) > 1:
                self.velocity_x *= -0.8
            else:
                self.velocity_x = 0

        if self.x < 100:
            self.x = 100
            if abs(self.velocity_x) > 1:
                self.velocity_x *= -0.8
            else:
                self.velocity_x = 0

        if abs(self.velocity_x) == max_speed:
            self.charge += 1
        else:
            if not self.airborne:
                self.charge = 0
        self.x += self.velocity_x

class Block(pygame.Rect):
    def __init__(self, x, y, width, height=20, image=None):
        self.x = x
        self.y = y
        self.w = width
        self.h = height
        pygame.Rect.__init__(self, x, y, width, height)
        self.image = image
        


def main():
    # Global variables
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 600

    #Boolean variables
    running = True
    camera_roll = False

    #Variables
    score = 0
    agg_combo = 0
    final_score = 0
    combo = 0
    best_combo = 0
    render_distance = 0
    last_land_time = pygame.time.get_ticks()
    blocks = {}
    roll_start_time = None

    pygame.init()

    # Set up the game window
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.font.init() 
    my_font = pygame.font.SysFont('Comic Sans MS', 30)
    pygame.display.set_caption("Hoty Tower")
    harold = Harold(INIT_X, INIT_Y, SHAPE_X, SHAPE_Y) 
    blocks["hello world"] = Block(100, 570, 400)
    
    # Game loop

    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]: # Left
            harold.velocity_x -= VELOCITY
        if keys[pygame.K_d]: # Right
            harold.velocity_x += VELOCITY
        if keys[pygame.K_w] and not harold.airborne: # Jump
            if harold.charge < 20:
                charge = 0.3
            else:
                charge = 1 
            harold.velocity_y = -10 - np.abs(harold.velocity_x)*MULTIPLIER*charge
            harold.airborne = True

        if render_distance not in blocks:
            if render_distance < 80_000:
                block_w = np.random.randint(5,30)*10
                block_x = -100
                while block_x + block_w > 500 or block_x == -100:
                    if not (len(blocks)+1)%50:
                        block_x = 100
                        block_w = 400
                    else: 
                        block_x = np.random.randint(10, 45)*10
                blocks[render_distance] = Block(block_x, 480 - render_distance, block_w)

                render_distance += 80


        screen.fill((0, 0, 0))

        text_surface = my_font.render(f'Score: {final_score}', False, (255, 0, 0))
        screen.blit(text_surface, (0,0))

        text_surface = my_font.render(f'Combo: {combo}', False, (255, 0, 0))
        screen.blit(text_surface, (0,20))

        if camera_roll:
            if harold.y < 200:
                scroll = 200 - harold.y
                harold.y = 200
                for block in blocks.values():
                    block.y += scroll

        on_ground = False

        for val, block in blocks.items():
            pygame.draw.rect(screen, COLOR_GRAY, block)
            if harold.velocity_y > 0 and harold.colliderect(block):
                if harold.bottom - harold.velocity_y <= block.top:
                    harold.bottom = block.top
                    harold.velocity_y = 0
                    on_ground = True
                    harold.airborne = False
                    if isinstance(val, str):
                        score = 0
                    else:
                        prev_score = score
                        if val//8 - prev_score != 0:
                            if val//8 - prev_score > 10:
                                combo += val//80 - prev_score//10
                                last_land_time = pygame.time.get_ticks()
                            else:
                                if combo > 4:
                                    agg_combo += combo**2
                                if best_combo < combo:
                                    best_combo = combo
                                combo = 0
                        score = val//8
                        final_score = score + agg_combo
            if camera_roll:
                elapsed = pygame.time.get_ticks() - roll_start_time
                block.y += 1 + elapsed // 30_000
                    
        # Reset po 3 sekundach
        if pygame.time.get_ticks() - last_land_time > COMBO_TIMEOUT:
            if combo > 4:
                agg_combo += combo**2
            if best_combo < combo:
                best_combo = combo
            combo = 0
            last_land_time = pygame.time.get_ticks()

        if not on_ground:
            harold.airborne = True

        if not camera_roll:
            if harold.y < CAMERA_LINE:
                camera_roll = True
                roll_start_time = pygame.time.get_ticks()
            harold.move(GRAVITY, FRICTION, MAX_SPEED)
        else:
            harold.move(GRAVITY, FRICTION, MAX_SPEED, cap_y=700) 

        if harold.y > 600:
            running = False
        
        pygame.draw.rect(screen, COLOR, harold)
        pygame.display.flip()
        clock.tick(60)


    # Quit Pygame
    pygame.quit()

if __name__=="__main__":
    main()