import pygame
import numpy as np
from Game import Harold, Block

def draw_block(screen, block, start_tile, tile_image, end_tile):
    tile_w = tile_image.get_width()
    tiles = list(range(block.x, block.x + block.width, tile_w))
    for i, x in enumerate(tiles):
        if i == 0:
            screen.blit(start_tile, (x, block.y))
        elif i == len(tiles)-1:
            screen.blit(end_tile, (x, block.y))
        else:
            screen.blit(tile_image, (x, block.y))

def main():
    pygame.init()
    pygame.font.init() 
    # Global variables
    INIT_X = 300
    INIT_Y = 520
    SHAPE_X = 20
    SHAPE_Y = 50
    GRAVITY = 0.5
    FRICTION = 0.3
    COLOR = (255,0,0)
    COLOR_GRAY = (200,200,200)
    VELOCITY = 0.6
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 600
    MAX_SPEED = 6
    MULTIPLIER = 0.8
    CAMERA_LINE = 350
    COMBO_TIMEOUT = 3000
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    START_TILE_1 = pygame.image.load("assets/left_1.png").convert_alpha()
    MIDDLE_TILE_1 = pygame.image.load("assets/middle_1.png").convert_alpha()
    END_TILE_1 = pygame.image.load("assets/right_1.png").convert_alpha()
    BACKGROUND = pygame.image.load("assets/background.png").convert_alpha()
    PLAYER_RIGHT1 = pygame.image.load("assets/right_player_1.png").convert_alpha()
    PLAYER_LEFT1 = pygame.image.load("assets/left_player_1.png").convert_alpha()
    PLAYER_FRONT = pygame.image.load("assets/Front_player.png").convert_alpha()
    BG_TILE_H = BACKGROUND.get_height()

    #Boolean variables
    running = True
    camera_roll = False

    #Variables
    background_y_offset = 0
    bg_tiles_count = SCREEN_HEIGHT // BG_TILE_H + 2 
    score = 0
    agg_combo = 0
    final_score = 0
    combo = 0
    best_combo = 0
    render_distance = 0
    last_land_time = pygame.time.get_ticks()
    blocks = {}
    roll_start_time = None  

    

    # Set up the game window
    my_font = pygame.font.SysFont('Comic Sans MS', 30)
    pygame.display.set_caption("Hoty Tower")
    harold = Harold(INIT_X, INIT_Y, SHAPE_X, SHAPE_Y, image=PLAYER_FRONT) 
    blocks["hello world"] = Block(50, 570, 500)
    blocks["left_wall"] = Block(0,0,120,600, image=pygame.image.load("assets/wall_left.png").convert_alpha())
    blocks["right_wall"] = Block(500,0,120,600, image=pygame.image.load("assets/wall_right.png").convert_alpha())
    
    # Game loop

    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        harold.image = PLAYER_FRONT
        if keys[pygame.K_a]: # Lewo
            harold.velocity_x -= VELOCITY
            harold.image = PLAYER_LEFT1
        if keys[pygame.K_d]: # Prawo
            harold.velocity_x += VELOCITY
            harold.image = PLAYER_RIGHT1
        if keys[pygame.K_w] and not harold.airborne: # Góra
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
                        block_x = 0
                        block_w = 500
                        break
                    else: 
                        block_x = np.random.randint(10, 45)*10
                blocks[render_distance] = Block(block_x, 480 - render_distance, block_w)
                render_distance += 80


        screen.fill((0, 0, 0))

        if camera_roll:
            elapsed = pygame.time.get_ticks() - roll_start_time
            background_y_offset -= (1 + elapsed // 30_000) / 2

        for i in range(bg_tiles_count):
            y = (i * BG_TILE_H - int(background_y_offset)) % (bg_tiles_count * BG_TILE_H) - BG_TILE_H
            screen.blit(BACKGROUND, (95, y))


        if camera_roll:
            if harold.y < 200:
                scroll = 200 - harold.y
                harold.y = 200
                for val, block in blocks.items():
                    if val in ["left_wall", "right_wall"]:
                        continue
                    block.y += scroll
                background_y_offset -= scroll / 2


        on_ground = False

        for val, block in blocks.items():
            if val in ["left_wall", "right_wall"]:
                continue
            draw_block(screen, block, START_TILE_1, MIDDLE_TILE_1, END_TILE_1)
            #pygame.draw.rect(screen, COLOR_GRAY, block)
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

        for val, block in blocks.items():
            if val == "hello world":
                continue
            if val in ["left_wall", "right_wall"]:
                screen.blit(block.image, block)
                continue
            break

        text_surface = my_font.render(f'Score: {final_score}', False, (255, 0, 0))
        screen.blit(text_surface, (0,0))

        text_surface = my_font.render(f'Combo: {combo}', False, (255, 0, 0))
        screen.blit(text_surface, (0,20))

        
        screen.blit(harold.image, harold)
        pygame.display.flip()
        clock.tick(60)


    # Quit Pygame
    pygame.quit()

if __name__=="__main__":
    main()