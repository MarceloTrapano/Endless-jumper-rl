import pygame
import numpy as np
from Game import Harold, Block, INIT_X, INIT_Y, SHAPE_X, SHAPE_Y, VELOCITY, MULTIPLIER, COLOR_GRAY, COLOR, COMBO_TIMEOUT, CAMERA_LINE, FRICTION, GRAVITY, MAX_SPEED

class HotyTowerRL:
    def __init__(self, width, height):
        self.width = width
        self.height = height

        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("Hoty Tower")

        self.screen = pygame.display.set_mode((self.width, self.height))
        self.my_font = pygame.font.SysFont('Comic Sans MS', 30)
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        self.camera_roll = False
        self.score = 0
        self.agg_combo = 0
        self.inactive_frames = 0
        self.final_score = 0
        self.combo = 0
        self.best_combo = 0
        self.render_distance = 0
        self.last_land_time = pygame.time.get_ticks()
        self.roll_start_time = None
        self.blocks = {}

        self.harold = Harold(INIT_X, INIT_Y, SHAPE_X, SHAPE_Y) 
        self.blocks["hello world"] = Block(100, 570, 400)
        
        self.frame = 0

        self.place_blocks()

    def step(self, action):
        self.frame += 1

        for event in pygame.event.get():
            if event == pygame.QUIT:
                pygame.quit()
                quit()
        
        self.move(action)

        reward = 0

        self.screen.fill((0, 0, 0))

        text_surface = self.my_font.render(f'Score: {self.final_score}', False, (255, 0, 0))
        self.screen.blit(text_surface, (0,0))

        text_surface = self.my_font.render(f'Combo: {self.combo}', False, (255, 0, 0))
        self.screen.blit(text_surface, (0,20))

        if self.camera_roll:
            if self.harold.y < 200:
                scroll = 200 - self.harold.y
                self.harold.y = 200
                for block in self.blocks.values():
                    block.y += scroll

        on_ground = False

        for val, block in self.blocks.items():
            pygame.draw.rect(self.screen, COLOR_GRAY, block)
            if self.harold.velocity_y > 0 and self.harold.colliderect(block):
                if self.harold.bottom - self.harold.velocity_y <= block.top:
                    self.harold.bottom = block.top
                    self.harold.velocity_y = 0
                    on_ground = True
                    self.harold.airborne = False
                    if isinstance(val, str):
                        self.score = 0
                    else:
                        prev_score = self.score
                        if val//8 - prev_score != 0:
                            if val//8 - prev_score > 10:
                                self.combo += val//80 - prev_score//10
                                self.last_land_time = pygame.time.get_ticks()
                            else:
                                if self.combo > 4:
                                    self.agg_combo += self.combo**2
                                if self.best_combo < self.combo:
                                    self.best_combo = self.combo
                                self.combo = 0
                        self.score = val//8
                        self.final_score = self.score + self.agg_combo
            if self.camera_roll:
                elapsed = pygame.time.get_ticks() - self.roll_start_time
                block.y += 1 + elapsed // 30_000
                    
        if pygame.time.get_ticks() - self.last_land_time > COMBO_TIMEOUT:
            if self.combo > 4:
                self.agg_combo += self.combo**2
            if self.best_combo < self.combo:
                self.best_combo = self.combo
            self.combo = 0
            self.last_land_time = pygame.time.get_ticks()

        if not on_ground:
            self.harold.airborne = True

        if not self.camera_roll:
            if self.harold.y < CAMERA_LINE:
                self.camera_roll = True
                self.roll_start_time = pygame.time.get_ticks()
            self.harold.move(GRAVITY, FRICTION, MAX_SPEED)
        else:
            self.harold.move(GRAVITY, FRICTION, MAX_SPEED, cap_y=700) 

        reward = self.score + self.best_combo

        if self.score == 0:
            reward = -10
        pygame.draw.rect(self.screen, COLOR, self.harold)
        pygame.display.flip()

        if self.score == 0:
            self.inactive_frames += 1

        if self.harold.y > 600 or self.inactive_frames > 2_000:
            return reward, True, self.score, self.best_combo
        
        return reward, False, self.score, self.best_combo
        
        

    def place_blocks(self):
        while self.render_distance not in self.blocks:
            if self.render_distance < 80_000:
                block_w = np.random.randint(5,30)*10
                block_x = -100
                while block_x + block_w > 500 or block_x == -100:
                    if not (len(self.blocks)+1)%50:
                        block_x = 100
                        block_w = 400
                    else: 
                        block_x = np.random.randint(10, 45)*10
                self.blocks[self.render_distance] = Block(block_x, 480 - self.render_distance, block_w)

                self.render_distance += 80
            else:
                break


    def move(self, action):
        # action - one hot [left, right, jump, do nothing]

        match np.argmax(action):
            case 0:  # left
                self.harold.velocity_x -= VELOCITY
            case 1:  # right
                self.harold.velocity_x += VELOCITY
            case 2:
                if not self.harold.airborne:
                    if self.harold.charge < 20:
                        charge = 0.3
                    else:
                        charge = 1 
                    self.harold.velocity_y = -10 - np.abs(self.harold.velocity_x)*MULTIPLIER*charge
                    self.harold.airborne = True
