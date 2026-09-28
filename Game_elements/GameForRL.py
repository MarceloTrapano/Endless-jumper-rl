import pygame
import numpy as np
from .Game_objects import *
class HotyTowerRL:
    def __init__(self, width, height, render=True, real_time=False, deterministic=False):
        self.width = width
        self.height = height
        self.do_render = render
        self.real_time = real_time
        self.deterministic = deterministic

        pygame.init()
        pygame.font.init()

        if self.do_render:
            pygame.display.set_caption("Hoty Tower")
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = pygame.Surface((self.width, self.height))

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
        self.max_score_reached = 0

        self.harold = Harold(INIT_X, INIT_Y, SHAPE_X, SHAPE_Y, image=pygame.transform.scale(pygame.image.load("assets/cat.png"), (SHAPE_X+10, SHAPE_Y)))
        self.blocks["hello world"] = Block(100, 570, 400)

        self.frame = 0
        if self.deterministic:
            self.platform_generator = np.random.default_rng(523)
        else:
            self.platform_generator = np.random.default_rng()

        self.place_blocks()

    def step(self, action):
        self.frame += 1
        prev_score = self.score

        for event in pygame.event.get():
            if event == pygame.QUIT:
                pygame.quit()
                quit()

        self.move(action)

        if self.do_render:
            self.screen.fill((0, 0, 0))
            text_surface = self.my_font.render(f'Score: {self.final_score}', False, (255, 0, 0))
            self.screen.blit(text_surface, (0, 0))
            text_surface = self.my_font.render(f'Combo: {self.combo}', False, (255, 0, 0))
            self.screen.blit(text_surface, (0, 20))

        if self.camera_roll:
            if self.harold.y < 200:
                scroll = 200 - self.harold.y
                self.harold.y = 200
                for block in self.blocks.values():
                    block.y += scroll

        on_ground = False

        reward = 0.0

        for val, block in self.blocks.items():
            if self.do_render:
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
                        if val // 8 - self.score != 0:
                            if val // 8 - self.score > 10:
                                self.combo += val // 80 - self.score // 10
                                self.last_land_time = pygame.time.get_ticks()
                            else:
                                if self.combo > 4:
                                    self.agg_combo += self.combo ** 2
                                if self.best_combo < self.combo:
                                    self.best_combo = self.combo
                                self.combo = 0
                        self.score = val // 8
                        self.final_score = self.score + self.agg_combo
            if self.camera_roll:
                elapsed = pygame.time.get_ticks() - self.roll_start_time
                block.y += 1 #+ elapsed // 30_000

        # Combo timeout
        if pygame.time.get_ticks() - self.last_land_time > COMBO_TIMEOUT:
            if self.combo > 4:
                self.agg_combo += self.combo ** 2   
            if self.best_combo < self.combo:
                self.best_combo = self.combo
            self.combo = 0
            self.last_land_time = pygame.time.get_ticks()

        if not on_ground:
            self.harold.airborne = True

        # Fizyka + kamera
        if not self.camera_roll:
            if self.harold.y < CAMERA_LINE:
                self.camera_roll = True
                self.roll_start_time = pygame.time.get_ticks()
            self.harold.move(GRAVITY, FRICTION, MAX_SPEED)
        else:
            self.harold.move(GRAVITY, FRICTION, MAX_SPEED, cap_y=700)

        # FUNKCJA NAGRODY
        

        #if not self.harold.airborne:    # Mała nagroda za samą prędkość, gdy jesteśmy na ziemi (zachęta do rozbiegu)
        #    reward += abs(self.harold.velocity_x) * 1/50

        if not self.harold.airborne:
            charge_fraction = min(self.harold.charge, 20) / 20
            height_fraction = min(self.score / 50, 1.0)  # scales up as agent climbs
            reward += charge_fraction * height_fraction

        # 2. Mała nagroda za wejście wyżej niż poprzednio w tej grze
        score_delta = self.score - prev_score
        if score_delta > 0:
            reward += 1/2*(score_delta-10)**2+10

        # 3. Kara za cofnięcie się
        if score_delta < 0:
            reward += score_delta * 1 # score_delta ujemne → kar

        if np.argmax(action) == 2:
            # reward -= 0.01
            if not self.harold.airborne:
                reward += abs(self.harold.velocity_x) * 1/6

        #if self.harold.airborne:
        #    if np.argmax(action) == 2:
        #        reward -= 0.1
        #if self.score == 0:
        #    reward -= 0.1

        # 5. Duża kara za śmierć
        done = self.harold.y > 600 or self.inactive_frames > 1_000
        if done:
            reward -= 15.0

        if self.do_render:
            self.screen.blit(self.harold.image, self.harold)
            pygame.display.flip()

        if self.score > self.max_score_reached:
            self.max_score_reached = self.score
            reward += 1
            self.inactive_frames = 0
        else:
            self.inactive_frames += 1

        if self.real_time:
            self.clock.tick(60)

        return reward, done, self.score, self.best_combo

    def place_blocks(self):
        while self.render_distance not in self.blocks:
            if self.render_distance < 80_000:
                block_w = self.platform_generator.integers(5, 30) * 10
                block_x = -100
                while block_x + block_w > 500 or block_x == -100:
                    if not (len(self.blocks) + 1) % 50:
                        block_x = 100
                        block_w = 400
                    else:
                        block_x = self.platform_generator.integers(10, 45) * 10
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
            case 2:  # jump
                if not self.harold.airborne:
                    if self.harold.charge < 20:
                        charge = 0.3
                    else:
                        charge = 1
                    self.harold.velocity_y = -10 - np.abs(self.harold.velocity_x) * MULTIPLIER * charge
                    self.harold.airborne = True