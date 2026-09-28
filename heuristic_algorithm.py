import pandas as pd
from Game_elements import HotyTowerRL
import pygame

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

def jump():
    return [0,0,1,0]

def sleep():
    yield [0,0,0,1]

def left():
    return [1,0,0,0]

def right():
    return [0,1,0,0]

def small_jump_right():
    yield right()
    for _ in range(5):
        yield jump()
    for _ in range(10):
        yield right()


def small_jump_left():
    yield left()
    for _ in range(5):
        yield jump()
    for _ in range(10):
        yield left()

def jump_right():
    yield right()
    for _ in range(5):
        yield jump()
    for _ in range(20):
        yield right()

def jump_left():
    yield left()
    for _ in range(5):
        yield jump()
    for _ in range(20):
        yield left()

def jump_straight():
    for _ in range(20):
        yield jump()


def pick_move(game: HotyTowerRL):
    x = game.harold.x
    y = game.harold.y
    vel_x = game.harold.velocity_x
    vel_y = game.harold.velocity_y
    airborne = game.harold.airborne
    charge = game.harold.charge
    choosen_x = x
    choosen_y = y
    if not airborne:
        for block in game.blocks.values():
            if -10 < (y - block.y) < 50:
                choosen_x = (block.x + block.w/2)
                choosen_y = block.y
                break
        if (x - choosen_x) > 50:
            return 0
        if (x - choosen_x) > 10:
            return 3
        if (x - choosen_x) < -50:
            return 1
        if (x - choosen_x) < -10:
            return 6
        return 2
    else:
        return 4



def main():
    game = HotyTowerRL(SCREEN_WIDTH, SCREEN_HEIGHT, render=True, real_time=True, deterministic=False)
    running = True
    scores = []
    rewards = []
    for _ in range(1_000):
        game.reset()
        done = False
        generator = None
        total_reward = 0
        while not done:
        
            move = pick_move(game)

            try:
                action = next(generator)

            except (TypeError, StopIteration):
                match move:
                    case 0:
                        generator = jump_left()
                    case 1:
                        generator = jump_right()
                    case 2:
                        generator = jump_straight()
                    case 3:
                        generator = small_jump_left()
                    case 6:
                        generator = small_jump_right()
                    case 4:
                        generator = sleep()

                action = next(generator)

            reward, done, score, best_combo = game.step(action)
            total_reward += reward

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
        scores.append(score)
        rewards.append(total_reward)

        print(f"Score: {score} | Best combo: {best_combo}")
    #pd.DataFrame({
    #    "Scores": scores,
    #    "Fitness": rewards,
    #}).to_csv("Heuristic_results.csv")

if __name__=="__main__":
    main()