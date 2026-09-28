import torch
from NeuroEvolution import Agent
from PPO import PPOAgent
from Game_elements import HotyTowerRL
from NeuroEvolution import Qnet
from PPO import ActorCritic
import pygame

PPO = 0
EVO = 1
mdl = ["model/ppo_model_final.pth", "model/best_evo_parallel_final.pth"]
MODEL_PATH = mdl[PPO]
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
MAX_DEPTH = 8

def watch():
    game = HotyTowerRL(SCREEN_WIDTH, SCREEN_HEIGHT, render=True, real_time=True, deterministic=False)

    if MODEL_PATH == mdl[EVO]:
        model = Qnet(6 + MAX_DEPTH * 3, 264, 128, 4)
        model = Qnet(6 + MAX_DEPTH * 3, 64, 32, 4)
        agent = Agent(game, model=model)
    else:
        model = ActorCritic(6 + MAX_DEPTH * 3, hidden=128)
        agent = PPOAgent(game, model=model)
    
    try:
        checkpoint = torch.load(MODEL_PATH)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
    except Exception as e:
        print(e)
        return

    running = True
    while running:
        game.reset()
        done = False
        score = 0
        
        while not done:
            state = agent.get_state()
            
            if MODEL_PATH == mdl[EVO]:
                action = agent.get_action(state)
            else:
                action, _, _ = agent.get_action(state)

            _, done, score, best_combo = game.step(action)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True

        print(f"Score: {score} | Best combo: {best_combo}")

if __name__ == "__main__":
    watch()