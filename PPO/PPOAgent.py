import os
from typing import Literal

import numpy as np
import torch
from torch.distributions import Categorical

from Game_elements import MAX_SPEED, HotyTowerRL
from .PPOmodels import ActorCritic, ActorCriticDual
from .PPOTrainer import PPOTrainer

MAX_DEPTH = 8
LR = 3e-4
GAMMA = 0.99
LAM = 0.95
ROLLOUT_STEPS = 2048
CLIP_EPS = 0.2
PPO_EPOCHS = 4
MINI_BATCH = 256
ENTROPY_COEF = 0.01
VF_COEF = 0.5

PPO_MODEL_PATH = "model/ppo_model.pth"
PPO_DUAL_MODEL_PATH = "model/ppo_dual_model.pth"

model_type: Literal["ppo", "ppo_dual"] = "ppo"


class PPOAgent:
    def __init__(self, game: HotyTowerRL, model=None):
        self.num_of_games = 0
        self.game = game

        input_size = 6 + MAX_DEPTH * 3
        if model is None:
            match model_type:
                case "ppo":
                    if os.path.exists(PPO_MODEL_PATH):
                        self.model = ActorCritic.load(PPO_MODEL_PATH)
                    else:
                        self.model = ActorCritic(input_size, hidden=128)
                case "ppo_dual":
                    if os.path.exists(PPO_DUAL_MODEL_PATH):
                        self.model = ActorCriticDual.load(PPO_DUAL_MODEL_PATH)
                    else:
                        self.model = ActorCriticDual()
        else:
            self.model = model

        self.trainer = PPOTrainer(
            self.model,
            lr=LR,
            gamma=GAMMA,
            lam=LAM,
            clip_eps=CLIP_EPS,
            epochs=PPO_EPOCHS,
            mini_batch=MINI_BATCH,
            entropy_coef=ENTROPY_COEF,
            vf_coef=VF_COEF,
        )

        self.steps_since_update = 0

    def save_model(self):
        match model_type:
            case "ppo":
                self.model.save(PPO_MODEL_PATH)
            case "ppo_dual":
                self.model.save(PPO_DUAL_MODEL_PATH)

    def get_state(self) -> np.ndarray:
        player_x = self.game.harold.x
        player_y = self.game.harold.y

        all_blocks = list(self.game.blocks.values())

        above = [b for b in all_blocks if -300 < b.y - player_y < 0]
        above.sort(key=lambda b: b.y - player_y, reverse=True)

        below = [b for b in all_blocks if 0 <= b.y - player_y < 150]
        below.sort(key=lambda b: b.y - player_y)

        visible = (above + below)[:MAX_DEPTH]

        field_of_view = []
        for b in visible:
            rel_x = ((b.x + b.w / 2) - player_x) / 500
            rel_y = (b.y - player_y) / 600
            norm_w = b.w / 400
            field_of_view.extend([rel_x, rel_y, norm_w])

        state = [
            self.game.harold.x / 500,
            self.game.harold.y / 600,
            self.game.harold.velocity_x / MAX_SPEED,
            self.game.harold.velocity_y / 20,
            float(self.game.harold.airborne),
            self.game.harold.charge / 20,
        ]

        while len(field_of_view) < MAX_DEPTH * 3:
            field_of_view.extend([0.0, 0.0, 0.0])

        state.extend(field_of_view)
        return np.array(state, dtype=np.float32)

    def get_action(self, state: np.ndarray):
        state_t = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.model(state_t)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        final_move = [0, 0, 0, 0]
        final_move[action.item()] = 1

        return final_move, log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, done, value):
        action_idx = int(np.argmax(action))
        self.trainer.store(state, action_idx, log_prob, reward, done, value)
        self.steps_since_update += 1

    def maybe_update(self, last_value: float = 0.0) -> dict | None:
        if self.steps_since_update >= ROLLOUT_STEPS:
            stats = self.trainer.update(last_value)
            self.steps_since_update = 0
            return stats
        return None