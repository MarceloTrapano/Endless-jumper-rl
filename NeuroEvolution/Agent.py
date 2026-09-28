from typing import Literal
import torch
import numpy as np
from .Models import Qnet, QnetDualStream, QnetAttention
from Game_elements import MAX_SPEED
from Game_elements import HotyTowerRL

ACTION_FRAME_LENGTH = 8
MAX_DEPTH = 8
model_type: Literal["dual","qnet","attention"] = "qnet"

class Agent:
    def __init__(self, game: HotyTowerRL, model=None):
        self.game = game
        self.fitness = 0
        self.action_frames = 0
        self.current_move = None
    
        if model is not None:
            self.model = model
        else:
            match model_type:
                case "qnet":
                    self.model = Qnet(6 + MAX_DEPTH * 3, 64, 32, 4)
                case "dual":
                    self.model = QnetDualStream()
                case "attention":
                    self.model = QnetAttention()

    def get_state(self):
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

    def get_action(self, state):
        if self.action_frames > 0:
            self.action_frames -= 1
            finalMove = [0, 0, 0, 0]
            finalMove[self.current_move] = 1
            return finalMove

        stateTensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(stateTensor)
            move = torch.argmax(prediction).item()

        self.current_move = move
        if move in (0, 1): 
            self.action_frames = ACTION_FRAME_LENGTH
        elif move == 2: 
            self.action_frames = 2

        finalMove = [0, 0, 0, 0]
        finalMove[move] = 1
        return finalMove