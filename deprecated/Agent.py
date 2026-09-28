from typing import Literal

from Qnet import Qnet, QnetDualStream, QnetAttention
from QTrainer import QTrainer
import numpy as np
import random
import torch
from Game_objects import MAX_SPEED
from GameForRL import HotyTowerRL
from collections import deque
import os

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
MAX_DEPTH = 8
LR = 5e-4
QNET_MODEL_PATH = "model/qnet_model.pth"
DUAL_MODEL_PATH = "model/dual_qnet_model.pth"
ATTENTION_MODEL_PATH = "model/attention_qnet_model.pth"
model_type: Literal["dual","qnet","attention"] = "qnet"

class Agent:
    def __init__(self, game: HotyTowerRL):
        self.num_of_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.action_frames = 0
        self.current_move = None
    
        match model_type:
            case "dual":
                if os.path.exists(DUAL_MODEL_PATH):
                    self.model = QnetDualStream.load(DUAL_MODEL_PATH)
                else:
                    self.model = QnetDualStream()
            case "qnet":
                if os.path.exists(QNET_MODEL_PATH):
                    self.model = Qnet.load(QNET_MODEL_PATH)
                else:
                    self.model = Qnet(6 + MAX_DEPTH * 3, 64, 32, 4)
            case "attention":
                if os.path.exists(ATTENTION_MODEL_PATH):
                    self.model = QnetAttention.load(ATTENTION_MODEL_PATH)
                else:
                    self.model = QnetAttention()

        self.trainer = QTrainer(self.model, LR, self.gamma)
        self.game = game

    def save_model(self):
        match model_type:
            case "dual":
                self.model.save(DUAL_MODEL_PATH)
            case "qnet":
                self.model.save(QNET_MODEL_PATH)
            case "attention":
                self.model.save(ATTENTION_MODEL_PATH)

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
    
    def remember(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            sample = self.memory
        else:
            sample = random.sample(self.memory, BATCH_SIZE)

        states, actions, rewards, nextStates, dones = zip(*sample)
        self.trainer.trainStep(states, actions, rewards, nextStates, dones)

    def train_short_memory(self, state, action, reward, nextState, done):
        self.trainer.trainStep(state, action, reward, nextState, done)

    def get_action(self, state):
        self.epsilon = max(5, 100 - self.num_of_games * 0.05)

        if self.action_frames > 0:
            self.action_frames -= 1
            finalMove = [0, 0, 0, 0]
            finalMove[self.current_move] = 1
            return finalMove

        finalMove = [0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = np.random.choice([0, 1, 2, 3], p=[0.40, 0.40, 0.19, 0.01])
        else:
            stateTensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            prediction = self.model(stateTensor)
            move = torch.argmax(prediction).item()

        self.current_move = move
        if move in (0, 1):
            self.action_frames = 8
        elif move == 2:
            self.action_frames = 2

        finalMove[move] = 1
        return finalMove