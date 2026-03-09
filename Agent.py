from Qnet import Qnet
from QTrainer import QTrainer
import numpy as np
import random
import torch
from GameForRL import HotyTowerRL
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
MAX_DEPTH = 20
LR = 1e-3

class Agent:
    def __init__(self, game: HotyTowerRL):
        self.num_of_games = 0
        self.epsilon = 0
        self.gamma = 0
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Qnet(6+MAX_DEPTH*3,264,128,4)
        self.trainer = QTrainer(self.model, LR, self.gamma)
        self.game = game
        self.blocks = dict(self.game.blocks)

    def get_state(self):
        self.blocks = {k: v for k, v in self.blocks.items() if v.y > -100}

        field_of_view = []
        for i ,b in enumerate(self.blocks.values()):
            if i >= MAX_DEPTH:
                break
            field_of_view.extend([b.x, b.y, b.w])

        state = [
            self.game.harold.x,
            self.game.harold.y,
            self.game.harold.airborne,
            self.game.harold.velocity_x,
            self.game.harold.velocity_y,
            self.game.harold.charge,
        ]

        while len(field_of_view) < MAX_DEPTH * 3:
            field_of_view.extend([0, 0, 0])

        state.extend(field_of_view)

        return np.array(state, dtype=int)
    
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
        # in the beginning will do some random moves, tradeoff between exploration and exploataition
        self.epsilon = 80 - self.num_of_games
        finalMove = [0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            finalMove[move] = 1
        else:
            stateTensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(stateTensor)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove