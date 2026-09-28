import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

class QTrainer:

    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma

        self.optimizer = optim.Adam(model.parameters(), self.lr)
        self.lossFunction = nn.MSELoss()

        # FIX 7: Target network — a frozen copy updated every N steps
        # Without this, the Q-targets shift every step, causing training divergence.
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        self.update_target_every = 500   # steps between hard target updates
        self.step_count = 0

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def trainStep(self, state, action, reward, newState, done):
        stateTensor    = torch.tensor(np.array(state),    dtype=torch.float)
        actionTensor   = torch.tensor(np.array(action),   dtype=torch.long)
        rewardTensor   = torch.tensor(np.array(reward),   dtype=torch.float)
        newStateTensor = torch.tensor(np.array(newState), dtype=torch.float)
        doneTensor     = torch.tensor(np.array(done),     dtype=torch.bool)

        if stateTensor.dim() == 1:
            stateTensor    = stateTensor.unsqueeze(0)
            newStateTensor = newStateTensor.unsqueeze(0)
            actionTensor   = actionTensor.unsqueeze(0)
            rewardTensor   = rewardTensor.unsqueeze(0)
            doneTensor     = doneTensor.unsqueeze(0)

        # FIX 7: Use frozen target network for next-state Q values
        with torch.no_grad():
            next_q_values = self.target_model(newStateTensor).max(dim=1).values

        target_q   = rewardTensor + self.gamma * next_q_values * (~doneTensor)

        prediction     = self.model(stateTensor)
        action_indices = actionTensor.argmax(dim=1, keepdim=True)
        predicted_q    = prediction.gather(1, action_indices).squeeze(1)

        self.optimizer.zero_grad()
        loss = self.lossFunction(predicted_q, target_q)
        loss.backward()

        # Gradient clipping — prevents exploding gradients with MSELoss
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.update_target_network()