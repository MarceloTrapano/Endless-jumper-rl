import torch
import torch.nn as nn
import torch.nn.functional as F

class Qnet(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, hidden_size_2)
        self.linear4 = nn.Linear(hidden_size_2, output_size)

    def forward(self, X):
        out = self.linear1(X)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)


        return out