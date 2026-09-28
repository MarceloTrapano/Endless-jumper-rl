import torch
import torch.nn as nn
import torch.nn.functional as F
import os

MAX_DEPTH = 8

class Qnet(nn.Module):

    def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, hidden_size_2)
        self.linear4 = nn.Linear(hidden_size_2, hidden_size_2)
        self.linear5 = nn.Linear(hidden_size_2, hidden_size_2)
        self.linear6 = nn.Linear(hidden_size_2, output_size)

    def forward(self, X):
        out = self.linear1(X)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        out = F.relu(out)
        out = self.linear6(out)


        return out

    def save(self, path="model/qnet_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
            "input_size": self.linear1.in_features,
            "hidden_size_1": self.linear1.out_features,
            "hidden_size_2": self.linear2.out_features,
            "output_size": self.linear6.out_features,
        }, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path="model/qnet_model.pth"):
        checkpoint = torch.load(path)
        model = cls(
            checkpoint["input_size"],
            checkpoint["hidden_size_1"],
            checkpoint["hidden_size_2"],
            checkpoint["output_size"],
        )
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"Model loaded ← {path}")
        return model
    
class QnetDualStream(nn.Module):
    def __init__(self):
        super().__init__()

        # Stream A — dane gracza (6 wartości)
        self.player_stream = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Stream B — dane platform (20 * 3 = 60 wartości)
        self.platform_stream = nn.Sequential(
            nn.Linear(MAX_DEPTH*3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Wspólne warstwy po concat (32 + 64 = 96)
        self.shared = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 akcje
        )

    def forward(self, x):
        player_data   = x[:, :6]    # pierwsze 6 wartości
        platform_data = x[:, 6:]    # pozostałe 60

        player_out   = self.player_stream(player_data)
        platform_out = self.platform_stream(platform_data)

        combined = torch.cat([player_out, platform_out], dim=1)
        return self.shared(combined)
    
    def save(self, path="model/dual_qnet_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
        }, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path="model/dual_qnet_model.pth"):
        checkpoint = torch.load(path)
        model = cls()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"Model loaded ← {path}")
        return model
    
class QnetAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # ===== PLAYER STREAM =====
        self.player_stream = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # ===== PLATFORM EMBEDDING =====
        self.platform_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        # ===== SELF ATTENTION =====
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )

        # ===== POST ATTENTION =====
        self.platform_post = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # ===== SHARED HEAD =====
        self.shared = nn.Sequential(
            nn.Linear(32 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)   # 4 akcje
        )

    def forward(self, x):

        player_data = x[:, :6]
        platform_data = x[:, 6:]

        # reshape -> (batch, 20, 3)
        platforms = platform_data.view(-1, MAX_DEPTH, 3)

        # embed każdej platformy
        platforms = self.platform_embed(platforms)

        # self-attention
        attn_out, _ = self.attention(platforms, platforms, platforms)

        # pooling (mean)
        platforms = attn_out.mean(dim=1)

        platforms = self.platform_post(platforms)

        # player stream
        player_out = self.player_stream(player_data)

        # concat
        combined = torch.cat([player_out, platforms], dim=1)

        return self.shared(combined)

    def save(self, path="model/attention_qnet_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.state_dict(),
        }, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path="model/attention_qnet_model.pth"):
        checkpoint = torch.load(path)
        model = cls()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        print(f"Model loaded ← {path}")
        return model