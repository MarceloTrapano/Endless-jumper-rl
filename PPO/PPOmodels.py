import os
import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_DEPTH = 8


class ActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden: int = 128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def save(self, path: str = "model/ppo_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state": self.state_dict(),
                "input_size": self.backbone[0].in_features,
                "hidden": self.backbone[0].out_features,
            },
            path,
        )
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = "model/ppo_model.pth"):
        ckpt = torch.load(path, weights_only=False)
        model = cls(ckpt["input_size"], ckpt["hidden"])
        model.load_state_dict(ckpt["model_state"])
        print(f"Model loaded ← {path}")
        return model


class ActorCriticDual(nn.Module):
    def __init__(self):
        super().__init__()

        self.player_stream = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.platform_stream = nn.Sequential(
            nn.Linear(MAX_DEPTH * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.shared = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(64, 4)
        self.critic_head = nn.Linear(64, 1)

        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        player_out = self.player_stream(x[:, :6])
        platform_out = self.platform_stream(x[:, 6:])
        features = self.shared(torch.cat([player_out, platform_out], dim=1))
        logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def save(self, path: str = "model/ppo_dual_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"model_state": self.state_dict()}, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = "model/ppo_dual_model.pth"):
        ckpt = torch.load(path, weights_only=False)
        model = cls()
        model.load_state_dict(ckpt["model_state"])
        print(f"Model loaded ← {path}")
        return model