import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOTrainer:
    def __init__(
        self,
        model,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        mini_batch: int = 256,
        entropy_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.model = model
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.mini_batch = mini_batch
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

        self.reset_buffer()

    def reset_buffer(self):
        self.buf_states = []
        self.buf_actions = []
        self.buf_log_probs = []
        self.buf_rewards = []
        self.buf_dones = []
        self.buf_values = []

    def store(self, state, action, log_prob, reward, done, value):
        self.buf_states.append(state)
        self.buf_actions.append(action)
        self.buf_log_probs.append(log_prob)
        self.buf_rewards.append(reward)
        self.buf_dones.append(done)
        self.buf_values.append(value)

    def compute_gae(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        rewards = np.array(self.buf_rewards, dtype=np.float32)
        dones = np.array(self.buf_dones, dtype=np.float32)
        values = np.array(self.buf_values, dtype=np.float32)

        n = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            next_done = dones[t]
            delta = rewards[t] + self.gamma * next_val * (1 - next_done) - values[t]
            last_gae = delta + self.gamma * self.lam * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_value: float = 0.0):
        advantages, returns = self.compute_gae(last_value)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(np.array(self.buf_states), dtype=torch.float)
        actions = torch.tensor(np.array(self.buf_actions), dtype=torch.long)
        old_lp = torch.tensor(np.array(self.buf_log_probs), dtype=torch.float)
        advs = torch.tensor(advantages, dtype=torch.float)
        rets = torch.tensor(returns, dtype=torch.float)

        n = len(states)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.mini_batch):
                idx = indices[start : start + self.mini_batch]

                b_states = states[idx]
                b_actions = actions[idx]
                b_old_lp = old_lp[idx]
                b_advs = advs[idx]
                b_rets = rets[idx]

                logits, values_pred = self.model(b_states)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(b_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - b_old_lp)

                surr1 = ratio * b_advs
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * b_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values_pred, b_rets)

                loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()

        self.reset_buffer()

        batches = max(1, (n // self.mini_batch) * self.epochs)
        return {
            "policy_loss": total_policy_loss / batches,
            "value_loss": total_value_loss / batches,
            "entropy": total_entropy / batches,
        }