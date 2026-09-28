import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from Game_elements import HotyTowerRL
from PPO import ROLLOUT_STEPS, PPOAgent

NUMBER_OF_GAMES = 10_000
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

scores_history = []
mean_scores = []
combo_history = []
mean_combos = []
entropy_history = []


def train():
    best_score = 0
    best_combo = 0
    total_score = 0
    total_combo = 0

    game = HotyTowerRL(SCREEN_WIDTH, SCREEN_HEIGHT, render=False, deterministic=False)
    agent = PPOAgent(game)

    print("=" * 60)
    print("  PPO Training — Hoty Tower")
    print(f"  Rollout steps : {ROLLOUT_STEPS}")
    print(f"  Target games  : {NUMBER_OF_GAMES}")
    print("=" * 60)

    while agent.num_of_games < NUMBER_OF_GAMES:
        game.reset()
        done = False
        score = 0
        ep_best_combo = 0

        while not done:
            state = agent.get_state()
            action, log_prob, value = agent.get_action(state)

            reward, done, score, ep_best_combo = game.step(action)

            agent.store_transition(state, action, log_prob, reward, done, value)

            if agent.steps_since_update >= ROLLOUT_STEPS:
                if not done:
                    next_state = agent.get_state()
                    next_t = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
                    with torch.no_grad():
                        _, lv = agent.model(next_t)
                    last_val = lv.item()
                else:
                    last_val = 0.0

                stats = agent.maybe_update(last_val)
                if stats:
                    entropy_history.append(stats["entropy"])

        agent.num_of_games += 1
        total_score += score
        total_combo += ep_best_combo

        mean_score = total_score / agent.num_of_games
        mean_combo = total_combo / agent.num_of_games

        scores_history.append(score)
        mean_scores.append(mean_score)
        combo_history.append(ep_best_combo)
        mean_combos.append(mean_combo)

        if score > best_score:
            best_score = score
            agent.save_model()

        if ep_best_combo > best_combo:
            best_combo = ep_best_combo

        print(
            f"Game {agent.num_of_games:>6} | "
            f"Score {score:>5} | Best {best_score:>5} | Mean {mean_score:>6.1f} | "
            f"Combo {ep_best_combo:>3} | Best combo {best_combo:>3} | "
            f"Mean combo {mean_combo:>4.1f}"
        )


if __name__ == "__main__":
    train()

    pd.DataFrame(
        {
            "Score": scores_history,
            "Mean Score": mean_scores,
            "Combo": combo_history,
            "Mean Combo": mean_combos,
        }
    ).to_csv("Training_result_PPO.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(scores_history, alpha=0.4, label="Score")
    axes[0].plot(mean_scores, label="Mean Score")
    axes[0].set_title("Score per episode")
    axes[0].legend()

    axes[1].plot(combo_history, alpha=0.4, label="Best Combo")
    axes[1].plot(mean_combos, label="Mean Combo")
    axes[1].set_title("Combo per episode")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("Training_result_PPO.png", dpi=120)
    plt.show()
    print("Saved → Training_result_PPO.csv / .png")