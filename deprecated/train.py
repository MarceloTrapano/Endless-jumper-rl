from Agent import Agent
import matplotlib.pyplot as plt
import os
import pandas as pd
from GameForRL import HotyTowerRL
import numpy as np

NUMBER_OF_GAMES = 10_000
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

scoresHistory = []
meanScores = []
comboHistory = []
meanCombos = []
rewards = []

def train():
    totalScore = 0
    bestScore = 0
    total_reward = 0

    totalCombo = 0
    bestCombo = 0

    game = HotyTowerRL(SCREEN_WIDTH, SCREEN_HEIGHT, render=False, deterministic=False)

    agent = Agent(game)

    # we train the model for 200 games
    while agent.num_of_games < NUMBER_OF_GAMES:

        # get old state
        oldState = agent.get_state()

        # move
        finalMove = agent.get_action(oldState)

        # perform move and get new state
        reward, done, score, best_combo = game.step(finalMove)

        newState = agent.get_state()
        # train short memory
        agent.train_short_memory(oldState, finalMove, reward, newState, done)

        # remember
        agent.remember(oldState, finalMove, reward, newState, done)
        total_reward += reward
        if done:
            # train long memory
            game.reset()
            agent.num_of_games += 1
            agent.train_long_memory()

            if score > bestScore:
                bestScore = score
                agent.save_model()
            if best_combo > bestCombo:
                bestCombo = best_combo


            totalScore += score
            meanScore = (totalScore / agent.num_of_games)

            totalCombo += best_combo
            meanCombo = (totalCombo / agent.num_of_games)

            scoresHistory.append(score)
            meanScores.append(meanScore)
            comboHistory.append(best_combo)
            meanCombos.append(meanCombo)
            rewards.append(total_reward)

            print("Game number: ", agent.num_of_games, "Score: ", score, "Best Score: ", bestScore, "Mean scores: ", meanScore, "Best Combo: ", bestCombo, "Mean combo: ", meanCombo, "Reward:", total_reward)

            total_reward = 0

            


if __name__ == "__main__":       
    train()

    pd.DataFrame({
        "Score": scoresHistory,
        "Mean Scores": meanScores,
        "Combo": comboHistory,
        "Mean Combo": meanCombos,
        "Fitness": rewards
    }).to_csv("Training_result_RL.csv")
    plt.plot(scoresHistory)
    plt.plot(meanScores)
    plt.legend(["Score", "Mean Score"])
    plt.show()