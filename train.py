from Agent import Agent
import matplotlib.pyplot as plt
from GameForRL import HotyTowerRL


SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

scoresHistory = []
meanScores = []
comboHistory = []
meanCombo = []

def train():
    totalScore = 0
    bestScore = 0

    totalCombo = 0
    bestCombo = 0

    game = HotyTowerRL(SCREEN_WIDTH, SCREEN_HEIGHT)
    agent = Agent(game)

    # we train the model for 200 games
    while agent.num_of_games < 200:

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

        if done:
            # train long memory
            game.reset()
            agent.num_of_games += 1
            agent.train_long_memory()

            if score > bestScore:
                bestScore = score
            if best_combo > bestCombo:
                bestCombo = bestCombo


            totalScore += score
            meanScore = (totalScore / agent.num_of_games)

            totalCombo += best_combo
            meanCombo = (totalCombo / agent.num_of_games)

            scoresHistory.append(score)
            meanScores.append(meanScore)

            if score == bestScore or agent.num_of_games % 10 == 0:
                print("Game number: ", agent.num_of_games, "Score: ", score, "Best Score: ", bestScore, "Mean scores: ", meanScore, "Best Combo: ", bestCombo, "Mean combo: ", meanCombo)


if __name__ == "__main__":       
    train()

    plt.plot(scoresHistory)
    plt.plot(meanScores)
    plt.legend(["Score", "Mean Score"])