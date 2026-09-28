import multiprocessing as mp
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuroEvolution import Agent
from Game_elements import HotyTowerRL
from NeuroEvolution import EvolutionManager
from NeuroEvolution import Qnet

NUMBER_OF_GENERATIONS = 2_000
POPULATION_SIZE = 64
MAX_DEPTH = 8
NUM_EVAL_SEEDS = 3      # evaluate each agent on 3 maps per generation
ELITE_COUNT = 10

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

def evaluate_agent(model_state, seeds):
    try:
        model = Qnet(6 + MAX_DEPTH * 3, 64, 32, 4)
        model.load_state_dict(model_state)
        model.eval()

        total_rewards = []

        for seed in seeds:
            game = HotyTowerRL(600, 600, render=False, deterministic=False)
            game.reset()
            game.platform_generator = np.random.default_rng(seed)
            game.blocks = {}
            game.render_distance = 0
            game.place_blocks()

            agent = Agent(game, model=model)
            done = False
            total_reward = 0.0

            while not done:
                state = agent.get_state()
                action = agent.get_action(state)
                reward, done, score, _ = game.step(action)
                total_reward += reward

            total_rewards.append(total_reward)

        return float(np.mean(total_rewards)), int(score)

    except Exception as e:
        print(f"Error in worker: {e}")
        return -99999.0, 0

def train_parallel():
    evo_manager = EvolutionManager(
        lambda: Qnet(6 + MAX_DEPTH * 3, 64, 32, 4),
        population_size=POPULATION_SIZE
    )
    best_overall_score = 0
    num_cores = min(mp.cpu_count(), 16)
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(processes=num_cores)

    g = []
    min_fit = []
    mean_fit = []
    max_fit = []
    best_scores = []
    hof_fit = []

    try:
        for gen in range(NUMBER_OF_GENERATIONS):
            eval_seeds = [np.random.randint(0, 1_000_000) for _ in range(NUM_EVAL_SEEDS)]

            population_states = [m.state_dict() for m in evo_manager.population]

            results = pool.starmap(
                evaluate_agent,
                [(state, eval_seeds) for state in population_states]
            )

            fitness_scores = [r[0] for r in results]
            scores = [r[1] for r in results]

            current_max_score = max(scores)
            best_idx = np.argmax(fitness_scores)

            if current_max_score > best_overall_score:
                best_overall_score = current_max_score
                evo_manager.population[best_idx].save("model/best_evo_parallel.pth")

            evo_manager.evolve(fitness_scores)

            print(
                f"Gen {gen:4d} | "
                f"Min: {min(fitness_scores):7.1f} | "
                f"Mean: {np.mean(fitness_scores):7.1f} | "
                f"Max: {max(fitness_scores):7.1f} | "
                f"Score: {current_max_score:4d} | "
                f"Best ever: {best_overall_score:4d} | "
                f"HoF fit: {evo_manager.hall_of_fame_fitness:7.1f}"
            )

            g.append(gen)
            min_fit.append(min(fitness_scores))
            mean_fit.append(np.mean(fitness_scores))
            max_fit.append(max(fitness_scores))
            best_scores.append(current_max_score)
            hof_fit.append(evo_manager.hall_of_fame_fitness)

    except KeyboardInterrupt:
        print("Saving best model")
        
        best_model = evo_manager.population[0] 
        best_model.save("model/interrupted_model.pth")
        
    finally:
        print("Closing background processes")
        best_model = evo_manager.population[0]
        best_model.save("model/best_model.pth")
        pool.terminate()
        pool.join()

    df = pd.DataFrame({
        "Gen": g,
        "Min fit": min_fit,
        "Mean fit": mean_fit,
        "Max fit": max_fit,
        "Best score": best_scores
    })
    df.to_csv("Training_results.csv", index=False)
    df.set_index("Gen").plot()
    plt.title("Postępy treningu")
    plt.xlabel("Generacja")
    plt.ylabel("Fitness")
    plt.show()

if __name__ == "__main__":
    mp.freeze_support()
    train_parallel()