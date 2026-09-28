import torch
import copy
import numpy as np

class EvolutionManager:
    def __init__(self, model_class, population_size=50, mutation_rate=0.03):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = [model_class() for _ in range(population_size)]
        self.hall_of_fame = None   # best model ever seen
        self.hall_of_fame_fitness = -np.inf

    def evolve(self, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)[::-1]
        best_models = [self.population[i] for i in sorted_indices[:10]]

        # Update hall of fame
        if fitness_scores[sorted_indices[0]] > self.hall_of_fame_fitness:
            self.hall_of_fame = copy.deepcopy(best_models[0])
            self.hall_of_fame_fitness = fitness_scores[sorted_indices[0]]

        new_population = list(best_models)
        # Always include hall of fame
        if self.hall_of_fame is not None:
            new_population.append(copy.deepcopy(self.hall_of_fame))

        while len(new_population) < self.population_size:
            parent = np.random.choice(best_models)
            new_population.append(self.mutate(parent))

        self.population = new_population
        return best_models[0]

    def mutate(self, model):
        child = copy.deepcopy(model)
        with torch.no_grad():
            for param in child.parameters():
                mutation = torch.randn_like(param) * self.mutation_rate
                param.add_(mutation)
        return child