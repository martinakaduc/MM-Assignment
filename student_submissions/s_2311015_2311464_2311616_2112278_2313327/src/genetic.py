# GeneticPolicy implementation
import random
import numpy as np
from policy import Policy

class GeneticPolicy(Policy):
    def __init__(self, population_size=100, generations=8, mutation_rate=0.01):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def get_action(self, observation, info):
        population = [self.random_action(observation) for _ in range(self.population_size)]
        for _ in range(self.generations):
            fitness_scores = [self.evaluate_fitness(action, observation, info) for action in population]
            selected = self.select_population(population, fitness_scores)
            offspring = self.crossover(selected)
            population = self.mutate(offspring)
        best_action = population[np.argmax(fitness_scores)]
        return best_action

    def random_action(self, observation):
        available_products = [product for product in observation['products'] if product['quantity'] > 0]
        
        selected_product = random.choice(available_products)
        size = selected_product['size']
        
        pos_x, pos_y = None, None
        while True:
            stock_idx = random.randint(0, len(observation['stocks']) - 1)
            stock = observation['stocks'][stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            if (stock_w < size[0]) or (stock_h < size[1]):
                continue
            for i in range(stock_w - size[0] + 1):
                for j in range(stock_h - size[1] + 1):
                    if self._can_place_(stock, (i, j), size):
                        pos_x, pos_y = i, j
                        break
                if pos_x is not None and pos_y is not None:
                    break
            if pos_x is not None and pos_y is not None:
                break
        return {'stock_idx': stock_idx, 'size': size, 'position': (pos_x, pos_y)}

    def evaluate_fitness(self, action, observation, info):
        stock = observation["stocks"][action["stock_idx"]]
        prod_size = action["size"]
        pos_x, pos_y = action["position"]

        stock_w, stock_h = self._get_stock_size_(stock)
        target_x, target_y = stock_w // 2, stock_h // 2
        distance_to_target = ((pos_x - target_x) ** 2 + (pos_y - target_y) ** 2) ** 0.5

        space_utilization = (prod_size[0] * prod_size[1]) / (stock_w * stock_h)

        used_area = np.sum(stock >= 0)

        fitness_score = space_utilization + distance_to_target + used_area ** 2

        if not self._can_place_(stock, (pos_x, pos_y), prod_size):
            fitness_score -= 1000

        return fitness_score
        

    def select_population(self, population, fitness_scores):
        sorted_population = [action for _, action in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]
        return sorted_population[:len(sorted_population)//2]

    def crossover(self, selected):
        offspring = []
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = {
                'stock_idx': parent1['stock_idx'],
                'size': parent2['size'],
                'position': parent1['position']
            }
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        for action in offspring:
            if random.random() < self.mutation_rate:
                size = list(action['size'])
                size[0] += random.randint(-1, 1)  
                size[1] += random.randint(-1, 1)  
                action['size'] = tuple(size[::-1])
        return offspring