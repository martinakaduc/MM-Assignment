from policy import Policy
import random
import numpy as np


class Policy2313955_2312944_2311202_2311593_2313593(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy_id = 1
        elif policy_id == 2:
            self.population_size = 10
            self.generations = 20
            self.mutation_rate = 0.1
            self.curr_stock = -1
            self.best_solution = None
            self.quantity = 0
            self.policy_id = 2

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self.FFD(observation)
        else:
            if self.quantity == 0:
                for prod in observation["products"]:
                    self.quantity += prod["quantity"]
                self.best_solution = None
                self.current_index = 0
            if self.best_solution is None or self.current_index >= len(self.best_solution):
                self.evolve(observation)
            prod = self.best_solution[self.current_index]
            self.current_index += 1
            self.quantity -=1
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod["size"]
                if stock_w >= prod_w and stock_h >= prod_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod["size"]):
                                return {"stock_idx": stock_idx, "size": prod["size"], "position": (x, y)}
                if stock_w >= prod_h and stock_h >= prod_w:
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod["size"][::-1]):
                                return {"stock_idx": stock_idx, "size": prod["size"][::-1], "position": (x, y)}
            return {"stock_idx": -1, "size": prod["size"], "position": (None, None)}

    # Student code here
    # You can add more functions if needed

    def FFD(self, observation):
        products = observation["products"]
        stocks = observation["stocks"]
        products = sorted(products, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)
        for prod in products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                placed = False
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for i in range(stock_w - prod_w + 1):
                            for j in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (i, j), prod_size):
                                    placed = True
                                    return {"stock_idx": stock_idx, "size": prod_size, "position": (i, j)}
                    if not placed and stock_w >= prod_h and stock_h >= prod_w:
                        for i in range(stock_w - prod_h + 1):
                            for j in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (i, j), prod_size[::-1]):
                                    placed = True
                                    return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (i, j)}
                    if placed:
                        break
        return None

    def initialize_population(self, observation):
        quantities = [prod["quantity"] for prod in observation["products"]]
        list_prod = observation["products"]
        population = []
        for _ in range(self.population_size):
            individual = random.choices(list_prod, weights = quantities, k = 5)
            population.append(individual)
        return population
    
    def fitness(self, individual, observation):
        total_waste = 0
        stocks = observation["stocks"]
        for prod in individual:
            used_area = 0
            if prod["quantity"] > 0:
                prod_w, prod_h = prod["size"]
                for stock in stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    if stock_w >= prod_w and stock_h >= prod_h:
                        used_area += prod_w * prod_h
                        total_area = stock_w * stock_h
                        waste = total_area - used_area
                        total_waste += waste
                        break
                    elif stock_w >= prod_h and stock_h >= prod_w:
                        used_area += prod_w * prod_h
                        total_area = stock_w * stock_h
                        waste = total_area - used_area
                        total_waste += waste
                        break
        return 1 / (1 + total_waste)
    
    def select_parents(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [f / total_fitness for f in fitness_scores]
        cumulative_probs = np.cumsum(probabilities)
        rand1 = random.random()
        parent1 = None
        for i, cp in enumerate(cumulative_probs):
            if rand1 <= cp:
                parent1 = population[i]
                break
        rand2 = random.random()
        parent2 = None
        for i, cp in enumerate(cumulative_probs):
            if rand2 <= cp:
                parent2 = population[i]
                break
        return parent1, parent2
    
    def crossover(self, parent1, parent2):
        if len(parent1) <= 1 or len(parent2) <= 1:
            return parent1, parent2
        cut = random.randint(1, len(parent1) - 1)
        child1 = parent1[:cut] + [p for p in parent2 if tuple(p["size"]) not in [tuple(x["size"]) for x in parent1[:cut]]]
        child2 = parent2[:cut] + [p for p in parent1 if tuple(p["size"]) not in [tuple(x["size"]) for x in parent2[:cut]]]
        return child1, child2
    
    def mutation(self, individual):
        if random.random() < self.mutation_rate and len(individual) >= 2:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]

    def evolve(self, observation):
        population = self.initialize_population(observation)
        for _ in range(self.generations):
            fitness_scores = [self.fitness(ind, observation) for ind in population]
            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                self.mutation(child1)
                self.mutation(child2)
                new_population.extend([child1, child2])
            population = new_population
        fitness_scores = [self.fitness(ind, observation) for ind in population]
        self.best_solution = population[fitness_scores.index(max(fitness_scores))]
        self.current_index = 0
