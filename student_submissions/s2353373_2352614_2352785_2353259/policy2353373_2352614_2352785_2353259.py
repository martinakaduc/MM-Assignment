from policy import Policy
import random
import copy
import numpy as np
class Policy2353373_2352614_2352785_2353259(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = FirstFitDecreasingPolicy()
        elif policy_id == 2:
            self.policy = GeneticPolicy()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

class FirstFitDecreasingPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        sorted_prods = self._sort_products_by_decreasing_area(list_prods)
        for prod in sorted_prods:
            prod_size = prod["size"]
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                action = self._attempt_place_product(stock, stock_w, stock_h, prod_size)
                if action:
                    action["stock_idx"] = i
                    return action
                rotated_size = self._rotate_size(prod_size)
                action = self._attempt_place_product(stock, stock_w, stock_h, rotated_size)
                if action:
                    action["stock_idx"] = i
                    action["size"] = rotated_size
                    return action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _sort_products_by_decreasing_area(self, products):
        sorted_prods = sorted(
            [prod for prod in products if prod["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],
            reverse=True
        )
        return sorted_prods

    def _attempt_place_product(self, stock, stock_w, stock_h, prod_size):
        prod_w, prod_h = prod_size
        if stock_w >= prod_w and stock_h >= prod_h:
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), prod_size):
                        return {
                            "size": prod_size,
                            "position": (x, y)
                        }
        return None

    def _rotate_size(self, size):
        return [size[1], size[0]]
    
class GeneticPolicy(Policy):
    def __init__(self):
        self.products = []
        self.num_stocks = 1
        self.population = []
        self.population_size = 20  
        self.generations = 100  
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1
        self.elite_size = 2

        self.counter = 0
        self.best_individual = None
        self.actions = []
        self.stocks = []

    def get_action(self, observation, info):
        if self.counter == 0:
            self._initialize_products_and_stocks(observation)

            if not self.products:
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

            if not self.population or len(self.population[0]) != len(self.products):
                self.initialize_population()

            self.best_individual = self.run_ga()
            self.counter += 1

            # Evaluate the best individual
            fitness = self.evaluate_fitness(self.best_individual)
            total_area = sum(stock.shape[0] * stock.shape[1] for stock in self.stocks)
            if fitness[0] >= total_area:
                # GA failed to place any products
                raise GAPlacementFailure("Genetic Algorithm failed to place any products.")

            for gene in self.best_individual:
                stock_idx, row_idx, col_idx, prod_idx = gene
                if prod_idx >= len(self.products):
                    continue
                num_rows, num_cols = self.products[prod_idx]
                self.actions.append({
                    "stock_idx": stock_idx,
                    "size": [num_rows, num_cols],
                    "position": (row_idx, col_idx)
                })

            if self.actions:
                return self.actions[self.counter - 1]
            else:
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        else:
            self.counter += 1
            if self.counter - 1 < len(self.actions):
                return self.actions[self.counter - 1]
            else:
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _initialize_products_and_stocks(self, observation):
        self.products = []
        temp_products = observation.get("products", [])
        for index, cur_product in enumerate(temp_products):
            quantity = cur_product.get("quantity", 0)
            size = cur_product.get("size")
            if size is None or len(size) != 2:
                continue
            num_rows, num_cols = size

            for _ in range(quantity):
                self.products.append((num_rows, num_cols))

        self.stocks = observation.get("stocks", [])
        self.num_stocks = len(self.stocks)

    def initialize_population(self):
        self.population = []
        feasible_positions = self._get_feasible_positions()

        for _ in range(self.population_size):
            individual = []
            temp_stocks = [stock.copy() for stock in self.stocks]
            for prod_idx in range(len(self.products)):
                possible_positions = [
                    (item['stock_idx'], pos)
                    for item in feasible_positions
                    if item['prod_idx'] == prod_idx and item['positions']
                    for pos in item['positions']
                ]

                possible_positions = [
                    (stock_idx, pos) for stock_idx, pos in possible_positions
                    if self._is_position_feasible(temp_stocks[stock_idx], pos, self.products[prod_idx])
                ]

                if not possible_positions:
                    continue

                stock_idx, (row_idx, col_idx) = random.choice(possible_positions)
                individual.append((stock_idx, row_idx, col_idx, prod_idx))

                prod_num_rows, prod_num_cols = self.products[prod_idx]
                temp_stocks[stock_idx][row_idx:row_idx + prod_num_rows, col_idx:col_idx + prod_num_cols] = -2  # Mark the used area

            self.population.append(individual)

    def _get_feasible_positions(self):
        feasible_positions = []
        for stock_idx in range(self.num_stocks):
            temp_stock = self.stocks[stock_idx]
            num_row_stock, num_col_stock = self._get_stock_size_(temp_stock)
            stock = temp_stock[:num_row_stock, :num_col_stock]
            stock_shape = stock.shape
            stock_num_rows, stock_num_cols = stock_shape

            for prod_idx, (prod_num_rows, prod_num_cols) in enumerate(self.products):
                positions = []
                for row_idx in range(stock_num_rows - prod_num_rows + 1):
                    for col_idx in range(stock_num_cols - prod_num_cols + 1):
                        if self._can_place_(stock, (row_idx, col_idx), (prod_num_rows, prod_num_cols)):
                            positions.append((row_idx, col_idx))

                feasible_positions.append({
                    'stock_idx': stock_idx,
                    'prod_idx': prod_idx,
                    'positions': positions
                })

        return feasible_positions

    def _is_position_feasible(self, stock, position, prod_size):
        row_idx, col_idx = position
        prod_num_rows, prod_num_cols = prod_size
        region = stock[row_idx:row_idx + prod_num_rows, col_idx:col_idx + prod_num_cols]
        return np.all(region == -1)

    def evaluate_fitness(self, individual):
        waste = 0
        used_stocks = set()
        stock_states = [stock.copy() for stock in self.stocks]  # Copy stock states
        total_area = sum(stock.shape[0] * stock.shape[1] for stock in self.stocks)
        used_area = 0
        for gene in individual:
            stock_idx, row_idx, col_idx, prod_idx = gene
            if prod_idx >= len(self.products):
                waste += 1000
                continue
            prod_num_rows, prod_num_cols = self.products[prod_idx]
            stock = stock_states[stock_idx]
            stock_num_rows, stock_num_cols = stock.shape

            if row_idx + prod_num_rows > stock_num_rows or col_idx + prod_num_cols > stock_num_cols:
                waste += prod_num_rows * prod_num_cols
                continue

            region = stock[row_idx:row_idx + prod_num_rows, col_idx:col_idx + prod_num_cols]
            if np.any(region != -1):
                waste += prod_num_rows * prod_num_cols
                continue

            stock[row_idx:row_idx + prod_num_rows, col_idx:col_idx + prod_num_cols] = prod_idx
            used_stocks.add(stock_idx)
            used_area += prod_num_rows * prod_num_cols

        waste = total_area - used_area
        return (waste, len(used_stocks))

    def tournament_selection(self, population, k=3):
        selected = []
        for _ in range(len(population)):
            aspirants = random.sample(population, min(k, len(population)))
            aspirants = sorted(aspirants, key=lambda ind: self.evaluate_fitness(ind))
            selected.append(copy.deepcopy(aspirants[0]))
        return selected

    def crossover(self, parent1, parent2):
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)

        if random.random() < self.crossover_rate and len(parent1) > 1:
            crossover_point = random.randint(1, len(parent1) - 1)
            child1[:crossover_point], child2[:crossover_point] = parent2[:crossover_point], parent1[:crossover_point]

        return child1, child2

    def mutate(self, individual):
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                stock_idx = random.randint(0, self.num_stocks - 1)
                stock_num_rows, stock_num_cols = self._get_stock_size_(self.stocks[stock_idx])
                prod_idx = individual[i][3]
                prod_num_rows, prod_num_cols = self.products[prod_idx]

                if stock_num_rows >= prod_num_rows and stock_num_cols >= prod_num_cols:
                    max_row_idx = stock_num_rows - prod_num_rows
                    max_col_idx = stock_num_cols - prod_num_cols
                    row_idx = random.randint(0, max_row_idx)
                    col_idx = random.randint(0, max_col_idx)
                    individual[i] = (stock_idx, row_idx, col_idx, prod_idx)
        return individual

    def run_ga(self):
        for generation in range(self.generations):
            fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
            sorted_population = sorted(zip(self.population, fitness_scores), key=lambda x: (x[1][0], x[1][1]))
            elite_individuals = [copy.deepcopy(individual) for individual, _ in sorted_population[:self.elite_size]]

            selected = self.tournament_selection(self.population, k=3)
            children = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self.crossover(selected[i], selected[i + 1])
                    children.extend([child1, child2])
                else:
                    children.append(copy.deepcopy(selected[i]))

            mutated_children = [self.mutate(child) for child in children]

            self.population = mutated_children

            fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
            for elite in elite_individuals:
                worst_fitness = max(fitness_scores, key=lambda x: (x[0], x[1]))
                worst_index = fitness_scores.index(worst_fitness)
                self.population[worst_index] = copy.deepcopy(elite)
                fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]

        fitness_scores = [self.evaluate_fitness(ind) for ind in self.population]
        best_fitness = min(fitness_scores, key=lambda x: (x[0], x[1]))
        best_individual = self.population[fitness_scores.index(best_fitness)]
        return best_individual
