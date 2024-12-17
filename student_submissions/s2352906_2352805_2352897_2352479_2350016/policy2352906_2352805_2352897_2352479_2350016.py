from policy import Policy
import random  
import numpy as np


class Policy2352906_2352805_2352897_2352479_2350016(Policy):
    def __init__(self, policy_id=1,
                 population_size=100,
                 mutation_rate=0.05,
                 generations=100,
                 stock_focus_weight=0.90,
                 allow_sticky = True,
                 sorted_products = False,
                 elitism_rate=0.10,
                 pre_firstmove = False):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # Attributes for branch-and-bound
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.stockidx_lastplaced = []
        self.stock_focus_weight = stock_focus_weight
        self.allow_sticky = allow_sticky
        self.sorted_products = sorted_products
        self.elitism_rate = elitism_rate
        self.pre_firstmove = pre_firstmove

    def get_action(self, observation, info):
        """
        Depending on policy_id:
        - Policy 1: Optimized stocks heuristic placement.
        - Policy 2: Genetic search.
        """

        if self.policy_id == 1:
            return self._heuristic_action(observation, info)
        elif self.policy_id == 2:
            if self.pre_firstmove:
                # Fill the first product in the corner of the largest stock
                stock = max(observation["stocks"], key=lambda s: np.sum(s != -2))
                stock_idx = next(i for i, s in enumerate(observation["stocks"]) if np.array_equal(s, stock))
                stock_w, stock_h = self._get_stock_size_(stock)
                prod = observation["products"][0]
                prod_size = prod["size"]
                pos_x = 0
                pos_y = 0
                if stock_w >= prod["size"][0] and stock_h >= prod["size"][1]:
                    pos_x = stock_w - prod["size"][0]
                    pos_y = stock_h - prod["size"][1]
                self.pre_firstmove = False
                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
                
            action = self.genetic_algorithm(observation, info, self.stock_focus_weight)
            while (self.evaluate_fitness(action, observation, info) == 0):
                action = self.genetic_algorithm(observation, info, self.stock_focus_weight / 3)
            self.stockidx_lastplaced.append(action["stock_idx"])
            return action

    def _heuristic_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        # Precompute and cache stock sizes
        cached_stock_sizes = [self._get_stock_size_(stock) for stock in stocks]

        # Sort products in descending order based on area
        sorted_prods = sorted(
            [prod for prod in products if prod["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],
            reverse=True
        )

        best_stock_idx = -1
        best_position = (0, 0)
        best_size = [0, 0]
        best_fit = float('inf')

        for prod in sorted_prods:
            original_size = prod["size"]
            prod_w, prod_h = original_size
            rot_prod_w, rot_prod_h = prod_h, prod_w

            # Sort stocks by available area
            stocks_sorted = sorted(
                enumerate(cached_stock_sizes),
                key=lambda x: (x[1][0] * x[1][1]),
                reverse=True
            )

            placed = False
            for stock_idx, (stock_w, stock_h) in stocks_sorted:
                # Try original orientation
                if stock_w >= prod_w and stock_h >= prod_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stocks[stock_idx], (x, y), [prod_w, prod_h]):
                                remaining_space = (stock_w - prod_w) * (stock_h - prod_h)
                                if remaining_space < best_fit:
                                    best_fit = remaining_space
                                    best_stock_idx = stock_idx
                                    best_position = (x, y)
                                    best_size = [prod_w, prod_h]

                                    if best_fit == 0:
                                        return {
                                            "stock_idx": best_stock_idx,
                                            "size": best_size,
                                            "position": best_position
                                        }
                                placed = True
                                break
                        if placed:
                            break

                # Try rotated orientation
                if not placed and stock_w >= rot_prod_w and stock_h >= rot_prod_h:
                    for x in range(stock_w - rot_prod_w + 1):
                        for y in range(stock_h - rot_prod_h + 1):
                            if self._can_place_(stocks[stock_idx], (x, y), [rot_prod_w, rot_prod_h]):
                                remaining_space = (stock_w - rot_prod_w) * (stock_h - rot_prod_h)
                                if remaining_space < best_fit:
                                    best_fit = remaining_space
                                    best_stock_idx = stock_idx
                                    best_position = (x, y)
                                    best_size = [rot_prod_w, rot_prod_h]

                                    if best_fit == 0:
                                        return {
                                            "stock_idx": best_stock_idx,
                                            "size": best_size,
                                            "position": best_position
                                        }
                                placed = True
                                break
                        if placed:
                            break

                if placed:
                    # Move to next product after placing this one
                    break

        return {
            "stock_idx": best_stock_idx,
            "size": best_size,
            "position": best_position
        }
    def genetic_algorithm(self, observation, info, stock_focus_weight):
        # Initialize the population with random valid placements
        self.stock_focus_weight = stock_focus_weight
        population = self.initialize_population(observation)

        for generation in range(self.generations):  
            # Evaluate fitness of the population
            fitness = [self.evaluate_fitness(individual, observation, info) for individual in population]

            # Select parents based on fitness
            parents = self.select_parents(population, fitness)

            if self.elitism_rate > 0:
                # Preserve the best individuals
                num_elites = max(1, int(self.elitism_rate * self.population_size))
                elites = sorted(population, key=lambda ind: self.evaluate_fitness(ind, observation, info), reverse=True)[:num_elites]

            # Create the next generation via crossover and mutation
            offspring = self.crossover_and_mutate(parents,observation=observation)

            # Replace the old population with the offspring
            population = offspring

            if self.elitism_rate > 0:
                # Ensure the best individuals are carried over to the next generation
                population[:num_elites] = elites

        # Select the best individual from the final population
        best_individual = max(population, key=lambda ind: self.evaluate_fitness(ind, observation, info))
        return best_individual

    def initialize_population(self, observation):
        population = []
        stock_count = len(observation["stocks"])
        for i in range(self.population_size):
            
            
            stock_idx = i % stock_count  # Ensure stocks are filled from 0
            stock = observation["stocks"][stock_idx]

            # if self.sorted_products: choose larger products first
            prod = random.choice([p for p in observation["products"] if p["quantity"] > 0])
            if self.sorted_products:
                products = self.sort_products(observation)
                # prod = highest size product
                prod = products[0]
            prod_size = prod["size"]

            stock_w, stock_h = self._get_stock_size_(stock)
            pos_x = random.randint(0, stock_w - prod_size[0])
            pos_y = random.randint(0, stock_h - prod_size[1])
            
            # 50% chance to be rotated (size from [w, h] to [h, w])
            if random.random() < 0.5:
                prod_size = prod_size[::-1]
                

            population.append({"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)})
        return population

    def evaluate_fitness(self, individual, observation, info):
        stock_idx = individual["stock_idx"]
        position = individual["position"]
        prod_size = individual["size"]

        stock = observation["stocks"][stock_idx]
        if self._can_place_(stock, position, prod_size):
            if (position[0] + prod_size[0] > self._get_stock_size_(stock)[0]) or (position[1] + prod_size[1] > self._get_stock_size_(stock)[1]):
                return 0
            # Trim loss
            bonus = 0
            if self.allow_sticky:
                if position[0] == 0 or position[0] + prod_size[0] == self._get_stock_size_(stock)[0]:
                    bonus += 0.01
                    
                if position[1] == 0 or position[1] + prod_size[1] == self._get_stock_size_(stock)[1]:
                    bonus += 0.01
                
                if position[1] < 100 and position[0] < 100 and position[0] > 0 and stock[position[0] - 1][position[1]] != -1:
                    bonus += 0.02
                    
                if position[1] < 100 and position[0] < 100 and position[1] > 0 and stock[position[0]][position[1] - 1] != -1:
                    bonus += 0.02

                if position[1] > 0 and position[0] > 0 and position[0] < 100 and position[1] < 100 and stock[position[0] - 1][position[1] - 1] != -1:
                    bonus += 0.02

                if position[1] > 0 and position[0] < 100 and position[1] < 100 and stock[position[0] + 1][position[1] - 1] != -1:
                    bonus += 0.02

                # North west not facing any -1
                if position[0] > 0 and position[1] > 0 and stock[position[0] - 1][position[1] - 1] != -1:
                    bonus += 0.01

                # North east not facing any -1
                if position[0] < 100 and position[1] > 0 and stock[position[0] + 1][position[1] - 1] != -1:
                    bonus += 0.01

                # South west not facing any -1
                if position[0] > 0 and position[1] < 100 and stock[position[0] - 1][position[1] + 1] != -1:
                    bonus += 0.01

                # South east not facing any -1
                if position[0] < 100 and position[1] < 100 and stock[position[0] + 1][position[1] + 1] != -1:
                    bonus += 0.01
                    
            count = 0
            total_size = 0
            set_stockidx_lastplaced = set(self.stockidx_lastplaced)
            for id in set_stockidx_lastplaced:
                temp_stock = observation["stocks"][id]
                count += np.sum(temp_stock == -1)
                total_size += self._get_stock_size_(temp_stock)[0] * self._get_stock_size_(temp_stock)[1]

            if stock_idx not in set_stockidx_lastplaced:
                temp_stock = observation["stocks"][stock_idx]
                count += np.sum(temp_stock == -1)
                total_size += self._get_stock_size_(temp_stock)[0] * self._get_stock_size_(temp_stock)[1]

            return 1 - count / total_size + bonus + (100 - set_stockidx_lastplaced.__len__())
        return 0  # Invalid placements have zero fitness

    def select_parents(self, population, fitness):
        # Select individuals with probability proportional to their fitness (roulette wheel selection)
        total_fitness = sum(fitness)
        if total_fitness == 0:
            return random.sample(population, k=len(population))  # Random selection if all fitness is zero
        probabilities = [f / total_fitness for f in fitness]
        parents = random.choices(population, probabilities, k=self.population_size)
        return parents

    def crossover_and_mutate(self, parents, observation=None):
        offspring = []
        for _ in range(0, len(parents), 2):
            # Select two parents for crossover
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            # Crossover: Create child by averaging parent positions
            child = {
                "stock_idx": parent1["stock_idx"],
                "size": parent1["size"],
                "position": (
                    (parent1["position"][0] + parent2["position"][0]) // 2,
                    (parent1["position"][1] + parent2["position"][1]) // 2,
                ),
            }

            # Mutation: Randomly adjust the position or change the stock index
            if random.random() < self.mutation_rate:
                # if random.random() < 0.5:
                    # Adjust the position
                stock_w, stock_h = self._get_stock_size_(observation["stocks"][child["stock_idx"]])
                child["position"] = (
                    max(0, min(stock_w - child["size"][0], child["position"][0] + random.randint(-1, 1))),
                    max(0, min(stock_h - child["size"][1], child["position"][1] + random.randint(-1, 1))),
                )
                # else:
                #     # Change the stock index
                #     child["stock_idx"] = random.randint(0, len(observation["stocks"]) - 1)
                #     # If can't place in new stock, revert to old stock
                #     if not self._can_place_(observation["stocks"][child["stock_idx"]], child["position"], child["size"]):
                #         child["stock_idx"] = parent1["stock_idx"]

                # Chance to switch to another stock that is in lastplaced
                if random.random() < self.stock_focus_weight and len(self.stockidx_lastplaced) > 0:
                    child["stock_idx"] = random.choice(self.stockidx_lastplaced)
                    # If can't place in new stock, revert to old stock
                    if not self._can_place_(observation["stocks"][child["stock_idx"]], child["position"], child["size"]):
                        child["stock_idx"] = parent1["stock_idx"]

            offspring.append(child)
        return offspring
    
    def sort_products(self, observation):
        products = observation["products"]
        # remove every product with quantity 0
        products = [p for p in products if p["quantity"] > 0]
        return sorted(products, key=lambda x: x["size"][0] * x["size"][1], reverse=True)