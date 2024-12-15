# import sys
# sys.path.append(r'C:\Users\huynh\MM241-Assignment')

from policy import Policy
import numpy as np
import random


class Policy2311147_2310615_2311071_2311012_2311142(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = GurelPolicy()
        elif policy_id == 2:
            self.policy = GeneticPolicy()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed

class GurelPolicy(Policy):
    def __init__(self, a=0.4, b=0.25, c=0.1):
        super().__init__()
        self.current_stock_idx = None
        self.last_position = None
        self.a = a
        self.b = b
        self.c = c

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        # 1. Classify products into groups
        P_max = max([stock.shape[0] * stock.shape[1] for stock in stocks])
        groups = {"L1": [], "L2": [], "M": [], "S": []}
        
        for idx, product in enumerate(products):
            if product["quantity"] > 0:
                area = product["size"][0] * product["size"][1]
                if area >= self.a * P_max:
                    groups["L1"].append(idx)
                elif area >= self.b * P_max:
                    groups["L2"].append(idx)
                elif area >= self.c * P_max:
                    groups["M"].append(idx)
                else:
                    groups["S"].append(idx)

        # 2. Place products (priority: L1 -> L2 -> M -> S)
        for group in ["L1", "L2", "M", "S"]:
            for product_idx in groups[group]:
                product = products[product_idx]
                prod_w, prod_h = product["size"]

                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Check if the product can fit in the stock
                    if stock_w >= prod_w and stock_h >= prod_h:
                        position = self._find_best_position(stock, (prod_w, prod_h))
                        if position:
                            return {
                                "stock_idx": stock_idx,
                                "size": (prod_w, prod_h),
                                "position": position,
                            }

                    # Rotate product 90 degrees
                    if stock_w >= prod_h and stock_h >= prod_w:
                        position = self._find_best_position(stock, (prod_h, prod_w))
                        if position:
                            return {
                                "stock_idx": stock_idx,
                                "size": (prod_h, prod_w),
                                "position": position,
                            }

        # No valid position found
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _find_best_position(self, stock, prod_size):
        prod_w, prod_h = prod_size
        stock_w, stock_h = stock.shape
        best_position = None
        min_waste = float("inf")

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    waste = self._calculate_waste(stock, (x, y), prod_size)
                    if waste < min_waste:
                        min_waste = waste
                        best_position = (x, y)

        return best_position

    def _calculate_waste(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        # Calculate waste around the product
        waste_left = np.sum(stock[pos_x:pos_x + prod_w, :pos_y] != -2)
        waste_right = np.sum(stock[pos_x:pos_x + prod_w, pos_y + prod_h:] != -2)
        waste_top = np.sum(stock[:pos_x, pos_y:pos_y + prod_h] != -2)
        waste_bottom = np.sum(stock[pos_x + prod_w:, pos_y:pos_y + prod_h] != -2)

        total_waste = waste_left + waste_right + waste_top + waste_bottom
        return total_waste
    
    

class GeneticPolicy(Policy):
    def __init__(self, generations=500, population_size=100, penalty_factor=2, mutation_probability=0.1):
        self.generations = generations
        self.population_size = population_size
        self.penalty_factor = penalty_factor
        self.mutation_probability = mutation_probability

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]
        if not stocks or not list_prods:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
        # Calculate stock sizes and create index-size pairs
        stock_sizes = []
        for idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_sizes.append((idx, stock_w * stock_h))
        
        # Sort stocks by size in descending order
        stock_sizes.sort(key=lambda x: x[1], reverse=True)
        sorted_stock_indices = [idx for idx, _ in stock_sizes]
        
        self.h_prods = [prod["size"][0] for prod in list_prods if prod["quantity"] > 0]
        self.w_prods = [prod["size"][1] for prod in list_prods if prod["quantity"] > 0]
        self.demand_prods = [prod["quantity"] for prod in list_prods if prod["quantity"] > 0]
        self.products = len(self.demand_prods)
        
        if self.products == 0:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
        # Use the largest stock for cut generation
        largest_stock = stocks[sorted_stock_indices[0]]
        self.stock_h, self.stock_w = self._get_stock_size_(largest_stock)
        
        cuts = self.generate_optimal_cuts()
        max_repeat_arr = self.calculate_max_cut_repetition(cuts)
        population = self.initialize_population(max_repeat_arr)
        best_solution, _, _ = self.run(population, cuts, max_repeat_arr)
        
        for i in range(0, len(best_solution), 2):
            cut_index = best_solution[i]
            if cut_index >= len(self.h_prods):
                continue
                
            prod_size = (self.h_prods[cut_index], self.w_prods[cut_index])
            
            # Try placing in each stock, starting from the largest
            for stock_idx in sorted_stock_indices:
                stock = stocks[stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                
                # Check if the product can fit in this stock
                if prod_size[0] <= stock_w and prod_size[1] <= stock_h:
                    # Try to find a valid position
                    for x in range(stock_w):
                        for y in range(stock_h):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y)
                                }
        
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def generate_optimal_cuts(self):
        cuts = []

        def backtrack(current_cut, h_used, w_used, index):
            if index >= self.products:
                return
            max_repeat = min(
                (self.stock_h - h_used) // self.h_prods[index],
                (self.stock_w - w_used) // self.w_prods[index],
                self.demand_prods[index]
            )
            for repeat in range(1, max_repeat + 1):
                new_cut = current_cut.copy()
                new_cut[index] += repeat
                cuts.append(new_cut)
                backtrack(
                    new_cut,
                    h_used + repeat * self.h_prods[index],
                    w_used + repeat * self.w_prods[index],
                    index + 1
                )

            backtrack(current_cut, h_used, w_used, index + 1)

        # Recursively generate all optimal cuts
        backtrack([0] * self.products, 0, 0, 0)
        return cuts
    
    def calculate_max_cut_repetition(self, cuts):
        max_repeats = []
        for cut in cuts:
            max_repeat_for_cut = float('inf')
            for i, quantity in enumerate(cut):
                if quantity > 0:
                    max_repeat_for_cut = min(
                        max_repeat_for_cut,
                        self.demand_prods[i] // quantity,
                        (self.stock_h // self.h_prods[i]) * (self.stock_w // self.w_prods[i])
                    )
            max_repeats.append(max_repeat_for_cut)
        return max_repeats

    def initialize_population(self, max_repeat_arr):
        init_population = []
        for _ in range(self.population_size):
            genome = []
            for i in np.argsort(-np.array(self.h_prods) * np.array(self.w_prods)):
                genome.append(i)
                genome.append(np.random.randint(1, max_repeat_arr[i] + 1))
            init_population.append(genome)
        return init_population

    def evaluate_fitness(self, genome, cuts):
        if self.stock_h <= 0 or self.stock_w <= 0:
            raise ValueError("Invalid stock dimensions")
        if len(genome) % 2 != 0:
            raise ValueError("Invalid genome")
        if len(cuts) == 0:
            raise ValueError("Empty cuts")

        pen = self.penalty_factor
        stock_area = self.stock_h * self.stock_w
        total_short_demand = 0
        total_unused_area = 0
        provided = [0] * self.products

        for i in range(0, len(genome), 2):
            cut_index = genome[i]
            repetition = genome[i + 1]

            if cut_index >= len(cuts):
                continue
            cut = cuts[cut_index]
            for j, qty in enumerate(cut):
                provided[j] += qty * repetition

            cut_area = sum(cut[j] * self.h_prods[j] * self.w_prods[j] for j in range(len(cut)))
            total_unused_area += stock_area - (cut_area * repetition)

        for i in range(self.products):
            short_demand = max(0, self.demand_prods[i] - provided[i])
            total_short_demand += short_demand * self.h_prods[i] * self.w_prods[i]
        fitness = (
            1 * (1 - total_unused_area / stock_area)
            - 0.3 * (pen * total_short_demand / sum(self.demand_prods))
        )
        return fitness

    def run(self, population, cuts, max_repeat_arr):
        best_results = []
        num_iters_same_result = 0
        last_result = float('inf')

        for i in range(self.generations):
            fitness_pairs = [(genome, self.evaluate_fitness(genome, cuts)) for genome in population]
            fitness_pairs.sort(key=lambda x: x[1], reverse=True)
            best_solution, best_fitness = fitness_pairs[0]
            best_results.append(best_fitness)

            if abs(best_fitness - last_result) < 1e-6:
                num_iters_same_result += 1
            else:
                num_iters_same_result = 0
            last_result = best_fitness

            if num_iters_same_result >= 50 or best_fitness == 1:
                break

            next_gen = [fitness_pairs[i][0] for i in range(2)]

            while len(next_gen) < self.population_size and len(fitness_pairs) > 1:
                prev_gen = [fp[0] for fp in fitness_pairs]
                prev_fitness = [fp[1] for fp in fitness_pairs]
                parent1 = self.select_parents(prev_gen, prev_fitness)
                parent2 = self.select_parents(prev_gen, prev_fitness)
                child1 = self.mutate(self.crossover(parent1, parent2), self.mutation_probability, max_repeat_arr)
                child2 = self.mutate(self.crossover(parent2, parent1), self.mutation_probability, max_repeat_arr)
                next_gen.extend([child1, child2])
            population = [x[:] for x in next_gen[:self.population_size]]

        return best_solution, best_fitness, best_results


    def select_parents(self, population, fitness_scores, tournament_size=5):
        tournament_size = min(tournament_size, len(population))
        indices = np.random.choice(len(population), tournament_size)
        tournament = [population[i] for i in indices]
        tournament_scores = [fitness_scores[i] for i in indices]
        best_index = np.argmax(tournament_scores)
        return tournament[best_index]

    def crossover(self, parent1, parent2):
        if parent1 is None or parent2 is None:
            raise ValueError("Need 2 parents")
        return [p1 if np.random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]

    def mutate(self, genome, mutation_rate, max_repeat_arr):
        mutated_genome = genome.copy()
        for i in range(0, len(genome), 2):
            if np.random.random() < mutation_rate and i + 1 < len(genome):
                cut_index = mutated_genome[i]
                mutated_genome[i + 1] = np.random.randint(1, max_repeat_arr[cut_index] + 1)
        return mutated_genome
    


    def _can_place_(self, stock, position, prod_size):
        # Sử dụng logic bạn đã viết
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        if pos_x + prod_w > stock.shape[0] or pos_y + prod_h > stock.shape[1]:
            return False

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)

