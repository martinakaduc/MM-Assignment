from policy import Policy
import numpy as np
import random


class Policy2312961_2312817_2112585_2310785_2311615(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy = None
        if policy_id == 1:
            self.policy = GeneticAlgorithmPolicy()
        elif policy_id == 2:
            self.policy = DynamicProgrammingPolicy()

    def get_action(self, observation, info):
        # Student code here
        if self.policy is not None:
            return self.policy.get_action(observation, info)
        else:
            raise NotImplementedError("Policy is not implemented yet.")


class DynamicProgrammingPolicy(Policy):
    def __init__(self, min_product_size=1):
        # Initialize the policy with a minimum product size threshold
        self.min_product_size = min_product_size

    def get_action(self, observation, info):
        # Extract stock and product information
        stocks = observation['stocks']  # List of 2D arrays representing stock layouts
        products = observation['products']  # List of products with 'size' and 'quantity'
        
        # Step 1: Sort products by area in descending order for greedy placement
        sorted_products = sorted(
            [p for p in products if p['quantity'] > 0 and p['size'][0] * p['size'][1] >= self.min_product_size],
            key=lambda x: x['size'][0] * x['size'][1],
            reverse=True
        )
        
        # Step 2: Iterate through products and attempt to place them in stocks
        for product in sorted_products:
            product_height, product_width = product['size']
            demand = product['quantity']
            
            # Step 3: Iterate through each stock
            for stock_idx, stock in enumerate(stocks):
                stock_height, stock_width = stock.shape
                
                # Initialize DP table for this stock
                dp = np.zeros((stock_height + 1, stock_width + 1), dtype=bool)
                dp[0][0] = True  # Base case
                
                # Build DP table
                for i in range(stock_height):
                    for j in range(stock_width):
                        if stock[i, j] == -1:  # Only consider empty spaces
                            if i > 0:
                                dp[i][j] = dp[i][j] or dp[i - 1][j]
                            if j > 0:
                                dp[i][j] = dp[i][j] or dp[i][j - 1]
                
                # Step 4: Check placement with DP table
                for rotation in [(product_height, product_width), (product_width, product_height)]:
                    rotated_height, rotated_width = rotation
                    
                    for x in range(stock_height - rotated_height + 1):
                        for y in range(stock_width - rotated_width + 1):
                            # Check if the product can fit
                            if self._can_place(stock, (x, y), (rotated_height, rotated_width)):
                                # Place the product
                                self._place_product(stock, (x, y), (rotated_height, rotated_width))
                                # Update product demand
                                demand -= 1
                                # Return action details
                                action = {
                                    "stock_idx": stock_idx,
                                    "size": (rotated_height, rotated_width),
                                    "position": (x, y),
                                }
                                return action
        
        # Step 5: Return default action if no valid placement found
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _can_place(self, stock, position, size):
        # Check if a product of a given size can be placed at a position in a stock.
        x, y = position
        height, width = size
        stock_height, stock_width = stock.shape
        
        # Ensure within bounds
        if x + height > stock_height or y + width > stock_width:
            return False
        
        # Ensure the placement area is empty
        for i in range(height):
            for j in range(width):
                if stock[x + i, y + j] != -1:
                    return False
        return True

    def _place_product(self, stock, position, size):
        # Place a product in the stock by marking its area.
        x, y = position
        height, width = size
        
        for i in range(height):
            for j in range(width):
                stock[x + i, y + j] = 1



class GeneticAlgorithmPolicy(Policy):
    def __init__(self, generations=500, population_size=100, penalty_factor=2, mutation_probability=0.1):
        self.generations = generations
        self.population_size = population_size
        self.penalty_factor = penalty_factor
        self.mutation_probability = mutation_probability

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]
        
        
        if not stocks or not products:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        stock_sizes = [(idx, stock.shape[0] * stock.shape[1]) for idx, stock in enumerate(stocks)]
        stock_sizes.sort(key=lambda x: x[1], reverse=True)
        sorted_stock_indices = [idx for idx, _ in stock_sizes]
        product_heights = [prod["size"][0] for prod in products if prod["quantity"] > 0]
        product_widths = [prod["size"][1] for prod in products if prod["quantity"] > 0]
        product_quantities = [prod["quantity"] for prod in products if prod["quantity"] > 0]
        num_products = len(product_quantities)

        if num_products == 0:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        largest_stock = stocks[sorted_stock_indices[0]]
        
        stock_height, stock_width = largest_stock.shape

        cuts = []
        def generate_cuts(current_cut, height_used, width_used, index):
            if index >= num_products:
                return
            max_possible_repeats = min(
                (stock_height - height_used) // product_heights[index],
                (stock_width - width_used) // product_widths[index],
                product_quantities[index]
            )
            for repeat in range(1, max_possible_repeats + 1):
                new_cut = current_cut.copy()
                new_cut[index] += repeat
                cuts.append(new_cut)
                generate_cuts(
                    new_cut,
                    height_used + repeat * product_heights[index],
                    width_used + repeat * product_widths[index],
                    index + 1
                )
            generate_cuts(current_cut, height_used, width_used, index + 1)
        generate_cuts([0] * num_products, 0, 0, 0)

        max_repeats_per_cut = []
        for cut in cuts:
            max_repeats = float('inf')
            for i, quantity in enumerate(cut):
                if quantity > 0:
                    max_repeats = min(
                        max_repeats,
                        product_quantities[i] // quantity,
                        (stock_height // product_heights[i]) * (stock_width // product_widths[i])
                    )
            max_repeats_per_cut.append(max_repeats)

        population = []
        for _ in range(self.population_size):
            genome = []
            for i in np.argsort(-np.array(product_heights) * np.array(product_widths)):
                genome.append(i)
                genome.append(np.random.randint(1, max_repeats_per_cut[i] + 1))
            population.append(genome)

        best_solution = None
        best_fit = float('-inf')
        same_result = 0
        last_result = float('inf')

        for _ in range(self.generations):
            fit_pair_list = []
            for genome in population:
                penalty = self.penalty_factor
                stock_area = stock_height * stock_width
                total_short_demand = 0
                total_unused_area = 0
                provided_quantities = [0] * num_products

                for i in range(0, len(genome), 2):
                    cut_index = genome[i]
                    repetition = genome[i + 1]

                    if cut_index >= len(cuts):
                        continue
                    cut = cuts[cut_index]
                    for j, qty in enumerate(cut):
                        provided_quantities[j] += qty * repetition

                    cut_area = sum(cut[j] * product_heights[j] * product_widths[j] for j in range(len(cut)))
                    total_unused_area += stock_area - (cut_area * repetition)

                for i in range(num_products):
                    short_demand = max(0, product_quantities[i] - provided_quantities[i])
                    total_short_demand += short_demand * product_heights[i] * product_widths[i]
                fitness = (1 * (1 - total_unused_area / stock_area) - 0.3 * (penalty * total_short_demand / sum(product_quantities)))
                fit_pair_list.append((genome, fitness))

            fit_pair_list.sort(key=lambda x: x[1], reverse=True)
            best_solution, best_fit = fit_pair_list[0]


            if abs(best_fit - last_result) < 1e-6:
                same_result += 1
            else:
                same_result = 0
            last_result = best_fit

            if same_result >= 50 or best_fit == 1:
                break

            next_gen = [fit_pair_list[i][0] for i in range(2)]

            while len(next_gen) < self.population_size and len(fit_pair_list) > 1:
                prev_gen = [fp[0] for fp in fit_pair_list]
                prev_fitness = [fp[1] for fp in fit_pair_list]
                parent1 = self.choose_parent(prev_gen, prev_fitness)
                parent2 = self.choose_parent(prev_gen, prev_fitness)
                
                child1 = [p1 if np.random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]
                child2 = [p2 if np.random.random() < 0.5 else p1 for p1, p2 in zip(parent1, parent2)]
                
                child1 = self.mutate(child1, self.mutation_probability, max_repeats_per_cut)
                child2 = self.mutate(child2, self.mutation_probability, max_repeats_per_cut)
                next_gen.extend([child1, child2])
            population = [x[:] for x in next_gen[:self.population_size]]

        for i in range(0, len(best_solution), 2):
            cut_index = best_solution[i]
            if cut_index >= len(product_heights):
                continue

            prod_size = (product_heights[cut_index], product_widths[cut_index])

            for stock_idx in sorted_stock_indices:
                stock = stocks[stock_idx]
                stock_w, stock_h = stock.shape

                if prod_size[0] <= stock_w and prod_size[1] <= stock_h:
                    for x in range(stock_w):
                        for y in range(stock_h):
                            if self._can_place(stock, (x, y), prod_size):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y)
                                }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def choose_parent(self, population, fitness_scores, tournament_size=5):
        tournament_size = min(tournament_size, len(population))
        indices = np.random.choice(len(population), tournament_size)
        tournament = [population[i] for i in indices]
        tournament_scores = [fitness_scores[i] for i in indices]
        index = np.argmax(tournament_scores)
        return tournament[index]

    def mutate(self, genome, mutation_rate, max_repeats_per_cut):
        mutated_genome = genome.copy()
        for i in range(0, len(genome), 2):
            if np.random.random() < mutation_rate and i + 1 < len(genome):
                cut_index = mutated_genome[i]
                mutated_genome[i + 1] = np.random.randint(1, max_repeats_per_cut[cut_index] + 1)
        return mutated_genome

    def _can_place(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        if pos_x + prod_w > stock.shape[0] or pos_y + prod_h > stock.shape[1]:
            return False

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)