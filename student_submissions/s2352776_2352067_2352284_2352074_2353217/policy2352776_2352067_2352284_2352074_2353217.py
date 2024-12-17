from policy import Policy
import numpy as np
import random
# Implement Genetic 

class GENETIC(Policy):
    def __init__(self):
        self.population_size = 50
        self.num_generation = 8
        self.mutation_rate = 0.1
        self.num_products = 0
        
    def get_action(self, observation, info):
        return self.genetic_algorithm(observation, info)

    def genetic_algorithm(self, observation, info):
        # flow of genetic algorithm
        # 1. Initialize population
        # 2. Evaluate population
        # 3. Repeat
        #     1. Select parents
        #     2. Crossover
        #     3. Mutation
        #     4. Evaluate population
        # ! population = len prod * 10 is better or len_stock * 10 is better
        self.num_population = int(len(observation['products']) )
        # make num_population even
        if self.num_population % 2 != 0:
            self.num_population += 1
        
        self.num_generation = 5
        population = self.initialize_population(observation)
        for _ in range(self.num_generation):
            score = [self.fitness(chromosome, observation) for chromosome in population]
            selected_parents = self.selection(population, score)
            
            # Generate offspring without removing the selected parents
            offspring = []
            for i in range(0, len(selected_parents), 2):
                parent1 = selected_parents[i]
                parent2 = selected_parents[i + 1] if i + 1 < len(selected_parents) else selected_parents[0]  # Handle odd number of parents
                offspring1, offspring2 = self.crossover([parent1, parent2], observation)
                offspring.append(offspring1)
                offspring.append(offspring2)

            # Append offspring to population without removing anything
            population.extend(offspring)

            if len(population) > self.population_size:
                population = sorted(population, key=lambda x: self.fitness(x, observation), reverse=True)[:self.population_size]

        score = [self.fitness(chromosome, observation) for chromosome in population]
        best_chromosome = population[np.argmax(score)]
        return best_chromosome

    def initialize_population(self, observation):
        # encode population as list of dictionaries
        # each dictionary contains stock_idx, size, position
        population = []
        products = observation['products']

        # Filter available products with quantity > 0
        available_products = [i for i in range(len(products)) if products[i]['quantity'] > 0]

        for _ in range(self.population_size):
            chromosome = []

            # Choose a random product
            product_index = random.choice(available_products)
            product = products[product_index]
            product_size = product['size']

            while True:
                # Choose a random stock
                stock_idx = random.randint(0, len(observation['stocks']) - 1)
                stock = observation['stocks'][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)

                fit_found = False

                for pos_i in range(stock_w):
                    for pos_j in range(stock_h):
                        # Check placement without rotation
                        if self._can_place_(stock, (pos_i, pos_j), product_size):
                            chromosome.append({'stock_idx': stock_idx, 'size': product_size, 'position': (pos_i, pos_j)})
                            population.append({'stock_idx': stock_idx, 'size': product_size, 'position': (pos_i, pos_j)})
                            fit_found = True
                            break

                        # Check placement with rotation
                        if self._can_place_(stock, (pos_i, pos_j), product_size[::-1]):
                            chromosome.append({'stock_idx': stock_idx, 'size': product_size[::-1], 'position': (pos_i, pos_j)})
                            population.append({'stock_idx': stock_idx, 'size': product_size[::-1], 'position': (pos_i, pos_j)})
                            fit_found = True
                            break

                    if fit_found:
                        break

                if fit_found:
                    break

            if not chromosome:
                # If no fit was found, continue to find another stock
                continue

        return population
    
    def get_product_idx(self, products, product):
        for i, prod in enumerate(products):
            if prod == product:
                return i
        return -1

    def fitness(self, chromosome, observation):
        stock_idx = chromosome['stock_idx']
        stock = observation['stocks'][stock_idx]
        prod_size = chromosome['size']
        pos_x, pos_y = chromosome['position']
        stock_w, stock_h = self._get_stock_size_(stock)

        # Calculate normalized distance to target (filling from bottom-left corner)
        target_x, target_y = stock_w//2, stock_h//2
        max_distance = (stock_w**2 + stock_h**2)**0.5
        distance_to_target = (((pos_x - target_x)**2 + (pos_y - target_y)**2)**0.5) / max_distance

        # Calculate areas
        prod_area = prod_size[0] * prod_size[1]
        stock_area = stock_w * stock_h
        used_area = self.calculated_used_area(stock)
        prop_area = used_area / stock_area  # Normalized [0, 1]
        initial_prop_area = prop_area  # Area before adding the new product

        # Determine if product count is small
        total_products = len(observation['products'])
        small_product_threshold = 20  # Example threshold, can be adjusted

        # Define weights
        w1, w2, w3, w4 = 0.6, 0.2, 0.1, 0.1  # Added w4 for stock size preference

        # Bonuses and penalties
        bonus_for_large_usage = 0.5 if (used_area + prod_area) / stock_area > 0.8 else 0  # High bonus for high fill
        bonus_for_large_product = 0.3 if prod_area > 0.5 * stock_area else 0  # Bonus for larger products

        # High penalty for small item requiring a new stock
        penalty_for_small_new_stock = -0.4 if used_area == 0 and prod_area < 0.5 * stock_area else 0

        # Bonus for preferring larger stocks if the number of products is small
        bonus_for_large_stock = 0.2 if total_products <= small_product_threshold else 0
        num_products_threshold = 40
        # total product available that > 0
        num_prod = [prod['quantity'] for prod in observation['products'] if prod['quantity'] > 0]
        if len(num_prod) < num_products_threshold:
            w1 = 0.8
        # Calculate fitness score
        fitness_score = (
            w1 * ((used_area + prod_area) / stock_area) +  # Focus on maximizing stock usage
            w2 * (1 - distance_to_target) +                # Encourage placement closer to bottom-left
            w3 * (prod_area / stock_area)               # Assess product size
            # w4 * bonus_for_large_stock * stock_area/100                   # Prefer larger stock for small product counts
        ) * 1000

        # Apply bonuses and penalties
        fitness_score += (
            bonus_for_large_usage +
            bonus_for_large_product +
            penalty_for_small_new_stock
        ) * 500

        # Apply heavy penalty for invalid placement
        if not self._can_place_(stock, (pos_x, pos_y), prod_size):
            fitness_score = -10000

        return fitness_score
    
    def calculated_used_area(self, stock):
        # all element > 0 is used area
        return np.sum(stock >= 0)
    
    # roulette wheel selection
    def selection(self, population, score):
        total_score = sum(score)
        prob = [s / total_score for s in score]
        # print(f"Prob: {prob}")
        parent = random.choices(population, weights=prob, k=len(population)//2)
        return parent
    
    # cross over using partial bit exchange
    def crossover(self, parents, observation):
        parent1, parent2 = parents[0], parents[1]
        
        offspring1 = {}
        offspring2 = {}

        # Perform crossover for each key in the dictionary
        for key in parent1.keys():
            if random.random() > 0.5:  # Probability > 50% to swap genes
                offspring1[key] = parent2[key]
                offspring2[key] = parent1[key]
            else:
                offspring1[key] = parent1[key]
                offspring2[key] = parent2[key]

        # Validate offspring
        self.validate_offspring(offspring1, observation)
        self.validate_offspring(offspring2, observation)

        return offspring1, offspring2
    
    def validate_offspring(self, offspring, observation):
        stock_idx = offspring['stock_idx']
        size = offspring['size']
        pos_x, pos_y = offspring['position']
        stock = observation['stocks'][stock_idx]
        if not self._can_place_(stock, (pos_x, pos_y), size):
            # print(f"Invalid placement for {offspring} in offspring!")
            self.adjust_placement(offspring, offspring, observation)

    def adjust_placement(self, key, gene, observation):
        # print("Adjusting placement for", key)
        stock_idx = gene['stock_idx']
        stock = observation['stocks'][stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        product_size = gene['size']

        best_position = None
        best_score = float('inf')

        for pos_x in range(stock_w -1, -1,-1):
            for pos_y in range(stock_h - 1 , -1 , -1):
                if self._can_place_(stock, (pos_x, pos_y), product_size):
                    score = self.fitness(gene, observation)
                    if score < best_score:
                        best_score = score
                        best_position = (pos_x, pos_y)

                if self._can_place_(stock, (pos_x, pos_y), product_size[::-1]):
                    score = self.fitness(gene, observation)
                    if score > best_score:
                        best_score = score
                        best_position = (pos_x, pos_y)
                        product_size = product_size[::-1] 

        if best_position:
            gene['position'] = best_position
            gene['size'] = product_size
            return

        for fallback_stock_idx in range(len(observation['stocks'])):
            if fallback_stock_idx == stock_idx:
                continue
            fallback_stock = observation['stocks'][fallback_stock_idx]
            fallback_stock_w, fallback_stock_h = self._get_stock_size_(fallback_stock)

            for pos_x in range(fallback_stock_w):
                for pos_y in range(fallback_stock_h):
                    if self._can_place_(fallback_stock, (pos_x, pos_y), product_size):
                        gene['stock_idx'] = fallback_stock_idx
                        gene['position'] = (pos_x, pos_y)
                        return

        # If still no valid placement, reset or flag as invalid
        gene['stock_idx'] = -1

        # print(f"Unable to adjust placement for {key}. Retaining original position.")
    def mutation(self, offspring, observation):
        for chromosome in offspring:
            # Mutate with the given mutation rate
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(['size', 'position', 'stock_idx'])
                
                if mutation_type == 'size':
                    # Mutate size by adding a small random value (-1, 0, or 1)
                    size = chromosome['size']
                    size[0] += random.randint(-1, 1)
                    size[1] += random.randint(-1, 1)
                    # Ensure size is valid (positive and not too small)
                    size[0] = max(1, size[0])
                    size[1] = max(1, size[1])
                    chromosome['size'] = size
                    
                elif mutation_type == 'position':
                    # Mutate position by adding a random change (-1, 0, or 1) to both x and y
                    new_position = (
                        chromosome['position'][0] + random.randint(-1, 1),
                        chromosome['position'][1] + random.randint(-1, 1)
                    )
                    # Get the stock and check if the new position is valid
                    stock_idx = chromosome['stock_idx']
                    stock = observation['stocks'][stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    
                    # Ensure the new position is within stock boundaries
                    pos_x, pos_y = new_position
                    if 0 <= pos_x < stock_w and 0 <= pos_y < stock_h:
                        chromosome['position'] = new_position
                    else:
                        # print(f"Position {new_position} out of bounds. Retaining original position.")
                        pass
                
                elif mutation_type == 'stock_idx':
                    # Mutate stock index by selecting a random stock
                    new_stock_idx = random.randint(0, len(observation['stocks']) - 1)
                    chromosome['stock_idx'] = new_stock_idx

        return offspring   

class GREEDY(Policy):
    def __init__(self, product_sorting_policy = 1, stock_sorting_policy = 1):
        assert product_sorting_policy in [1, 2], "product_sorting_policy must be 1 or 2"
        assert stock_sorting_policy in [1, 2], "stock_sorting_policy must be 1 or 2"
        self.product_sorting_policy = product_sorting_policy
        self.stock_sorting_policy = stock_sorting_policy
        self.full_stock = [False] * 100
    
    def get_action(self, observation, info):
        list_prods = observation["products"]
        list_stocs = observation["stocks"]
        
        # Get index, width and height of all stocks
        stock_size = [(i,) + self._get_stock_size_(stock) for i, stock in enumerate(list_stocs)]
        
        # Sort products by largest area or by longest side
        if self.product_sorting_policy == 1:
            list_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        else:
            list_prods = sorted(list_prods, key=lambda x: max(x["size"]), reverse=True)

        # Sort stocks by largest area or by smallest area
        if self.stock_sorting_policy == 1:
            stock_size = sorted(stock_size, key=lambda x: x[1] * x[2], reverse=True)
        else:
            stock_size = sorted(stock_size, key=lambda x: x[1] * x[2])
        
        # print(list_prods)
        # print(stock_size)

        # Loop through all stocks
        for i, stock_w, stock_h in stock_size:
            # Skip stock if it is already full
            if self.full_stock[i]: continue

            # Pick a product that has quantity > 0
            for prod in list_prods:
                if prod["quantity"] > 0:
                    stock = list_stocs[i]
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - min(prod_w, prod_h) + 1):
                            for y in range(stock_h - min(prod_w, prod_h) + 1):
                                # Skip if the cell is already occupied
                                if stock[x][y] != -1: continue

                                if x + prod_w <= stock_w and y + prod_h <= stock_h:
                                    if self._can_place_(stock, (x, y), prod_size):
                                        # If last product, reset the stock
                                        if sum([prod["quantity"] for prod in list_prods]) == 1:
                                            self.full_stock = [False] * 100
                                        return {"stock_idx": i, "size": prod_size, "position": (x, y)}
                                
                                if x + prod_h <= stock_w and y + prod_w <= stock_h:
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        if sum([prod["quantity"] for prod in list_prods]) == 1:
                                            self.full_stock = [False] * 100
                                        return {"stock_idx": i, "size": prod_size[::-1], "position": (x, y)}
            
            # If no remaining products can be placed in the stock, mark the stock as full
            self.full_stock[i] = True

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
class Policy2352776_2352067_2352284_2352074_2353217(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.policy_1 = GREEDY()
        self.policy_2 = GENETIC()

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self.policy_1.get_action(observation, info)
        else:
            return self.policy_2.get_action(observation, info)


    # Student code here
    # You can add more functions if needed
