from policy import Policy
import random
import numpy as np

class PolicyGA_2(Policy):
    def __init__(self, policy_id=1, population_size=50, generations=20, mutation_rate=0.1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Student code here
        self.policy_id = policy_id
        if policy_id == 1:
            self.policy_type = 'firstfit'
        elif policy_id == 2:
            self.policy_type = 'genetic'

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.current_individual = []
        self.current_stock = 0
        self.current_stock_idx = -1
        
    
    def get_action(self, observation, info):
        # Student code here
        # Get products and stocks lists
        if self.policy_type == 'genetic':
    ##############################################################
            products = sorted(
                [prod for prod in observation["products"] if prod["quantity"] > 0], 
                key=lambda prod: prod["size"][0] * prod["size"][1],
                reverse=True
            )

            stocks = []
            for i, stock in enumerate(observation["stocks"]):
                stock_size = self._get_stock_size_(stock)
                stock_area = stock_size[0] * stock_size[1]
                stocks.append((i, stock, stock_size, stock_area))
            stocks.sort(key=lambda x: x[3], reverse=True)

    ##############################################################
            # If already have solution, return, else get new solution
            if (len(self.current_individual)):
                stock_idx, prod_size, position = self._decode_solution_(self.current_individual, self.current_stock)
                self.current_individual.pop(0)
                # print("Already have solution", stock_idx, prod_size, position)
                return {"stock_idx": stock_idx, "size": prod_size, "position": position}
    ###############################################################
            # Getting new solution
            # print("Getting new solution")
            prod_size = [0, 0]
            position = (0, 0)
            population = []
            # stock_area_print = 0
            for stock_idx, stock, _, stock_area in stocks:
                self.current_stock_idx = stock_idx
                self.current_stock = np.copy(stock)
                population = self._initialize_population_(products, self.current_stock)
                if (len(population)):
                    break
            if (len(population) == 0): 
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)} # No space left for any prod or no prod left

            if (len(population) == 1): population += population # If only have 1 individual, double it
            for _ in range(self.generations):
                # Cal fitness_scores for each individual in population
                # Get the best 2 for next generation
                # Select the best one out of each 3 random individuals, add it to the new population
                # Mutate new population
                # Insert the best 2 individuals to the begining of new population
                # Repeat every generation for the new population
                fitness_scores = [self._calculate_fitness_(individual, self.current_stock) for individual in population]
                top_two_ind = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:2]
                new_population = self._select_(population, fitness_scores, self.population_size - 2)
                new_population = self._mutate_(new_population, products, self.mutation_rate)
                new_population = [population[top_two_ind[0]], population[top_two_ind[1]]] + new_population
                population = new_population

            # Get the best individual base on the fitness scores and shortest length.
            # Store it and return the first one
            best_individual = max(population, key=lambda ind: (self._calculate_fitness_(ind, self.current_stock), -len(ind)))
            stock_idx, prod_size, position = self._decode_solution_(best_individual, self.current_stock)
            if (len(best_individual) > 1):
                self.current_individual = best_individual
                self.current_individual.pop(0)

            return {"stock_idx": self.current_stock_idx, "size": prod_size, "position": position}
        elif self.policy_type == 'firstfit':
            products = []
        for i, prod in enumerate(observation["products"]):
            # print(f"Processing product {i}: {prod} ,{prod["quantity"]} left")
            prod_area = prod["size"][0] * prod["size"][1]
            products.append((i, prod, prod_area))
        products.sort(key=lambda x: x[2], reverse=True)

        # Precompute stock areas and sort stocks by ascending area
        stocks = []
        for i, stock in enumerate(observation["stocks"]):
            stock_size = self._get_stock_size_(stock)
            stock_area = stock_size[0] * stock_size[1]
            stocks.append((i, stock, stock_size, stock_area))
        stocks.sort(key=lambda x: x[3], reverse=True)
        # Iterate over products and find placement
        for prod_idx, prod, _ in products:
            if prod["quantity"] <= 0:
                continue
            long_side = max(prod["size"])
            short_side = min(prod["size"])
            
            for stock_idx, stock, stock_size, _ in stocks:
                stock_w, stock_h = stock_size
                if stock_w > stock_h:
                    prod_w = long_side
                    prod_h = short_side
                else: 
                    prod_w = short_side
                    prod_h = long_side
                possible_orientations = [(prod_w, prod_h), (prod_h, prod_w)]  # Both orientations
                # Check if the product can fit in this stock (any orientation)
                for prod_w, prod_h in possible_orientations:
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = self._find_placement(stock, (prod_w, prod_h))
                        if pos_x is not None:
                            return {
                                "stock_idx": stock_idx,
                                "size": [prod_w, prod_h],
                                "position": (pos_x, pos_y)
                            }
        
        # No valid placement found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _initialize_individual_(self, products, stock):
        # stock_state copy of the current stock to determine remaning space without changing the original stock
        # Go through each prod
        # Randomly 50% chance rotate prod
            # If prod too big for the stock: next prod
            # If prod's area bigger than stock_state's remaining space: next prod
        # Get the list of the corners
            # If no conner left (stock is fulled): return individual
        # Random select 1 corner for position
        # If canplace prod at position: append to individual
        # Continue till no prod left
        stock_state = np.copy(stock)
        stock_w, stock_h = self._get_stock_size_(stock_state)
        remain_area = np.sum(stock_state == -1)
        individual = []
        for prod in products:
            prod_sizes = [prod["size"], prod["size"][::-1]]
            prod_area = prod["size"][0] * prod["size"][1]
            for _ in range(prod["quantity"]):
                if prod_area > remain_area: break
                prod_w, prod_h = prod_sizes[random.randint(0, 1)]

                if (prod_w > stock_w or prod_h > stock_h): break #  If prod too large for the stock: next prod
                
                corners = self._get_corners_(stock_state)  # get corner list from the stock
                if not corners: return individual

                position = corners[random.randint(0, len(corners) - 1)] # Random corner from corner list
                pos_x, pos_y = position
                
                if (
                    (pos_x + prod_w <= stock_w) and (pos_y + prod_h <= stock_h) and 
                    self._can_place_(stock_state, position, [prod_w, prod_h])
                ):    #if can place: place it, add to the individual list, update stock state
                    
                    stock_state[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] = 1
                    individual.append((self.current_stock_idx, [prod_w, prod_h], position))
                    remain_area -= prod_area
                else: break  # Else if it can't be placeed: next prod
        return individual

    def _initialize_population_(self, products, stock):
        
        population = []
        for _ in range(self.population_size):
            individual = self._initialize_individual_(products, stock)
            if (len(individual)):
                population.append(individual)
        return population
    
    def _get_corners_(self, stock):
        up = np.roll(stock, shift=-1, axis=1)  # Shift the grid upward
        down = np.roll(stock, shift=1, axis=1)  # Shift downward
        left = np.roll(stock, shift=1, axis=0)  # Shift left
        right = np.roll(stock, shift=-1, axis=0)  # Shift right
        
        # Combine conditions to find corners
        corners = (stock == -1) & (down != -1) & (left != -1) & ((up == -1) | (right == -1))
        corner_indices = np.argwhere(corners)  # Get corner coordinates
        corner_list = [tuple(coord) for coord in corner_indices]  # Convert to list of tuples
        return corner_list
    
    def _calculate_fitness_(self, individual, stock):
        stock_w, stock_h = self._get_stock_size_(stock)
        stock_area = np.sum(stock[0 : stock_w, 0 : stock_h] == -1)
        used_area = 0
        for stock_idx, prod_size, position in individual:
            used_area += prod_size[0] * prod_size[1]
        return used_area / stock_area
    
    def _select_(self, population, fitness_scores, population_size):
        new_population = []
        combined = list(zip(population, fitness_scores))

        # If the population is smaller than 3, return the best individual directly
        if len(combined) < 3:
            best_individual = max(combined, key=lambda x: (x[1], -len(x[0])))
            return [best_individual[0]] * population_size  # Fill the new population with the best one

        # Otherwise, perform tournament selection
        for _ in range(population_size):
            candidates = random.sample(combined, k=3)
            winner = max(
                candidates,
                key=lambda x: (x[1], -len(x[0]))  # Sort by fitness, then by shorter length
            )
            new_population.append(winner[0])

        return new_population
    
    def _mutate_(self, population, products, mutation_rate):
        new_population = []
        for individual in population:
            remain_prods = [prod.copy() for prod in products]
            current_mutation_rate = mutation_rate
            stock_state = np.copy(self.current_stock)
            mutation_point = -1
            for i in range(len(individual) - 1, -1, -1):
                if (len(individual) == 1): current_mutation_rate = 1
                else:
                    current_mutation_rate = mutation_rate + (len(individual) - 1 - i) * (1 - mutation_rate) / (len(individual) - 1)
                if random.random() <= current_mutation_rate:
                    mutation_point = i
                    break
            if mutation_point == -1:
                new_population.append(individual)
                continue
            mutate_individual = individual[:mutation_point + 1]

            for unit in mutate_individual:
                stock_idx, prod_size, position = unit
                pos_x, pos_y = position
                prod_w, prod_h = prod_size
                for prod in remain_prods:
                    if np.array_equal(prod["size"], prod_size) or np.array_equal(prod["size"], prod_size[::-1]):
                        prod["quantity"] -= 1
                        break
                stock_state[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] = 1
            mutated_individual = mutate_individual + self._initialize_individual_(remain_prods, stock_state)
            new_population.append(mutated_individual)
        return new_population



    def _decode_solution_(self, solution, stock):
        """Decode the solution to retrieve the action."""
        for stock_idx, prod_size, position in solution:
            if self._can_place_(stock, position, prod_size):
                return stock_idx, prod_size, position
        return -1, [0, 0], (0, 0)
    # Student code here
    # You can add more functions if needed
    def _find_placement(self, stock, prod_size):
        # Optimized placement search (bounding box or 2D grid representation)
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Assume we have a grid representation of the stock
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None, None
