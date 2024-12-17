from policy import Policy
import numpy as np
import random


class Policy2352068_2352538_2352755_2352911(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.max_attempts = 10
        self.trim_loss_factor = 1.0
        self.policy_id = policy_id
        
        # Policy 2 parameters
        self.population_size = 20
        self.generations = 100
        self.mutation_rate = 0.01
        self.used_stocks = set()  # Keep track of used stocks
        self.stock_placements = {}  # Keep track of placements in each stock
        self.current_stock = {}  # Keep track of current stock
        



        # # Student code here
        # if policy_id == 1:

        #     pass
        # elif policy_id == 2:
        #     pass

    def get_action(self, observation, info):
        # Student code here
        
        if self.policy_id == 1:
            return self.get_action_id1(observation, info)
        elif self.policy_id == 2:
            print("Policy 2")
            return self.get_action_id2(observation, info)

    # Student code here
    # You can add more functions if needed

    def get_action_id1(self, observation, info):
        stocks = observation["stocks"]
        list_prods = observation["products"]
        stock_idx = -1
        pos_x, pos_y = None, None
        prod_size = [0, 0]
        

        sorted_prods = sorted(list_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # check all of stocks
                for i, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # not rotate
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = self._find_best_position(stock, stock_w, stock_h, prod_size)
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

                    # rotate 
                    if stock_w >= prod_h and stock_h >= prod_w:
                        rotated_size = prod_size[::-1]
                        pos_x, pos_y = self._find_best_position(stock, stock_w, stock_h, rotated_size)
                        if pos_x is not None and pos_y is not None:
                            prod_size = rotated_size
                            stock_idx = i
                            break

            if stock_idx != -1:
                break 

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def _find_best_position(self, stock, stock_w, stock_h, prod_size):

        prod_w, prod_h = prod_size

        # create and using DP
        dp = np.full((stock_w + 1, stock_h + 1), np.inf)
        dp[0][0] = 0  # origin 

        # using finding to find best location
        best_pos_x, best_pos_y = None, None
        min_trim_loss = float('inf')

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                # check block ( suitable or not )
                if self._can_place_(stock, (x, y), prod_size):
                    # counting trimloss
                    trim_loss = self._calculate_trim_loss(stock, stock_w, stock_h, (x, y), prod_size)
                    trim_loss *= self.trim_loss_factor  

                    if trim_loss < min_trim_loss:
                        min_trim_loss = trim_loss
                        best_pos_x, best_pos_y = x, y

        return best_pos_x, best_pos_y

    def _can_place_(self, stock, position, prod_size):
        x, y = position
        prod_w, prod_h = prod_size
        
        # limited the are
        if x + prod_w > len(stock) or y + prod_h > len(stock[0]):
            return False
        
        # check space
        for i in range(prod_w):
            for j in range(prod_h):
                if stock[x + i][y + j] != -1:  # -1 => not allow to cut
                    return False

        return True

    def _calculate_trim_loss(self, stock, stock_w, stock_h, position, prod_size):
        x, y = position
        prod_w, prod_h = prod_size
        unused_area = (stock_w - prod_w) * (stock_h - prod_h)
        product_area = prod_w * prod_h
        trim_loss = unused_area - product_area
        return trim_loss
    
    def sort_product(self, product):
        valid_products = [p for p in product if p["quantity"] > 0]
        return sorted(valid_products,
                      key=lambda p: p["size"][0] * p["size"][1],
                      reverse=True) 
    def get_action_id2(self, observation, info):
        list_prods = observation["products"]
        list_prods = self.sort_product(list_prods)
        stocks = observation["stocks"]

        if not np.array_equal(self.current_stock, stocks) or not self.used_stocks:
            self.current_stock = stocks
            self.used_stocks = set()
            self.stock_placements = {}

        if not hasattr(self, 'stock_placements') or not self.stock_placements:
            self.stock_placements = {i: [] for i in range(len(stocks))}
        
        # Find first available product
        for prod in list_prods:
            if prod["quantity"] > 0:
                current_product = prod
                break
        else:
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

        population = self._initialize_population(current_product, stocks)
        best_solution = None
        best_fitness = float('-inf')
        
        for _ in range(self.generations):
            # Calculate fitness for each solution
            fitness_scores = []
            for solution in population:
                fitness = self._evaluate_fitness(solution, stocks, current_product)
                fitness_scores.append(fitness)
            
            # Select elite solutions
            elite = self._select_elite(population, fitness_scores)
            
            # Update best solution if necessary
            current_best_idx = fitness_scores.index(max(fitness_scores))
            if fitness_scores[current_best_idx] > best_fitness:
                best_fitness = fitness_scores[current_best_idx]
                best_solution = population[current_best_idx]
            
            # Create new population starting with elite solutions
            new_population = elite.copy()
            
            # Fill rest of population through crossover and mutation
            while len(new_population) < self.population_size:
                if len(elite) >= 2:
                    parent1, parent2 = random.sample(elite, 2)
                    child = self._crossover(parent1, parent2)
                    if random.random() < self.mutation_rate:
                        child = self._mutate(child, stocks)
                    new_population.append(child)
                else:
                    # If not enough elite solutions, add random solutions
                    new_solution = self._initialize_population(current_product, stocks)[0]
                    new_population.append(new_solution)
            
            population = new_population

        if best_solution is None or not self._is_valid_solution(best_solution, stocks, current_product):
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

        stock_idx, x, y, is_rotated = best_solution
        size = current_product["size"][::-1] if is_rotated else current_product["size"]
        
        # Update used stocks and placements
        self.used_stocks.add(stock_idx)
        self.stock_placements[stock_idx].append((x, y, size[0], size[1]))

        return {
            "stock_idx": stock_idx,
            "size": size,
            "position": (x, y)
        }
    
    def _initialize_population(self, product, stocks):
        population = []
        prod_w, prod_h = product["size"]

        # Sort stocks by current utilization (descending)
        stock_utilization = [
            (idx, self._calculate_used_area(idx) / (self._get_stock_size_(stocks[idx])[0] * self._get_stock_size_(stocks[idx])[1]))
            for idx in range(len(stocks))
        ]
        stock_utilization.sort(key=lambda x: x[1], reverse=True)
        available_stocks = [idx for idx, _ in stock_utilization]

        while len(population) < self.population_size:
            for stock_idx in available_stocks:
                if len(population) >= self.population_size:
                    break

                stock = stocks[stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                orientations = [(False, prod_w, prod_h), (True, prod_h, prod_w)]

                for is_rotated, w, h in orientations:
                    positions = self._find_valid_positions(stock_idx, w, h, stock_w, stock_h)
                    if positions:
                        for pos in positions:
                            if len(population) >= self.population_size:
                                break
                            population.append((stock_idx, pos[0], pos[1], is_rotated))

        return population
    def _calculate_stock_suitability(self, stock_idx, stock_w, stock_h, prod_w, prod_h, stocks):
        score = 0
        
        # Prefer previously used stocks
        if stock_idx in self.used_stocks:
            score += 1000
        
        # Calculate current utilization
        current_utilization = self._calculate_used_area(stock_idx) / (stock_w * stock_h)
        potential_utilization = ((self._calculate_used_area(stock_idx) + (prod_w * prod_h)) 
                            / (stock_w * stock_h))
        
        # Prefer stocks that would result in better utilization
        score += potential_utilization * 2000
        
        # Penalize if stock is too large compared to product
        size_ratio = (prod_w * prod_h) / (stock_w * stock_h)
        if size_ratio < 0.1:  # If product is less than 10% of stock size
            score -= 500
        
        # Bonus for perfect fit
        if (stock_w == prod_w and stock_h >= prod_h) or \
        (stock_w == prod_h and stock_h >= prod_w) or \
        (stock_h == prod_w and stock_w >= prod_h) or \
        (stock_h == prod_h and stock_w >= prod_w):
            score += 1500
        
        # Penalize stocks that are already nearly full
        if current_utilization > 0.9:
            score -= 1000
        
        # Check if product can fit in remaining space
        can_fit_normal = prod_w <= stock_w and prod_h <= stock_h
        can_fit_rotated = prod_h <= stock_w and prod_w <= stock_h
        
        if not (can_fit_normal or can_fit_rotated):
            score -= 5000
        
        return score

    def _evaluate_fitness(self, solution, stocks, product):
        stock_idx, x, y, is_rotated = solution

        if not self._is_valid_solution(solution, stocks, product):
            return float('-inf')

        fitness = 0

        # Reward or penalize based on utilization
        stock_w, stock_h = self._get_stock_size_(stocks[stock_idx])
        prod_size = product["size"][::-1] if is_rotated else product["size"]
        used_area = self._calculate_used_area(stock_idx) + prod_size[0] * prod_size[1]
        total_stock_area = stock_w * stock_h
        utilization = used_area / total_stock_area
        waste = 1 - utilization

        fitness += int(5000 * utilization)  # Higher reward for better utilization
        fitness -= int(3000 * waste)       # Penalize waste explicitly

        # Check for overlaps with existing placements
        current_rect = (x, y, prod_size[0], prod_size[1])
        overlaps = any(self._rect_overlap(current_rect, rect)
                    for rect in self.stock_placements.get(stock_idx, []))
        if overlaps:
            fitness -= 10000

        return fitness


    def _is_valid_solution(self, solution, stocks, product):
        stock_idx, x, y, is_rotated = solution
        if stock_idx >= len(stocks):
            return False

        stock = stocks[stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        w, h = product["size"] if not is_rotated else product["size"][::-1]

        # Check if within stock boundaries
        if x + w > stock_w or y + h > stock_h:
            return False

        # Check for overlaps with existing placements
        for placed_rect in self.stock_placements.get(stock_idx, []):
            if self._rect_overlap((x, y, w, h), placed_rect):
                return False

        return True

    def _select_elite(self, population, fitness_scores):
        # Create list of (solution, fitness) pairs
        solution_fitness = list(zip(population, fitness_scores))
        
        # Sort by fitness score
        solution_fitness.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 20% of population
        elite_size = max(2, self.population_size // 5)
        elite = [solution for solution, _ in solution_fitness[:elite_size]]
        
        return elite

    def _crossover(self, parent1, parent2):
        # Randomly choose attributes from either parent
        if random.random() < 0.5:
            stock_idx = parent1[0]
        else:
            stock_idx = parent2[0]
            
        x = min(parent1[1], parent2[1])  # Choose the smallest x
        y = min(parent1[2], parent2[2])  # Choose the smallest y
        
        # Randomly choose rotation
        is_rotated = random.choice([parent1[3], parent2[3]])

        return (stock_idx, x, y, is_rotated)

    def _mutate(self, solution, stocks):
        stock_idx, x, y, is_rotated = solution
        stock = stocks[stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)

        mutation_type = random.randint(0, 2)
        if mutation_type == 0:
            # Rotate the product
            is_rotated = not is_rotated
        elif mutation_type == 1:
            # Move to least utilized stock
            least_utilized_stock = min(
                range(len(stocks)),
                key=lambda idx: self._calculate_used_area(idx) / 
                                (self._get_stock_size_(stocks[idx])[0] * self._get_stock_size_(stocks[idx])[1])
            )
            stock_idx = least_utilized_stock
            x, y = 0, 0
        elif mutation_type == 2:
            # Place closer to an existing product to minimize fragmentation
            positions = self._find_valid_positions(stock_idx, stock_w, stock_h, stock_w, stock_h)
            if positions:
                x, y = positions[0]

        return (stock_idx, x, y, is_rotated)
    
    def _rect_overlap(self, rect1, rect2):
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        return not (x1 + w1 <= x2 or x1 >= x2 + w2 or
                    y1 + h1 <= y2 or y1 >= y2 + h2)

    def _calculate_used_area(self, stock_idx):
        # Calculate total area of placed products in the stock
        used_area = 0
        for x, y, w, h in self.stock_placements.get(stock_idx, []):
            used_area += w * h
        return used_area

    def _find_valid_positions(self, stock_idx, w, h, stock_w, stock_h):
        # Find all positions where the product can fit without overlapping
        positions = []
        for y in range(stock_h - h + 1):  # Row by row
            for x in range(stock_w - w + 1):  # Left to right (systematic approach)
                overlaps = False
                new_rect = (x, y, w, h)
                for placed_rect in self.stock_placements.get(stock_idx, []):
                    if self._rect_overlap(new_rect, placed_rect):
                        overlaps = True
                        break
                if not overlaps:
                    positions.append((x, y))
        return positions
