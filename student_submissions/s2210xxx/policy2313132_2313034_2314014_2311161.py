import numpy as np
import random
from policy import Policy

class Policy2210xxx(Policy):
    def __init__(self, policy_id = 1, population_size=75, generations=100, mutation_rate=0.3):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy_id = policy_id
            pass
        elif policy_id == 2:
            self.policy_id = policy_id
            self.population_size = population_size
            self.generations = generations
            self.mutation_rate = mutation_rate
        
            self.best_solution = None
            self.start = True
            self.i = 0
            self.count = 0
            self.stocks = []
            pass
#----------------------------------- GREEDY ALGORITHM + HEURISTIC -----------------------------------
    def _bottom_left_fit_(self, stock, prod_size):
        """Heuristic: Tìm vị trí Bottom-Left phù hợp nhất để đặt sản phẩm."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        best_pos = None
        min_distance = float('inf')

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    distance = x + y
                    if distance < min_distance:
                        min_distance = distance
                        best_pos = (x, y)

        return best_pos
    def _update_stock_usage_greedy(self, stocks_copy, placement):
        stock_idx = placement["stock_idx"]
        position = placement["position"]
        prod_size = placement["size"]

        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        stocks_copy[stock_idx][pos_x:pos_x+prod_w, pos_y:pos_y+prod_h] = 1
# ----------------------------------- GENETIC ALGORITHM -----------------------------------
    def generate_initial_population(self, observation):
        population = []
        self.stocks = observation["stocks"]
        products = observation["products"]
        for _ in range(self.population_size):
            individual = self._generate_solution(self.stocks, products)
            population.append(individual)
        
        return population

    def _generate_solution(self, stocks, products):
        solution = []
        stocks_copy = [np.copy(stock) for stock in stocks]

        # Try to place products randomly
        for prod in products:
            num_prod = prod["quantity"]
            for _ in range(num_prod):
                placement_found = False
                for _ in range(100):  # Limit attempts to prevent infinite loop
                    # Randomly select a stock
                    stock_idx = random.randint(0, len(stocks_copy) - 1)
                    stock = stocks_copy[stock_idx]
                    
                    # Get stock dimensions
                    stock_w, stock_h = self._get_stock_size_(stock)
                    x = np.array([-1,-1,-1,-1])
                    prod_w, prod_h = prod["size"]

                    # Check if product fits in stock
                    if (stock_w < prod_w or stock_h < prod_h): # and (stock_w < prod_h or stock_h < prod_w):
                        continue
                    
                    # Try random position
                    pos_x = random.randint(0, stock_w - prod_w)
                    pos_y = random.randint(0, stock_h - prod_h)
                    
                    # Try random position with original size
                    if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                        cut_pattern = {"stock_idx": stock_idx, "size": prod["size"], "position": (pos_x, pos_y)}
                        solution.append(cut_pattern)
                        self._update_stock_usage_genetic(stocks_copy, cut_pattern)
                        placement_found = True
                        break

            # If no placement found, skip this product
                if not placement_found:
                    continue    
        return solution
    def cal_fitness(self, population, observation):
        
        fitness_scores = []
        
        for individual in population:
            stocks_copy = [np.copy(stock) for stock in observation["stocks"]]
            
            # Track placed products and remaining products
            placed_products = 0
            total_products = sum(prod["quantity"] for prod in observation["products"])
            wasted_space = 0
            
            for action in individual:
                stock_idx = action["stock_idx"]
                position = action["position"]
                prod_size = action["size"]
                if 0 <= stock_idx < len(stocks_copy):
                    stock = stocks_copy[stock_idx]
                    if self._can_place_(stock, position, prod_size):
                        pos_x, pos_y = position
                        prod_w, prod_h = prod_size
                        stock[pos_x:pos_x+prod_w, pos_y:pos_y+prod_h] = 1
                        placed_products += 1
            
            # Calculate wasted space 
            total_size = 0
            total_stock_used = 0
            for stock in stocks_copy:
                placed = False
                stock_w,stock_h = self._get_stock_size_(stock)
                for i in range(stock_w):
                    for j in range(stock_h):
                        if stock[i][j] != - 1:
                            placed = True
                            break
                    if placed == True:
                        break
                if(placed == True):
                    wasted_space += np.sum(stock == -1)
                    total_size += stock.size
                    total_stock_used += 1
            # Calculate fitness
            total_wasted_space = sum(np.prod(self._get_stock_size_(stock)) for stock in observation["stocks"])
            normalized_wasted_space = wasted_space / total_wasted_space if total_wasted_space > 0 else 0
            fitness = - placed_products * 100 + 0.001 * normalized_wasted_space + total_stock_used 
            fitness_scores.append(fitness)

        return fitness_scores

    def crossover(self, parent1, parent2):
        # Intelligent crossover to minimize overlaps
        if len(parent1) != len(parent2):
            return random.choice([parent1, parent2])
        
        # Smart crossover with placement validation
        child1 = []
        child2 = []
        
        # Create copies of stocks to track placement
        stocks_copy1 = [np.copy(stock) for stock in self.stocks]
        stocks_copy2 = [np.copy(stock) for stock in self.stocks]
        
        for i in range(len(parent1)):
            if random.random() < 0.5 and self._can_place_safe(stocks_copy1, parent1[i]):
                child1.append(parent1[i])
                self._update_stock_usage_genetic(stocks_copy1, parent1[i])
            elif self._can_place_safe(stocks_copy1, parent2[i]):
                child1.append(parent2[i])
                self._update_stock_usage_genetic(stocks_copy1, parent2[i])
    
            if random.random() < 0.5 and self._can_place_safe(stocks_copy2, parent2[i]):
                child2.append(parent2[i])
                self._update_stock_usage_genetic(stocks_copy2, parent2[i])
            elif self._can_place_safe(stocks_copy2, parent1[i]):
                child2.append(parent1[i])
                self._update_stock_usage_genetic(stocks_copy2, parent1[i])
        
        return child1, child2
    def _remove_product_from_stock(self, stock, position, size):
   
        pos_x, pos_y = position
        width, height = size

        if pos_x + width > stock.shape[0] or pos_y + height > stock.shape[1]:
            raise ValueError("Kích thước hoặc vị trí vượt quá giới hạn kho!")
        stock[pos_y:pos_y + height, pos_x:pos_x + width] = 0
    def mutation(self, individual, observation):
        mutated = np.copy(individual)
        stocks = observation["stocks"]
        stocks_copy = [np.copy(stock) for stock in stocks]
        for action in individual:
            self._update_stock_usage_genetic(stocks_copy,action)
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                prod_size = mutated[i]["size"]
                old_stock_idx = mutated[i]["stock_idx"]
                old_position = mutated[i]["position"]


                self._remove_product_from_stock(stocks_copy[old_stock_idx], old_position, prod_size)


                placement_found = False
                for _ in range(100): 
                    stock_idx = random.randint(0, len(stocks_copy) - 1)
                    candidates = self._generate_placement_candidates(stocks_copy[stock_idx], prod_size)
                    np.random.shuffle(candidates)
                    for pos_x, pos_y in candidates:
                        if self._can_place_(stocks_copy[stock_idx], (pos_x, pos_y), prod_size):
                            mutated[i] = {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (pos_x, pos_y),
                            }
                            self._update_stock_usage_genetic(stocks_copy, mutated[i])
                            placement_found = True
                            break
                    if placement_found:
                        break

                if not placement_found:
                    self._update_stock_usage_genetic(stocks_copy, mutated[i])

        return mutated

    def _generate_placement_candidates(self, stock, prod_size):
        strategies  = []
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        
        # Generate multiple placement strategies
        if stock_w > prod_w and stock_h > prod_h :
            strategies.extend([
                # Random placement - Original size
                (random.randint(0, stock_w - prod_w), random.randint(0, stock_h - prod_h)),
                # Edge-aligned placements
                (0, 0),  # Top-left
                (stock_w - prod_w, 0),  # Top-right
                (0, stock_h - prod_h),  # Bottom-left
                (stock_w - prod_w, stock_h - prod_h),  # Bottom-right
                # Center placements
                # ((stock_w - prod_w) // 2, (stock_h - prod_h) // 2)
            ])
        
            # Add some grid-based placements
            for x in range(0, stock_w - prod_w + 1, max(1, prod_w // 2)):
                for y in range(0, stock_h - prod_h + 1, max(1, prod_h // 2)):
                    strategies.append((x, y))
        
        return strategies

    def _can_place_safe(self, stocks_copy, placement):
        stock_idx = placement["stock_idx"]
        position = placement["position"]    
        prod_size = placement["size"]
        
        # Validate stock index
        if 0 <= stock_idx < len(stocks_copy):
            stock = stocks_copy[stock_idx]
            
            # Check if placement is possible
            if self._can_place_(stock, position, prod_size):
                # Mark the space as used
                pos_x, pos_y = position
                prod_w, prod_h = prod_size
                
                # Check for any existing overlaps
                if np.any(stock[pos_x:pos_x+prod_w, pos_y:pos_y+prod_h] == 1):
                    return False
                
                # Mark the space as used
                stock[pos_x:pos_x+prod_w, pos_y:pos_y+prod_h] = 1
                return True
        
        return False

    def _update_stock_usage_genetic(self, stocks_copy, placement):
        stock_idx = placement["stock_idx"]
        position = placement["position"]
        prod_size = placement["size"]
        
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        
        stocks_copy[stock_idx][pos_x:pos_x+prod_w, pos_y:pos_y+prod_h] = 1
    def geneticAlgo(self, observation, info):
        population = self.generate_initial_population(observation)
        global_best_solution = None
        global_best_fitness = float('inf')
        
        # Main evolutionary loop
        for generation in range(self.generations):
            # Calculate fitness for current population
            fitness_scores = self.cal_fitness(population, observation)
            
            # Find best solution in current generation
            current_best_idx = np.argmin(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            current_best_solution = population[current_best_idx]
            
            # Update global best solution if current is better
            if current_best_fitness < global_best_fitness:
                global_best_fitness = current_best_fitness
                global_best_solution = current_best_solution
            
            # Selection - choose top performers
            num_selected = max(2, int(self.population_size * 0.5))
            selected_indices = np.argsort(fitness_scores)[-num_selected:]
            selected_population = [population[i] for i in selected_indices]
            
            # Create new population
            new_population = selected_population[:]
            
            # Elitism - always keep the best solution
            new_population.append(global_best_solution)
            
            # Reproduce and fill population
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                try:
                    child1, child2 = self.crossover(parent1, parent2)
                except Exception:
                    # Fallback if crossover fails
                    child1, child2 = parent1, parent2
                
                # Mutation
                child1 = self.mutation(child1, observation)
                child2 = self.mutation(child2, observation)
                
                # Add children to population
                new_population.extend([child1, child2])
            
            # Trim population to original size
            population = new_population[:self.population_size]
            
            # Optional: Print generation progress
            if generation % 20 == 0:
                print(f"Generation {generation}: Fitness = {global_best_fitness}")
            # print("Length of best solution = ", len(global_best_solution))

        # Set the best solution found
        self.best_solution = global_best_solution
        return global_best_solution

    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        # Randomly select individuals for the tournament
        tournament_indices = np.random.choice(
            len(population), 
            size=tournament_size, 
            replace=False
        )
        
        # Find the best individual in the tournament( has the minimum value of fitness)
        best_tournament_idx = min(
            tournament_indices, 
            key=lambda idx: fitness_scores[idx]
        )
        
        return population[best_tournament_idx]
# ------------------Action------------------
    def get_action(self, observation, info):
        if self.policy_id == 1:
            list_prods = observation["products"]
            stocks = observation["stocks"]

            sorted_prods = sorted(
                list_prods,
                key=lambda prod: prod["size"][0] * prod["size"][1],
                reverse=True
            )
            sorted_stocks = sorted(
                enumerate(stocks),
                key=lambda s: self._get_stock_size_(s[1])[0] * self._get_stock_size_(s[1])[1],
                reverse=True
            )

            for prod in sorted_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]

                    for stock_idx, stock in sorted_stocks:
                        pos = self._bottom_left_fit_(stock, prod_size)
                        if pos:
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": pos
                            }

                        rotated_size = (prod_size[1], prod_size[0])
                        pos = self._bottom_left_fit_(stock, rotated_size)
                        if pos:
                            return {
                                "stock_idx": stock_idx,
                                "size": rotated_size,
                                "position": pos
                            }

            return None
        elif self.policy_id == 2:
            if(self.start == True):
                print("---START RUNNING ALGORITHM ---")
                self.geneticAlgo(observation,info)
                self.start = False
                self.i = 0
                print("---FINISH RUNNING ALGORITHM && START PLACING---")
            # If end of the problem, reset and return a empty cut
            if(self.i >= len(self.best_solution)):
                self.i = 0
                self.start = True
                return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
            print(len(self.best_solution))
            cut = self.best_solution[self.i]
            self.i += 1
            return cut
    
        
