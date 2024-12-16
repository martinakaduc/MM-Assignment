from policy import Policy
import numpy as np
from random import randint, shuffle, choice, choices
import random  # Để giữ hàm random() nguyên bản
from scipy.optimize import linprog 


class Policy2312845_2312460_2312665_2312934_2312580(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = ColumnGeneration()
        elif policy_id == 2:
            self.policy = GeneticPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)



class ColumnGeneration(Policy):
    def __init__(self):
        # Initialize basic parameters of the policy
        super().__init__()
        self.patterns = []  # List of patterns
        self.init = False  
        self.num_prods = None  # Number of products

    def init_patterns(self, num_prods, sizes, stocks):
        """
        Function to initialize cutting patterns for products based on the number of products, product sizes, and material stocks.
        """
        self.patterns = []  
        for i in range(len(stocks)):
            stock_size = self._get_stock_size_(stocks[i])  # Get stock size
            for j in range(num_prods):
                # Check if the product fits the stock
                if stock_size[0] >= sizes[j][0] and stock_size[1] >= sizes[j][1]:
                    pattern = np.zeros(num_prods, dtype=int)
                    pattern[j] = 1  # Mark the product to be cut
                    self.patterns.append(pattern)

        # Filter out unique patterns
        unique_patterns = []
        seen = set()
        for p in self.patterns:
            t = tuple(p)  
            if t not in seen:
                seen.add(t)
                unique_patterns.append(p)
        self.patterns = unique_patterns

    def get_action(self, observation, info):
        """
        Function receives observation and information, then computes the optimal action.
        Uses linear programming to find suitable patterns and stocks, then returns the action.
        """
        products = observation["products"]  # Get list of products
        stocks = observation["stocks"]  # Get list of stocks

        # Sort stocks by area (from largest to smallest)
        sorted_stocks = sorted(
            enumerate(stocks),
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
            reverse=True  # Sort by descending area
        )
        
        # Compute product demand
        demand = []
        for prod in products:
            demand.append(prod["quantity"])
        demand = np.array(demand)

        # Get product sizes
        sizes = []
        for prod in products:
            sizes.append(prod["size"])

        # Sort products by area (from largest to smallest)
        sizes_area = np.array([size[0] * size[1] for size in sizes])
        sorted_indices = np.argsort(sizes_area)[::-1]
        sizes = [sizes[i] for i in sorted_indices]
        demand = demand[sorted_indices]

        num_prods = len(products)

        # Initialize patterns if not initialized or if number of products changed
        if not self.init or self.num_prods != num_prods:
            self.init_patterns(num_prods, sizes, stocks)
            self.init = True
            self.num_prods = num_prods

        # Solve Master Problem and Pricing Problem
        while True:
            solve = self.solve_restricted_master_problem(demand)
            if solve.status != 0:
                break
            if 'marginals' in solve.ineqlin.__dict__:
                dual_prices = solve.ineqlin.marginals
            else:
                break  # Exit if no dual prices are available

            new_pattern = self.solve_pricing_problem(dual_prices, sizes, stocks)
            if new_pattern is None:
                break

            # Check if the new pattern already exists in the list of patterns
            for p in self.patterns:
                if np.array_equal(new_pattern, p):
                    break
            else:
                # If no matching pattern is found, add the new pattern
                self.patterns.append(new_pattern)

        # Select the optimal pattern and corresponding stock
        pattern = self.select_pattern(self.patterns, demand)
        action = self.select_stock(pattern, sizes, stocks)
        return action

    def solve_restricted_master_problem(self, demand):
        """
        Function to solve the Master Problem using linear programming.
        The goal is to minimize the number of patterns used.
        """
        obj_func = np.ones(len(self.patterns))  # Objective function: minimize number of patterns
        coefficient_matrix = np.transpose(self.patterns)  # Coefficient matrix (patterns)
        b = demand  # Product demand
        result = linprog(obj_func, A_ub=-coefficient_matrix, b_ub=-b, bounds=(0, None), method='highs')
        return result

    def solve_pricing_problem(self, dual_prices, sizes, stocks):
        """
        Function to solve the pricing subproblem to find the best cutting pattern based on reduced costs.
        """
        best_pattern = None
        best_reduced_cost = -1
        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)  # Get stock size
            if stock_w <= 0 or stock_h <= 0:
                continue

            n = len(sizes)
            # Setup linear programming problem
            obj_func = -np.array(dual_prices)  # Objective: maximize reduced cost
            coefficient_matrix = [
                [1 if j == i else 0 for j in range(n)]
                for i, (w, h) in enumerate(sizes)
            ]
            b = [
                stock_w // w * stock_h // h
                for w, h in sizes
            ]

            # Constraints: Products must fit within the material size
            bounds = []
            for _ in range(n):
                bounds.append((0, None))

            # Solve the linear programming problem
            result = linprog(obj_func, A_ub=obj_func, b_ub=b, bounds=bounds, method='highs')

            if result.success and (reduced_cost := np.dot(result.x, dual_prices) - 1) > best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_pattern = result.x

        return best_pattern if best_reduced_cost > 1e-6 else None

    def select_pattern(self, patterns, demand):
        """
        Function to select the optimal pattern based on demand and product quantity.
        """
        best_pattern = max(patterns, key=lambda pattern: (np.sum(np.minimum(pattern, demand)), -np.sum(pattern)))
        return best_pattern

    def select_stock(self, pattern, sizes, stocks):
        """
        Function to select the appropriate stock for placing the products according to the chosen pattern.
        Sorts stocks by descending area and finds a suitable position for the products in the stock.
        """
        sorted_stocks_by_area = [
            (stock_idx, self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1], stock)
            for stock_idx, stock in enumerate(stocks)
        ]
        sorted_stocks_by_area.sort(key=lambda x: x[1], reverse=True)  # Sort by descending area

        # Iterate through each product in the pattern and find a suitable stock
        for product_index, product_count in enumerate(pattern):
            if product_count > 0:  # If at least one product needs to be cut
                product_size = sizes[product_index]
                
                # Iterate through the sorted stocks by area
                for stock_idx, stock_area, stock in sorted_stocks_by_area:
                    stock_width, stock_height = self._get_stock_size_(stock)
                    product_width, product_height = product_size

                    # Check if the product fits in the stock
                    if stock_width >= product_width and stock_height >= product_height:
                        # Find placement position
                        placement_position = self.find_placement(stock, product_size)
                        if placement_position is not None:
                            return {
                                "stock_idx": stock_idx,
                                "size": product_size,
                                "position": placement_position
                            }

        # Return default action if no suitable stock is found
        return {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (0, 0)
        }

    def find_placement(self, stock, product_size):
        """
        Function to find the placement position for a product in a stock.
        Checks all possible positions.
        """
        stock_width, stock_height = self._get_stock_size_(stock)
        product_width, product_height = product_size

        # Check if the product fits in the stock
        if stock_width < product_width or stock_height < product_height:
            return None

        # Iterate through all possible positions to place the product in the stock
        for y in range(stock_height - product_height + 1):
            for x in range(stock_width - product_width + 1):
                if self._can_place_(stock, (x, y), product_size):
                    return (x, y)
        return None

    
    ##########################################################################
    ############################GeneticPolicy#################################
    ##########################################################################
    
class GeneticPolicy(Policy):
    def __init__(self, population_size=90, generations=9, mutation_rate=0.001, penaty=0.3):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.parent1 =[]
        self.parent2 =[]
        self.penaty = penaty
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
            if (stock_w >= size[0]) and (stock_h >= size[1]):
                
                for i in range(stock_w - size[0] + 1):
                    for j in range(stock_h - size[1] + 1):
                        if self._can_place_(stock, (i, j), size):
                            
                            pos_x, pos_y = i, j
                            break
                    if pos_x is not None and pos_y is not None:
                        break
                if pos_x is not None and pos_y is not None:
                    break
            if (stock_w >= size[1]) and (stock_h >= size[0]): # xoay chiều 
                for i in range(stock_w - size[1] + 1):
                    for j in range(stock_h - size[0] + 1):
                        if self._can_place_(stock, (i, j), size[::-1]):
                            size = size[::-1]
                            pos_x, pos_y = i, j
                            break
                    if pos_x is not None and pos_y is not None:
                        break
                if pos_x is not None and pos_y is not None:
                    break
                
        return {'stock_idx': stock_idx, 'size': size, 'position': (pos_x, pos_y)}

    def evaluate_fitness(self, action, observation, info):
        stock_data = observation['stocks'][action['stock_idx']]
        product_size = action['size']
        pos_x, pos_y = action['position']
        if not self._can_place_(stock_data, (pos_x, pos_y), product_size):
            return -2000  # trả về giá trị vô cùng nhỏ
        stock_width, stock_height = self._get_stock_size_(stock_data)
        stock_area = stock_width * stock_height
        product_area = product_size[0] * product_size[1]
        if stock_area == 0:
            stock_area = 1
        total_unused_area = stock_area - product_area
        
        demand = [product['quantity'] for product in observation['products']]
        product_sizes = [product['size'] for product in observation['products']]

    # Sản phẩm được đáp ứng (dựa trên hành động hiện tại)
        provided = np.zeros(len(demand))
        for idx, size in enumerate(product_sizes):
            if (product_size == size).all() or (product_size[::-1] == size).all():
                provided[idx] += 1  # Cung cấp một sản phẩm từ hành động
            break

    # Tính số lượng sản phẩm không được đáp ứng
        unsupplied_sum = 0
        for i, d in enumerate(demand):
            unsupplied = max(0, d - provided[i])
            unsupplied_sum += unsupplied * product_sizes[i][0] * product_sizes[i][1]
        fitness_score = 0.7 * (1 - total_unused_area / stock_area) - self.penaty * (unsupplied_sum / sum(demand))
        return fitness_score


    def select_population(self, population, fitness_scores):
        sorted_population = [action for _, action in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]
        return sorted_population[:len(sorted_population)//2]

   
    
    def crossover(self, selected):
        offspring = []
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(selected, 2)
            num = random.random()
            if num < 0.25:
                child = {
                    'stock_idx': parent1['stock_idx'],
                    'size': parent1['size'],
                    'position': parent1['position']
                }
            if 0.25 <= num < 0.5:
                child = {
                    'stock_idx': parent2['stock_idx'],
                    'size': parent1['size'],
                    'position': parent2['position']
                }
            if 0.5<= num < 0.75:
                child = {
                    'stock_idx': parent1['stock_idx'],
                    'size': parent2['size'],
                    'position': parent1['position']
                }
            else:
                child = {
                    'stock_idx': parent2['stock_idx'],
                    'size': parent2['size'],
                    'position': parent2['position']
                }
            offspring.append(child)
        return offspring

    def mutate(self, offspring):
        for action in offspring:
            if random.random() < self.mutation_rate:
                mutation_type = random.choice(['size', 'position'])
                if mutation_type == 'size':
                    size = list(action['size'])
                    size[0] += random.randint(-1, 1)
                    size[1] += random.randint(-1, 1)
                    action['size'] = tuple(max(1, x) for x in size)  # Đảm bảo kích thước hợp lệ
                elif mutation_type == 'position':
                    pos_x, pos_y = action['position']
                    pos_x += random.randint(-1, 1)
                    pos_y += random.randint(-1, 1)
                    action['position'] = (max(0, pos_x), max(0, pos_y))
                
        return offspring
    