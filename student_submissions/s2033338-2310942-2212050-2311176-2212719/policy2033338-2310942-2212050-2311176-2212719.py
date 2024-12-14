from policy import Policy
import numpy as np
import random

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = ColumnGerenation()
        elif policy_id == 2:
            self.policy = GeneticAlgorithm()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed



class GeneticAlgorithm(Policy):
    def __init__(self, population_size=20, generations=50, mutation_rate=0.1):
        '''
            """
            Initialize the genetic algorithm with population size, number of generations, and mutation rate.
            - population_size: Number of chromosomes in the population.

            Args:
                population_size (int): Number of chromosomes in the population.
                generations (int): Number of iterations to evolve the population.
                mutation_rate (float): Probability of mutation for each gene in a chromosome.

            Attributes:
                population_size (int): Number of chromosomes in the population.
                generations (int): Number of iterations to evolve the population.
                mutation_rate (float): Probability of mutation for each gene in a chromosome.
                stocks (list): Available stocks.
                products (list): Products to arrange.
            """
            - generations: Number of iterations to evolve the population.
            - mutation_rate: Probability of mutation for each gene in a chromosome.
        '''
        self.stocks = None 
        self.products = None 
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate


    def is_position_available(self, stock, pos_x, pos_y, prod_w, prod_h):
        '''
            Check if a product can be placed at a specific position in a stock.
            - Verifies that the product dimensions fit within the stock dimensions.
            - Checks if the product overlaps with other placed items.

            Args:
                stock: The current stock matrix.
                pos_x, pos_y: Top-left coordinates of the placement.
                prod_w, prod_h: Dimensions of the product.

            Returns:
                Tuple: (bool, prod_w, prod_h) indicating availability and product dimensions.
        '''
        stock_w, stock_h = self._get_stock_size_(stock)

        # Check if product exceeds stock boundaries.
        if pos_x + prod_w > stock_w or pos_y + prod_h > stock_h:
            return False, None, None

        # Check if the position is free to place the product.
        if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
            return True, prod_w, prod_h
        
        return False, None, None

    def get_empty_spaces(self, stock):
        '''
            Identify all empty spaces in the stock where products can be placed.
            - Scans the stock to locate contiguous empty areas.

            Args:
                stock: The current stock matrix.

            Returns:
                List of tuples (x, y, width, height) representing empty spaces.
        '''
        empty_spaces = []
        stock_w, stock_h = self._get_stock_size_(stock)

        visited = np.zeros_like(stock, dtype=bool)  # Keep track of visited cells.

        for x in range(stock_w):
            for y in range(stock_h):
                if stock[x, y] == -1 and not visited[x, y]:
                    width, height = 0, 0

                    # Calculate the width of the empty space.
                    for dx in range(x, stock_w):
                        if stock[dx, y] == -1:
                            width += 1
                        else:
                            break

                    # Calculate the height of the empty space.
                    for dy in range(y, stock_h):
                        if stock[x, dy] == -1:
                            height += 1
                        else:
                            break

                    # Mark the area as visited.
                    for dx in range(x, x + width):
                        for dy in range(y, y + height):
                            visited[dx, dy] = True

                    empty_spaces.append((x, y, width, height))

        # Sort spaces by position for consistency.
        return sorted(empty_spaces, key=lambda space: (space[0], space[1]))

    def initialize_population(self):
        '''
            Create an initial population of chromosomes based on product placements.
            - Each chromosome is a list of product placements in stocks.

            Returns:
                List of chromosomes representing the initial population.
        '''
        population = []
        for _ in range(self.population_size):
            chromosome = []
            products_sorted = sorted(self.products, key=lambda p: p["size"][0] * p["size"][1], reverse=True)
            for product in products_sorted:
                if product["quantity"] < 1:
                    continue

                product_width, product_height = product["size"]
                valid_position_found = False

                # Try to place the product in available stocks.
                for stock_idx, stock in enumerate(self.stocks):
                    empty_spaces = self.get_empty_spaces(stock)
                    for x, y, _, _ in empty_spaces: 
                        if self.is_position_available(stock, x, y, product_width, product_height)[0]:
                            chromosome.append((stock_idx, x, y, product_width, product_height))
                            valid_position_found = True
                            break

                    if valid_position_found:
                        break

            if chromosome:
                population.append(chromosome)

        if not population:
            raise ValueError("Failed to initialize population: no valid chromosomes found.")

        return population

    def evaluate_fitness(self, chromosome):
        '''
            Calculate the fitness score of a chromosome based on used area and bonuses.
            - Higher scores indicate better utilization of stock space.

            Args:
                chromosome: A list of product placements in stocks.

            Returns:
                Fitness score (float) of the chromosome.
        '''
        used_area = 0
        edge_bonus = 0
        rotation_bonus = 0

        for stock_idx, x, y, width, height in chromosome:
            stock = self.stocks[stock_idx]

            if self.is_position_available(stock, x, y, width, height)[0]:
                used_area += width * height  # Add area of the product.
                # Bonus for placement along stock edges.
                if x == 0 or y == 0 or x + width == self._get_stock_size_(stock)[0] or y + height == self._get_stock_size_(stock)[1]:
                    edge_bonus += 0.1 * (width * height)

                # Bonus for placing taller products.
                if width < height:
                    rotation_bonus += 0.05 * (width * height)

        total_area = sum(self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for stock in self.stocks)
        return (used_area + edge_bonus + rotation_bonus) / total_area if total_area > 0 else 0

    def selection(self, population, fitness_scores):
        '''
            Select two parent chromosomes using a tournament selection process.
            - Ensures that fitter chromosomes are more likely to be selected.

            Args:
                population: List of chromosomes.
                fitness_scores: List of fitness scores corresponding to each chromosome.

            Returns:
                Two selected parent chromosomes.
        '''
        feasible_indices = [i for i, f in enumerate(fitness_scores) if f > 0]
        if len(feasible_indices) < 2:
            # Fallback to random selection if not enough feasible candidates
            return random.choices(population, k=2)

        tournament_size = 5
        tournament_indices = random.sample(feasible_indices, min(tournament_size, len(feasible_indices)))
        tournament = [(fitness_scores[i], population[i]) for i in tournament_indices]

        parent1 = max(tournament, key=lambda x: x[0])[1]  # Select the fittest.
        tournament.remove(max(tournament, key=lambda x: x[0]))
        parent2 = max(tournament, key=lambda x: x[0])[1]  # Select the next fittest.

        return parent1, parent2

    def crossover(self, parent1, parent2):
        '''
            Perform crossover operation to generate new chromosomes.
            - Combines genetic material from two parents to create offspring.

            Args:
                parent1: First parent chromosome.
                parent2: Second parent chromosome.

            Returns:
                Two offspring chromosomes.
        '''
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1, parent2

        split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2

    def mutate(self, chromosome):
        '''
            Mutate a chromosome by randomly altering some gene positions.
            - Introduces genetic diversity to avoid local optima.

            Args:
                chromosome: Chromosome to mutate.

            Returns:
                Mutated chromosome.
        '''
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                stock_idx = chromosome[i][0]
                stock = self.stocks[stock_idx]
                width, height = chromosome[i][3], chromosome[i][4]

                # Reassign a random valid position.
                empty_spaces = self.get_empty_spaces(stock)
                if empty_spaces:
                    x, y, _, _ = random.choice(empty_spaces)
                    chromosome[i] = (stock_idx, x, y, width, height)
        return chromosome

    def run(self):
        '''
            Execute the genetic algorithm to find the best chromosome.
            - Evolves the population over generations.

            Returns:
                Best chromosome and its fitness score.
        '''
        population = self.initialize_population()
        if not population:
            return None, 0
        
        max_previous_fitness = -1  # Initialize optimal fitness score.

        for generation in range(self.generations):
            # Calculate fitness scores for the current population
            fitness_scores = [self.evaluate_fitness(ch) for ch in population]

            # If the maximum fitness is zero, return no valid solution
            if max(fitness_scores) == 0:
                return None, 0

            new_population = []

            # Perform selection, crossover, and mutation to create a new population
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([self.mutate(child1), self.mutate(child2)])

            # Update population with the new generation
            population = new_population

            # Convergence check: if fitness does not improve, break the loop
            if generation > 0 and max(fitness_scores) == max_previous_fitness:
                break
            max_previous_fitness = max(fitness_scores)

        # Identify the best chromosome and its fitness score
        best_index = np.argmax([self.evaluate_fitness(ch) for ch in population])
        return population[best_index], max([self.evaluate_fitness(ch) for ch in population])

    def get_action(self, observation, info):
        '''
            Get the best action based on the genetic algorithm's solution.
            - Returns the placement details for the best solution found.

            Args:
                observation: The current state of stocks and products.
                info: Additional information (not used here).

            Returns:
                Dictionary with stock index, size, and position for the product placement.
        '''
        self.stocks = observation["stocks"]
        self.products = observation["products"]
        best_chromosome, fitness_score = self.run()

        # If no valid solution, return a default placement
        if not best_chromosome or fitness_score == 0:
            return {"stock_idx": -1, "size": [1, 1], "position": (0, 0)}

        # Extract placement details from the best chromosome
        for stock_idx, x, y, width, height in best_chromosome:
            return {"stock_idx": stock_idx, "size": [width, height], "position": (x, y)}

        # Fallback default placement
        return {"stock_idx": -1, "size": [1, 1], "position": (0, 0)}


class ColumnGerenation(Policy):
    def __init__(self):
        # Initialize attributes:
        # - matrix: Stores pre-generated pattern matrices for each stock.
        # - current_stock: Tracks the current stock index being processed.        
        self.matrix = {}         
        self.current_stock = 0     

    def _generate_matrix_(self, products, stock_size):
        """
            Generate a pattern matrix for arranging products into a stock.
            - Sort products in descending order based on their area.
            - Determine the maximum number of products that can fit into the stock 
            and handle the remaining space.
            
            Args:
                products (list): List of products, each with size and quantity attributes.
                stock_size (tuple): Current stock dimensions (width, height).
            
            Returns:
                list: A list of columns (cutting patterns), where each column represents 
                    a way of arranging products.
        """
        sorted_products = sorted(enumerate(products), 
                                 key=lambda prod: prod[1]['size'][0] * prod[1]['size'][1], 
                                 reverse=True)
        matrix = []

        for i, prod in sorted_products:
            if prod['quantity'] > 0:
                stock_w, stock_h = stock_size
                prod_w, prod_h = prod['size']

                if prod_w > stock_w or prod_h > stock_h:
                    continue

                column = [0] * len(products)
                remain_stock_w, ramain_stock_h = stock_w, stock_h
                element = min(prod['quantity'], (remain_stock_w // prod_w) * (ramain_stock_h // prod_h)) 

                if element > 0:
                    column[i] = element

                    for j, other_prod in sorted_products:
                        if j != i and other_prod['quantity'] > 0:
                            other_prod_w, other_prod_h = other_prod['size']
                            if other_prod_w <= remain_stock_w and other_prod_h <= ramain_stock_h:
                                num_of_other = min(
                                    other_prod['quantity'],
                                    (remain_stock_w // other_prod_w) * (ramain_stock_h // other_prod_h)
                                )
                                if num_of_other > 0:
                                    column[j] = num_of_other
                                    remain_stock_w -= other_prod_w * num_of_other
                                    ramain_stock_h -= other_prod_h * num_of_other

                    matrix.append(column)

        return sorted(matrix, key=lambda column: self._evaluate_column_(column, products), reverse=True)

    def _evaluate_column_(self, column, products):
        """
            Evaluate the quality of a pattern based on area and product quantity.
            
            Args:
                column (list): A pattern representing the quantity of each product.
                products (list): List of products.
            
            Returns:
                float: A score representing the quality of the pattern.
        """
        total_area = 0
        total_quantity = 0

        for prod_idx, element in enumerate(column):
            if element > 0:
                prod = products[prod_idx]
                total_area += element * prod['size'][0] * prod['size'][1]
                total_quantity += element
        
        # Calculate the score, prioritizing area used (80%) and product quantity (20%)
        return total_area * 0.8 + total_quantity * 0.2
    
    def _is_position_available_(self, stock, product_size):
        """
            Find an available position to place the product in the stock.
            
            Args:
                stock (np.array): Matrix representing the current stock.
                product_size (tuple): Dimensions of the product (width, height).
            
            Returns:
                tuple or None: A suitable position to place the product, or None if unavailable.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = product_size

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    return (x, y)  

        return None

    def get_action(self, observation, info):
        """
            Determine the next action based on the current observation state.
            
            Returns:
                dict: The selected action, including stock index, product size, and position.
        """
        stocks = observation["stocks"]  
        products = observation["products"] 

        sorted_stocks = sorted(
            enumerate(stocks),
            key=lambda stock: self._get_stock_size_(stock[1])[0] * self._get_stock_size_(stock[1])[1],
            reverse=True 
        )

        for idx, cur_stock in sorted_stocks[self.current_stock:]:
            stock_size = self._get_stock_size_(cur_stock)  

            if idx not in self.matrix:
                self.matrix[idx] = self._generate_matrix_(products, stock_size)

            for column in self.matrix[idx]:
                for prod_idx, element in enumerate(column):
                    if prod_idx < len(products) and element > 0 and products[prod_idx]["quantity"] > 0:
                        position = self._is_position_available_(
                            cur_stock,
                            products[prod_idx]["size"]
                        )

                        if position:
                            return {
                                "stock_idx": idx,  
                                "size": products[prod_idx]["size"],  
                                "position": position 
                            }

            self.current_stock += 1
            if self.current_stock == len(stocks):
                self.current_stock = 0
                self.matrix.clear()
                
        # Return a no-action response if all stocks have been processed
        return { "stock_idx": -1, "size": [0, 0], "position": (0, 0) }



