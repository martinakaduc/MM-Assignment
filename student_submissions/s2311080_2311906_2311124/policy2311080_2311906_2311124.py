from policy import Policy
import numpy as np
import random


class Policy2311080_2311906_2311124(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Student code here
        if policy_id == 1:
            self.policy = GeneticAlgorithm(100, 8, 0.01)
        elif policy_id == 2:
            self.policy = BestShortSideFit()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)
    
# BEGIN OF GENETIC ALGORITHM CLASS
class GeneticAlgorithm(Policy):
    def __init__(self, population_size =100, gen_size =8, mutation_rate =0.01):
        self.population_size = population_size
        self.gen_size = gen_size
        self.mutation_rate = mutation_rate
        self.bssf = GA_BSSF_SP()

    def get_action(self, observation, info):
        """ 
            Step 1: Initialize population by using bestshortsidefit helper, each individual is a chromosome
            Step 2: Calculating fitness of each individual in the population
            Step 3: Selection of parents based on fitness_score
            Step 4: Crossover of parents to produce offspring
            Step 5: Mutation of offspring for diversity population
            Step 6: Replacement of least fit individuals with new offspring                
        """
        population = [self.init_population(observation, info) for _ in range(self.population_size)]
        for _ in range(self.gen_size):
            fitness = self.fitness_evaluation(population, observation)
            parents = self.selection_tournament(population, fitness, 3)
            offspring = self.crossover(parents)
            offspring = self.mutate(offspring)
            population = self.match_population(population, fitness, 3, offspring)
        # Calculating fitness to avoid collision  and choose priority_chromosome
        fitness = self.fitness_evaluation(population, observation)
        priority_chromosome = population[np.argmax(fitness)]
        return priority_chromosome
    
    def init_population(self, observation: dict, info: dict) -> dict:
        """
            - Randomly choose product
            - Randomly choose stock
            - Using bssf to place product in stock 
        Args:
            observation (dict): observation_space to get information of stocks and products
            info (dict): _description_

        Returns:
            dict: One possible chromosome
        """        
        return self.bssf.get_action(observation, info)
    
    def fitness_evaluation(self, population: list, observation: dict) -> list:
        """_summary_
            Fitness function: FF = space_ultilization  + distance_to_target + used_area ** 2 - 0.005*penalty (-1000 if invalid position else 0)
            The FF have trend to fulfill stock by product 

        Args:
            population (list): List of chromosomes
            observation (spaces.Dict): _description_
            info (dict): _description_

        Returns:
            list: fitness_scores list
        """        
        fitness_scores = []
        for chromosome in population:
            stock = observation["stocks"][chromosome["stock_idx"]]
            prod_size = chromosome["size"]
            prod_w, prod_h = prod_size
            pos_x, pos_y = chromosome["position"]
            stock_w, stock_h = self._get_stock_size_(stock)
            
            space_utilization = (prod_w * prod_h) / (stock_w * stock_h)
            target_x, target_y = stock_w // 2, stock_h // 2
            distance_to_target = ((pos_x - target_x) ** 2 + (pos_y - target_y) ** 2) ** 0.5
            used_area = np.sum(stock >= 0)
            penalty = (stock_w * stock_h - used_area)
            
            # i = random.randint(0, 1)
            # penalty = (used_area / stock_w * stock_h)
            # if penalty < 0.5:
            #     fitness_score = (space_utilization + distance_to_target + used_area ** 2 + 0.5*penalty -(1000 if not self._can_place_(stock, (pos_x, pos_y), prod_size) else 0))
            # else:
            #     penalty = (stock_w * stock_h - used_area)
            #     fitness_score = (space_utilization + distance_to_target + used_area ** 2 + 0.01*penalty -(1000 if not self._can_place_(stock, (pos_x, pos_y), prod_size) else 0))
            fitness_score = space_utilization  + distance_to_target + used_area ** 2 - 0.005*penalty - (1000 if not self._can_place_(stock, (pos_x, pos_y), prod_size) else 0)
            fitness_scores.append(fitness_score)
        return fitness_scores

    def selection_tournament(self, population, fitness_score: list, tournament_size: int = 3) -> list:
        """
        Select individuals from the population using tournament selection.

        Args:
            fitness_score (list): A list of fitness scores corresponding to the population.
            tournament_size (int): The number of individuals to compete in each tournament.

        Returns:
            list: A list of selected individuals based on tournament selection.
        """
        parents = []
        population_size = len(fitness_score)

        for _ in range(self.population_size // 2):
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_score[i] for i in tournament_indices]

            best_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            parents.append(population[best_index])

        return parents

    def crossover(self, parents: list) -> list:
        """
            - Choose a quarter offspring from random selected parents.
            - Swap 2 middle genes of 2 selected parents.

        Args:
            parents (list): selected parents

        Returns:
            list: offspring 
        """        
        offspring = []
        while len(offspring) < self.population_size // 4:
            parent1, parent2 = random.sample(parents, 2)
            child1 = {
                "stock_idx": parent1["stock_idx"],
                "size": parent2["size"],
                "position": parent1["position"]
            }
            child2 = {
                "stock_idx": parent2["stock_idx"],
                "size": parent1["size"],
                "position": parent2["position"]
            }
            offspring.append(child1)
            if len(offspring) < self.population_size // 4:
                offspring.append(child2)
        return offspring

    def mutate(self, offspring: list) -> list:
        """
            - Randomly move product.
            - Randomly rotate product.

        Args:
            offspring (list): offspring after doing crossover.

        Returns:
            list: offspring after mutating.
        """        
        for gene in offspring:
            if random.random() < self.mutation_rate:
                position = list(gene["position"])
                position[0] += random.randint(-1, 1)  
                position[1] += random.randint(-1, 1)  
                gene["position"] = tuple(position[::-1])
            if random.random() >= 0.5:
                prod_size = list(gene["size"])
                width_store = prod_size[0]
                prod_size[0] = prod_size[1]
                prod_size[1] = width_store
        return offspring
    
    def match_population(self, population, fitness_score: list, tournament_size: int = 3, offspring=[]) -> list:
        """
            - Choose 3/4 population of old population.
            - Extend 1/4 of offsprings below into new population.

        Args:
            population (_type_): _description_
            fitness_score (list): _description_
            tournament_size (int, optional): _description_. Defaults to 3.
            offspring (list, optional): _description_. Defaults to [].

        Returns:
            list: _description_
        """        
        selected = []
        population_size = len(fitness_score)

        for _ in range(self.population_size*3 // 4):
            # Randomly select individuals for the tournament
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_score[i] for i in tournament_indices]

            # Find the index of the best individual in the tournament
            best_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
            selected.append(population[best_index])
        selected.extend(offspring)
        return selected
    
# END OF GENETIC ALGORITHM CLASS

# BEGIN OF GA_BSSF_SP CLASS
class GA_BSSF_SP(Policy):
    def get_action(self, observation, info):
        prod = random.choice([product for product in observation["products"] if product["quantity"] > 0])
        pos_x, pos_y = -1, -1
        prod_size = prod["size"]
        placed = False

        while not placed:
            prod_size = prod["size"]
            stock_idx = random.randint(0, len(observation["stocks"]) - 1)
            stock = observation["stocks"][stock_idx]
            empty_rectangles = self.get_empty_rectangles(stock)
            empty_rectangles.sort(key=lambda rect: self.score_fit(rect, prod_size))

            for rect in empty_rectangles:
                rect_x, rect_y, rect_w, rect_h = rect
                prod_w, prod_h = prod_size

                # Check if the product can fit in the rectangle, considering rotation
                if (prod_w <= rect_w and prod_h <= rect_h) or (prod_h <= rect_w and prod_w <= rect_h):
                    if self._can_place_(stock, (rect_x, rect_y), prod_size):
                        pos_x, pos_y = rect_x, rect_y
                        placed = True
                        break

            if placed:
                break   

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def get_empty_rectangles(self, stock):
        rectangles = []
        stock_w, stock_h = self._get_stock_size_(stock)
        visited = [[False for _ in range(stock_h)] for _ in range(stock_w)]

        for y in range(stock_h):
            for x in range(stock_w):
                if stock[x, y] == -1 and not visited[x][y]:
                    rect = self.find_max_rect(stock, x, y, visited)
                    rectangles.append(rect)
        return rectangles

    def find_max_rect(self, stock, x, y, visited):
        stock_w, stock_h = self._get_stock_size_(stock)

        max_width = 0
        for i in range(x, stock_w):
            if stock[i, y] == -1:
                max_width += 1
            else:
                break

        max_height = 0
        for j in range(y, stock_h):
            if all(stock[i, j] == -1 for i in range(x, x + max_width)):
                max_height += 1
            else:
                break

        for i in range(x, x + max_width):
            for j in range(y, y + max_height):
                visited[i][j] = True

        return (x, y, max_width, max_height)

    def score_fit(self, rect, prod_size):
        rect_w, rect_h = rect[2], rect[3]
        prod_w, prod_h = prod_size
        
        # Calculate the "short side fit" and "long side fit" scores
        short_side = min(abs(rect_w - prod_w), abs(rect_h - prod_h))
        long_side = max(abs(rect_w - prod_w), abs(rect_h - prod_h))

        return short_side, long_side
# END OF GA_BSSF_SP CLASS

# BEGIN OF BESTSHORTSIDEFIT
class BestShortSideFit(Policy):
    def get_action(self, observation, info):
        list_prods = observation["products"]
        pos_x, pos_y = -1, -1
        prod_size = [0, 0]
        stock_idx = -1

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    empty_rectangles = self.get_empty_rectangles(stock)
                    empty_rectangles.sort(key=lambda rect: self.score_fit(rect, prod_size))

                    placed = False
                    for rect in empty_rectangles:
                        rect_x, rect_y, rect_w, rect_h = rect
                        prod_w, prod_h = prod_size

                        # Check if the product can fit in the rectangle, considering rotation
                        if (prod_w <= rect_w and prod_h <= rect_h) or (prod_h <= rect_w and prod_w <= rect_h):
                            if self._can_place_(stock, (rect_x, rect_y), prod_size):
                                pos_x, pos_y = rect_x, rect_y
                                stock_idx = i
                                placed = True
                                break

                    if placed:
                        break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def get_empty_rectangles(self, stock):
        rectangles = []
        stock_w, stock_h = self._get_stock_size_(stock)
        visited = [[False for _ in range(stock_h)] for _ in range(stock_w)]

        for y in range(stock_h):
            for x in range(stock_w):
                if stock[x, y] == -1 and not visited[x][y]:
                    rect = self.find_max_rect(stock, x, y, visited)
                    rectangles.append(rect)
        return rectangles

    def find_max_rect(self, stock, x, y, visited):
        stock_w, stock_h = self._get_stock_size_(stock)

        max_width = 0
        for i in range(x, stock_w):
            if stock[i, y] == -1:
                max_width += 1
            else:
                break

        max_height = 0
        for j in range(y, stock_h):
            if all(stock[i, j] == -1 for i in range(x, x + max_width)):
                max_height += 1
            else:
                break

        for i in range(x, x + max_width):
            for j in range(y, y + max_height):
                visited[i][j] = True

        return (x, y, max_width, max_height)

    def score_fit(self, rect, prod_size):
        rect_w, rect_h = rect[2], rect[3]
        prod_w, prod_h = prod_size
        
        # Calculate the "short side fit" and "long side fit" scores
        short_side = min(abs(rect_w - prod_w), abs(rect_h - prod_h))
        long_side = max(abs(rect_w - prod_w), abs(rect_h - prod_h))

        return short_side, long_side
# END OF BESTSHORTSIDEFIT
