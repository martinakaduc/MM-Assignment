from policy import Policy
import numpy as np
import random
 
class BestFitPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        num_products = len([prod for prod in list_prods if prod["quantity"] > 0])

        if 1 <= num_products <= 100:
            sorted_prods = sorted(
                list_prods,
                key=lambda prod: (
                    prod["size"][0] * prod["size"][1],  
                ),
                reverse=True
            )
        else:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # initialize value
        stock_idx, pos_x, pos_y = -1, 0, 0
        prod_size = [0, 0]
        best_waste = float('inf') 


        for prod in sorted_prods:
            if prod["quantity"] > 0:
                original_size = prod["size"]

                # check rotating case and non-rotating case 
                for size in [original_size, original_size[::-1]]:
                    if not self._fit_no_rotate(observation, size):
                        continue

                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        if stock_w < size[0] or stock_h < size[1]:
                            continue

                        for x, y in self._get_possible_positions(stock, size):
                            if self._can_place_(stock, (x, y), size):
                                waste = (stock_w * stock_h) - (size[0] * size[1])
                                if waste < best_waste:
                                    best_waste = waste
                                    stock_idx, pos_x, pos_y = i, x, y
                                    prod_size = size

                                    if waste == 0:
                                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

                if stock_idx != -1:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def _fit_no_rotate(self, observation, prod_size):
        for stock in observation["stocks"]:
            stock_w, stock_h = self._get_stock_size_(stock)
            if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                return True
        return False

    def _get_possible_positions(self, stock, size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = size

        for x in range(0, stock_w - prod_w + 1):
            for y in range(0, stock_h - prod_h + 1):
                yield x, y

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))  
        stock_h = np.sum(np.any(stock != -2, axis=0))  
        return stock_w, stock_h
    
    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position 
        prod_w, prod_h = prod_size 

        if pos_x + prod_w > stock.shape[0] or pos_y + prod_h > stock.shape[1]:
            return False
        
        return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)

class GeneticAlgorithmPolicy(Policy):
    def __init__(self, population_size=5, generations=10, alpha=0.7, beta=0.3):
        # initialize parameters
        self.population_size = population_size
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.population = []

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        stock_x, stock_y = position
        if 0 <= stock_x <= stock_w - prod_w and 0 <= stock_y <= stock_h - prod_h:
            return True
        return False

    def _initialize_population(self, observation):
        # initialize population
        population = []
        for _ in range(self.population_size):
            chromosome = self._generate_random_chromosome(observation)
            population.append(chromosome)
        return population

    def _generate_random_chromosome(self, observation):
        # generate random chromosome (solution)
        chromosome = []
        for prod in observation["products"]:
            if prod["quantity"] > 0:
                stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                position = (random.randint(0, stock_w - prod["size"][0]), 
                            random.randint(0, stock_h - prod["size"][1]))
                chromosome.append({"stock_idx": stock_idx, "position": position, "prod_size": prod["size"]})
        return chromosome

    def _evaluate_population(self, observation):
        # evaluate each chromosome (solution) and calculate fitness score (performance)
        fitness_scores = []
        for chromosome in self.population:
            used_stock = set()
            total_waste = 0

            for gene in chromosome:
                stock_idx = gene["stock_idx"]
                position = gene["position"]
                prod_size = gene["prod_size"]
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size

                if self._can_place_(stock, position, prod_size):
                    used_stock.add(stock_idx)
                else:
                    waste_area = (stock_w * stock_h) - (prod_w * prod_h)
                    total_waste += waste_area

            N = len(used_stock)
            W = total_waste

            if N == 0:
                fitness_score = 0
            else:
                fitness_score = self.alpha * (1 / N) + self.beta * (1 / (1 + W))
            
            fitness_scores.append(fitness_score)

        return fitness_scores

    def _select_parents(self, fitness_scores):
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(self.population), random.choice(self.population)

        probabilities = [score / total_fitness for score in fitness_scores]
        parent1, parent2 = random.choices(self.population, probabilities, k=2)
        return parent1, parent2

    def _crossover(self, parent1, parent2):
        if len(parent1) == 1 or len(parent2) == 1:
            return parent1, parent2

        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def _mutate(self, chromosome, observation):
        if len(chromosome) == 0:
            return chromosome

        mutation_point = random.randint(0, len(chromosome) - 1)
        chromosome[mutation_point]["stock_idx"] = random.randint(0, len(observation["stocks"]) - 1)
        return chromosome

    def _create_new_generation(self, fitness_scores, observation):
        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), key=lambda x: x[0], reverse=True)]

        elitism_count = int(self.population_size * 0.2)  
        new_population.extend(sorted_population[:elitism_count])
    
   
        while len(new_population) < self.population_size:
              parent1, parent2 = self._select_parents(fitness_scores)
              child1, child2 = self._crossover(parent1, parent2)
              new_population.append(self._mutate(child1, observation))
              new_population.append(self._mutate(child2, observation))

        self.population = new_population[:self.population_size]

    def run(self, observation):
        self.population = self._initialize_population(observation)

        for generation in range(self.generations):
            fitness_scores = self._evaluate_population(observation)
            self._create_new_generation(fitness_scores, observation)

        best_fitness = max(fitness_scores)
        best_chromosome = self.population[fitness_scores.index(best_fitness)]
        return best_chromosome

    def get_action(self, observation, info):
        best_chromosome = self.run(observation)
        best_gene = best_chromosome[0]
        action = {
            "stock_idx": best_gene["stock_idx"],
            "position": best_gene["position"],
            "size": best_gene["prod_size"]
        }

        return action


class Policy2313866_2313867_2310975_2310554_2313017(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        self.genetic_algorithm_policy = GeneticAlgorithmPolicy()
        self.best_fit_policy = BestFitPolicy()

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.best_fit_policy.get_action(observation, info)
        if self.policy_id == 2:
            return self.genetic_algorithm_policy.get_action(observation,info)