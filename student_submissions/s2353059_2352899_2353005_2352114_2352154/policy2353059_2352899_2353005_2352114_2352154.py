from policy import Policy
import random
import numpy as np


class Policy2353059_2352899_2353005_2352114_2352154(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = FirstFitPolicy()
        elif policy_id == 2:
            self.policy = GeneticPolicy()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed
class FirstFitPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.current_stock_idx = None  
        self.last_position = None     

    def sort_products(self, products):
        return sorted(products, key=lambda p: -p["size"][0] * p["size"][1])

    def find_first_fit(self, stock, product_size):
        
        stock_height, stock_width = stock.shape
        product_height, product_width = product_size

        best_position = None
        min_waste_area = float("inf")

        for i in range(stock_height - product_height + 1):
            for j in range(stock_width - product_width + 1):
                if (stock[i:i + product_height, j:j + product_width] == -1).all():
                    # Calculate waste area
                    waste_area = ((stock[i + product_height:, j:j + product_width] == -1).sum() +
                                  (stock[i:i + product_height, j + product_width:] == -1).sum())

                    if waste_area < min_waste_area:
                        min_waste_area = waste_area
                        best_position = (i, j)

        return best_position

    def get_action(self, observation, info):
        
        stocks = observation["stocks"]
        products = observation["products"]

        sorted_products = self.sort_products(products)

        for product in sorted_products:
            product_size = product["size"]
            quantity = product["quantity"]

            if quantity <= 0:
                continue

            for stock_index, stock in enumerate(stocks):
                placement = self.find_first_fit(stock, product_size)
                if placement:

                    self.current_stock_idx = stock_index
                    self.last_position = placement

                    return {
                        "stock_idx": stock_index,
                        "position": placement,
                        "size": product_size,
                    }

            
                rotated_size = product_size[::-1]  # Rotate dimensions
                placement = self.find_first_fit(stock, rotated_size)
                if placement:
                    self.current_stock_idx = stock_index
                    self.last_position = placement

                    return {
                        "stock_idx": stock_index,
                        "position": placement,
                        "size": rotated_size,
                    }

        self.current_stock_idx = None
        self.last_position = None
        return {
            "stock_idx": -1,
            "position": (0, 0),
            "size": (0, 0),
        }
    
class GeneticPolicy(Policy):
    def __init__(self, population_size=5, mutation_rate=0.2, crossover_rate=0.9, generations=10):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations

    def get_action(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]

        population = self.initialize_population(stocks, products)

        for generation in range(self.generations):
            fitness_scores = self.evaluate_population_fitness(population, stocks)
            elite_count = max(1, int(0.1 * self.population_size))

            elite_individuals = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)[:elite_count]
            parents = self.select_parents(population, fitness_scores)
            offspring = self.crossover(parents, products, stocks)
            population = self.mutation(offspring, stocks)

            population[:elite_count] = [individual for individual, _ in elite_individuals]

            if len(set(fitness_scores)) == 1:
                break

        best_individual = max(population, key=lambda ind: self.evaluate_fitness(ind, stocks))
        for placement in best_individual:
            stock_idx = placement["stock_idx"]
            position = placement["position"]
            product_size = placement["size"]

            if 0 <= stock_idx < len(stocks) and position and all(product_size):
                return {
                    "stock_idx": stock_idx,
                    "position": position,
                    "size": product_size,
                }

        return {
            "stock_idx": -1,
            "position": (0, 0),
            "size": (0, 0),
        }

    def initialize_population(self, stocks, products):
        return [self.generate_heuristic_placement(stocks, products) for _ in range(self.population_size)]

    def generate_heuristic_placement(self, stocks, products):
        placement = []
        for product in products:
            if product["quantity"] > 0:
                product_size = product["size"]
                for stock_index, stock in enumerate(stocks):
                    placement_position = self.find_placement(stock, product_size)
                    if placement_position:
                        placement.append({
                            "stock_idx": stock_index,
                            "position": placement_position,
                            "size": product_size
                        })
                        break
        return placement

    def find_placement(self, stock, product_size):
        rows, cols = stock.shape
        prod_rows, prod_cols = product_size

        available_positions = np.argwhere(stock == -1)
        for row, col in available_positions:
            if row + prod_rows <= rows and col + prod_cols <= cols:
                if np.all(stock[row:row + prod_rows, col:col + prod_cols] == -1):
                    return row, col
        return None

    def evaluate_population_fitness(self, population, stocks):
        return [self.evaluate_fitness(ind, stocks) for ind in population]

    def evaluate_fitness(self, individual, stocks):
        total_area_used = 0
        for placement in individual:
            stock_idx = placement["stock_idx"]
            position = placement["position"]
            product_size = placement["size"]

            if stock_idx < 0 or stock_idx >= len(stocks):
                continue

            stock = stocks[stock_idx]
            row, col = position
            prod_rows, prod_cols = product_size

            if (
                row + prod_rows <= stock.shape[0]
                and col + prod_cols <= stock.shape[1]
                and np.all(stock[row:row + prod_rows, col:col + prod_cols] == -1)
            ):
                total_area_used += prod_rows * prod_cols
        return total_area_used

    def select_parents(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choices(population, k=len(population))
        probabilities = [score / total_fitness for score in fitness_scores]
        return random.choices(population, probabilities, k=len(population))

    def crossover(self, parents, products, stocks):
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents) and random.random() < self.crossover_rate:
                p1, p2 = parents[i], parents[i + 1]
                child1 = [random.choice([g1, g2]) for g1, g2 in zip(p1, p2)]
                offspring.append(child1)
            else:
                offspring.append(parents[i])
        return offspring

    def mutation(self, population, stocks):
        for individual in population:
            if random.random() < self.mutation_rate:
                idx = random.randint(0, len(individual) - 1)
                stock_idx = individual[idx]["stock_idx"]
                product_size = individual[idx]["size"]

                if 0 <= stock_idx < len(stocks):
                    stock = stocks[stock_idx]
                    new_position = self.find_placement(stock, product_size)
                    if new_position:
                        individual[idx] = {
                            "stock_idx": stock_idx,
                            "position": new_position,
                            "size": product_size,
                        }
        return population
