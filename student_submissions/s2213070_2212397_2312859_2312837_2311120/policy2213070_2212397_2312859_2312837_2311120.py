from policy import Policy
import numpy as np


class Policy2213070_2212397_2312859_2312837_2311120(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.policy = 1
        elif policy_id == 2:
            self.policy = 2

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1 and self.policy:
            return self.Genetic(observation, info)

        elif self.policy_id == 2:
            return self.Greed(observation, info)
        return None

    def Greed(self, observation, info):
        """
        Greedy algorithm that prioritizes filling the most under-utilized stock 
        with the largest products that fit.
        """
        # Sort products by area in decreasing order
        sorted_products = sorted(
            observation["products"], 
            key=lambda prod: prod["size"][0] * prod["size"][1], 
            reverse=True
        )
        
        # Sort stocks by their current utilization (minimally utilized stocks first)
        stocks = list(enumerate(observation["stocks"]))
        stocks.sort(key=lambda s: np.sum(s[1] != -2), reverse=False)
        
        for product in sorted_products:
            if product["quantity"] <= 0:
                continue  # Skip products with no quantity left
            
            prod_w, prod_h = product["size"]
            
            # Find the first stock that can fit the product
            for stock_idx, stock in stocks:
                stock_w, stock_h = self._get_stock_size_(stock)
                
                for rotate in [False, True]:
                    if rotate:
                        prod_w, prod_h = prod_h, prod_w
                    
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                    # Return the action with placement details
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": (prod_w, prod_h),
                                        "position": (x, y),
                                    }
        
        # If no placement is possible
        return {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (None, None),
        }


    def Genetic(self, observation, info):
        """
        Genetic algorithm that evolves placement plans for products.
        """
        population_size = 20
        num_generations = 30
        mutation_rate = 0.2
        
        # Step 1: Generate the initial population (random placement plans)
        population = [self.random_placement(observation) for _ in range(population_size)]
        
        for generation in range(num_generations):
            # Step 2: Evaluate fitness for the population
            fitness_scores = [self.evaluate_plan_fitness(plan, observation) for plan in population]
            
            # Step 3: Select parents based on fitness
            parents = self.select_parents(population, fitness_scores)
            
            # Step 4: Apply crossover to generate offspring
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    offspring.extend(self.crossover_plans(parents[i], parents[i + 1]))
            
            # Step 5: Mutate offspring
            offspring = [self.mutate_plan(child, mutation_rate, observation) for child in offspring]
            
            # Step 6: Combine parents and offspring for the next generation
            population = parents + offspring
        
        # Step 7: Return the best solution
        best_idx = np.argmax([self.evaluate_plan_fitness(plan, observation) for plan in population])
        return self.convert_plan_to_action(population[best_idx], observation)

    def random_placement(self, observation):
        """
        Generates a random placement plan for the products.
        """
        placement_plan = []
        for product in observation["products"]:
            if product["quantity"] > 0:
                placement_plan.append({
                    "product": product,
                    "stock_idx": np.random.randint(0, len(observation["stocks"])),
                    "rotate": np.random.choice([True, False])
                })
        return placement_plan
    
    def evaluate_plan_fitness(self, plan, observation):
        """
        Evaluates the fitness of a placement plan based on trim loss.
        Lower trim loss results in higher fitness.
        """
        total_trim_loss = 0
        for placement in plan:
            stock_idx = placement["stock_idx"]
            stock = observation["stocks"][stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            
            prod_w, prod_h = placement["product"]["size"]
            if placement["rotate"]:
                prod_w, prod_h = prod_h, prod_w
            
            if stock_w >= prod_w and stock_h >= prod_h:
                trim_loss = (stock_w * stock_h) - (prod_w * prod_h)
                total_trim_loss += trim_loss
            else:
                total_trim_loss += float("inf")  # Penalize invalid placements
        
        return -total_trim_loss  # Lower trim loss means higher fitness

    def crossover_plans(self, plan1, plan2):
        """
        Crossover two plans to create offspring.
        """
        split_idx = len(plan1) // 2
        child1 = plan1[:split_idx] + plan2[split_idx:]
        child2 = plan2[:split_idx] + plan1[split_idx:]
        return [child1, child2]
    
    def mutate_plan(self, plan, mutation_rate, observation):
        """
        Mutates a plan by randomly altering some placements.
        """
        for placement in plan:
            if np.random.rand() < mutation_rate:
                placement["stock_idx"] = np.random.randint(0, len(observation["stocks"]))
                placement["rotate"] = not placement["rotate"]
        return plan

    
    def convert_plan_to_action(self, plan, observation):
        """
        Converts the best placement plan into an action for a single product.
        """
        for placement in plan:
            product = placement["product"]
            if product["quantity"] > 0:
                stock_idx = placement["stock_idx"]
                stock = observation["stocks"][stock_idx]
                
                prod_w, prod_h = product["size"]
                if placement["rotate"]:
                    prod_w, prod_h = prod_h, prod_w
                
                stock_w, stock_h = self._get_stock_size_(stock)
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                            return {
                                "stock_idx": stock_idx,
                                "size": (prod_w, prod_h),
                                "position": (x, y),
                            }
        return {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (None, None),
        }
    
    def select_parents(self, population, fitness_scores):
        """
        Selects parents from the population using fitness scores.
        Uses a roulette wheel (proportional) selection method.
        """
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:  # Avoid division by zero
            probabilities = [1 / len(fitness_scores)] * len(fitness_scores)
        else:
            probabilities = [fitness / total_fitness for fitness in fitness_scores]

        # Select parents based on their probabilities
        num_parents = len(population) // 2
        selected_indices = np.random.choice(
            range(len(population)), size=num_parents, replace=False, p=probabilities
        )
        return [population[i] for i in selected_indices]

    # Student code here
    # You can add more functions if needed