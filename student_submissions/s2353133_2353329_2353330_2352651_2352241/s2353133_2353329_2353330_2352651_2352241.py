from policy import Policy
from collections import deque
import numpy as np
import copy

class Policy2353133_2353329_2353330_2352651_2352241(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = Bottom_left_fill()
        elif policy_id == 2:
            self.policy = Genetic_algorithm()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed

"""
This is the bottom-left fill policy for the cutting stock problem.
The policy picks a product that has quantity > 0, then it loops through all stocks to find the best position to place the product.
The product is rotated if it can't fit in the stock.
The policy returns the action with the stock index, product size, and position.
"""

class Bottom_left_fill(Policy):
    def __init__(self):
        pass
    def get_action(self, observation, info):
        # Sort the products based on the area in descending order to optimize the placement (this is extra part more than the report)
        products_list = list(observation["products"])
        products_list.sort(key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        observation["products"] = products_list
            
        # Pick a product that has quantity > 0
        for prod in observation["products"]:
            if prod["quantity"] > 0:
                original_size = prod["size"]
                rotated_size = (original_size[1], original_size[0])  # Rotated dimensions
                possible_orientations = [original_size, rotated_size]

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for orientation in possible_orientations:
                        prod_w, prod_h = orientation

                        if stock_w < prod_w or stock_h < prod_h:
                            continue

                        pos_x, pos_y = -1, -1
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), orientation):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x != -1 and pos_y != -1:
                                break

                        if pos_x != -1 and pos_y != -1:
                            prod_size = orientation
                            break

                    if pos_x != -1 and pos_y != -1:
                        stock_idx = i
                        break

                if pos_x != -1 and pos_y != -1:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
"""
This is the genetic algorithm policy for the cutting stock problem.
The policy generates the best combination of products to fit in the stock using the genetic algorithm.
Generates the initial population of combinations, then it calculates the fitness of each combination (actually the unused area because the fitness is inversely proportional to the unused area).
Generates the next generations by mutation, the mutation is done by deleting a random action from the half best combination and replace it with a new actions until we try out all available products.
Repeats the process for the next generations.
Keeps the best combinations and inserts them to the output queue.
Take the next action from the output queue by each call of the get_action function.
When the output queue is empty, generate the new combination for the next stock and repeat the process.
"""
class Genetic_algorithm(Policy):
    def __init__(self):
        self.population_size = 4 # Should be even number
        self.num_generations = 2
        self.output_queue = deque()
        self.best_combination = []
        self.current_stock_index = -1


    def generate_combination(self, observation):
        combination = []
        current_stock = copy.deepcopy(observation["stocks"][self.current_stock_index])
        available_products = copy.deepcopy(observation["products"])
        remaining_products = available_products # Track original products

        # Get the total number of products
        number_of_products = 0
        for prod in available_products:
            number_of_products += prod["quantity"]
        
        # Fill the stock using bottom-left fill algorithm in random order until try out all available_products, the product is rotated if it can't fit in the stock
        for _ in range(number_of_products):
            # Get indices of products with quantity > 0
            valid_indices = []
            for i, prod in enumerate(remaining_products):
                if prod["quantity"] > 0:
                    valid_indices.append(i)

            if valid_indices:  # If there are valid products
                # Get random index from valid products
                prod_idx = valid_indices[np.random.randint(len(valid_indices))]
                prod = available_products[prod_idx]
            else:
                # No valid products left
                break

            placed = False
            stock_w, stock_h = self._get_stock_size_(current_stock)
            original_size = prod["size"]
            rotated_size = (original_size[1], original_size[0])

            for orientation in [original_size, rotated_size]:
                prod_w, prod_h = orientation
                if stock_w < prod_w or stock_h < prod_h:
                    continue

                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(current_stock, (x, y), orientation):
                            current_stock[x:x + prod_w, y:y + prod_h] = 1
                            combination.append({
                                "stock_idx": self.current_stock_index,
                                "size": orientation,
                                "position": (x, y)
                            })
                            
                            prod["quantity"] -= 1  # Decrease quantity
                            remaining_products[prod_idx]["quantity"] -= 1  # Decrease quantity in original products
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
        return {
        "combination": combination,
        "remaining_products": remaining_products
        }

    def get_unused_area(self, population, observation):
        unused_area = []
        # Get the total area of the stock
        stock = copy.deepcopy(observation["stocks"][self.current_stock_index]) # Get the current stock
        w, h = self._get_stock_size_(stock)
        total_area = w * h

        # Calculate the area used by combination
        for combination in population:
            used_area = 0

            for action in combination:
                size = action["size"]
                used_area += size[0] * size[1]

            unused_area.append(total_area - used_area)
        return unused_area

    def get_best_mutation(self, population, remain_products_group, unused_area, observation):
        # Generate the next generations
        for _ in range(self.num_generations):
            # Generate the next half combinations by mutation
            for i in range(self.population_size // 2):
                current_stock = copy.deepcopy(observation["stocks"][self.current_stock_index])
                # Get the best combination
                current_index = i + self.population_size // 2

                # Delete a random action from the best combinations and replace it with a new actions until we try out all available products
                population[current_index] = copy.deepcopy(population[i])
                remain_products_group[current_index] = copy.deepcopy(remain_products_group[i])
                unused_area[current_index] = copy.deepcopy(unused_area[i])

                # Remove a random action from the current combination, when removed, update the remaining products and unused area
                if len(population[current_index]) > 0:
                    action_idx = np.random.randint(len(population[current_index]))
                    action = population[current_index][action_idx]

                    # Keep the size to find it in the remaining products to update the quantity
                    size = action["size"]

                    # Update the remaining products and delete the action from the combination
                    for prod_idx, prod in enumerate(remain_products_group[current_index]):
                        if prod["size"][0] == size[0] and prod["size"][1] == size[1]:
                            remain_products_group[current_index][prod_idx]["quantity"] += 1
                            unused_area[current_index] += size[0] * size[1]
                            del population[current_index][action_idx]
                            break
                    number_of_products = 0
                    for prod in remain_products_group[current_index]:
                        number_of_products += prod["quantity"]

                    # Fill the stock using bottom-left fill algorithm in random order until try out all available_products, the product is rotated if it can't fit in the stock
                    for _ in range(number_of_products):
                        # Get indices of products with quantity > 0
                        valid_indices = []
                        for i, prod in enumerate(remain_products_group[current_index]):
                            if prod["quantity"] > 0:
                                valid_indices.append(i)

                        if valid_indices:
                            # Get random index from valid products
                            prod_idx = valid_indices[np.random.randint(len(valid_indices))]
                            prod = remain_products_group[current_index][prod_idx]
                        else:
                            # No valid products left
                            break
                            
                        placed = False
                        stock_w, stock_h = self._get_stock_size_(current_stock)
                        original_size = prod["size"]
                        rotated_size = (original_size[1], original_size[0])

                        for orientation in [original_size, rotated_size]:
                            prod_w, prod_h = orientation
                            if stock_w < prod_w or stock_h < prod_h:
                                continue

                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(current_stock, (x, y), orientation):
                                        current_stock[x:x + prod_w, y:y + prod_h] = 1
                                        population[current_index].append({
                                            "stock_idx": self.current_stock_index,
                                            "size": orientation,
                                            "position": (x, y)
                                        })
                                        remain_products_group[current_index][prod_idx]["quantity"] -= 1
                                        unused_area[current_index] -= prod_w * prod_h
                                        placed = True
                                        break
                                if placed:
                                    break
                            if placed:
                                break

            # Sort the population and remain_products_group based on the unused_area in ascending order
            combined = list(zip(unused_area, range(len(unused_area))))
            order = [i for _, i in sorted(combined)]

            population = [population[i] for i in order]
            remain_products_group = [remain_products_group[i] for i in order]
            unused_area = [unused_area[i] for i in order]

        return population[0]
    
    def get_best_combination(self, observation):
        # Generate the initial population
        population = []
        remain_products_group = []
        for _ in range(self.population_size):
            result = self.generate_combination(observation)
            population.append(result["combination"])
            remain_products_group.append(result["remaining_products"])
        
        # Calculate the fitness of each combination (actually the unused area because the fitness is inversely proportional to the unused area)
        unused_area = self.get_unused_area(population, observation)

        # Sort the population and remain_products_group based on the unused_area in ascending order
        combined = list(zip(unused_area, range(len(unused_area))))
        order = [i for _, i in sorted(combined)]

        population = [population[i] for i in order]
        remain_products_group = [remain_products_group[i] for i in order]
        unused_area = [unused_area[i] for i in order]
        ###################################################################################################
        # To generate the next generation, we will use the following steps:
        # 1. Keep the best half combinations
        # 2. Generate the next half combinations by mutation
        # 3. Mutation is done by delete a random action from the best combination and replace it with a new actions until we try out all available products
        # 4. Repeat the process for the next generations
        best_combinations = self.get_best_mutation(population, remain_products_group, unused_area, observation)
        ###################################################################################################
        
        # Keep the best combinations
        best_combinations = population[0]
        
        # Insert the best combinations to the output queue
        for action in best_combinations:
            self.output_queue.append(action)
        return None

    def get_action(self, observation, info):
        # Check if the output queue is empty, then get the new combination for the next stock
        if not self.output_queue:
            self.current_stock_index += 1
            # If the position (0,0) of the first stock is -1 then it is empty, so we reset the current_stock_index to 0
            if observation["stocks"][0][0][0] == -1:
                self.current_stock_index = 0
            self.get_best_combination(observation)
        # Check if the output queue is not empty, then return the next action
        return self.output_queue.popleft()