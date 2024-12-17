from policy import Policy
import numpy as np
import random

class Policy2210737_2311075_2213857_2212594_2211122(Policy):
    def __init__(self, policy_id=1):
        """
        Initializes the custom policy with a specific ID.
        """
        assert policy_id in [1, 2, 3], "Policy ID must be 1 or 2 or 3"
        self.policy_id = policy_id  # Store the policy ID for later use

    def get_action(self, observation, info):
        """
        Executes the policy based on the given policy_id.
        """
        if self.policy_id == 1:
            return self._genetic_algorithm(observation, info)
        elif self.policy_id == 2:
            return self._greedy_policy(observation, info)
        elif self.policy_id == 3:
            return self._brute_force_policy(observation, info)

    def _genetic_algorithm(self, observation, info):
      
        
        """
        Implements a Genetic Algorithm (GA) based policy to minimize trim loss.
        """
        population_size = 100  # Increased population size
        generations = 100  # Increased number of generations
        mutation_rate = 0.1  # Adjusted mutation rate

        stocks = observation["stocks"]
        products = sorted(
            observation["products"],
            key=lambda p: (p["size"][0] * p["size"][1], p["quantity"]),
            reverse=True
        )

        # Initial population: Random solutions
        population = [self._generate_random_solution(products, stocks) for _ in range(population_size)]

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = [self._evaluate_fitness(solution, stocks) for solution in population]

            # Selection: Select the top solutions
            selected_indices = np.argsort(fitness_scores)[:population_size // 2]
            selected_population = [population[i] for i in selected_indices]

            # Crossover: Create offspring by combining parents
            offspring = []
            while len(offspring) < population_size - len(selected_population):
                parent1, parent2 = random.sample(selected_population, 2)
                child = self._crossover(parent1, parent2)
                offspring.append(child)

            # Mutation: Randomly mutate offspring
            for child in offspring:
                if random.random() < mutation_rate:
                    self._mutate(child, products, stocks)

            # Update population
            population = selected_population + offspring

        # Choose the best solution from the final generation
        best_solution = min(population, key=lambda sol: self._evaluate_fitness(sol, stocks))

        # Return the first valid action from the best solution
        for action in best_solution:
            if self._is_action_valid(action, products, stocks):
                return action

        # Fallback: return a random action if no valid solution is found
        return self._generate_random_solution(products, stocks)[0]

    def _generate_random_solution(self, products, stocks):
    
       solution = []  # Đảm bảo thụt lề đúng ở đây
       for product in products:
          if product["quantity"] > 0:
              prod_size = product["size"]

            # Tìm vị trí phù hợp
              pos_x, pos_y, stock_idx = None, None, None
              for _ in range(10):  # Thử tối đa 10 lần
                # Random chọn một kho
                stock_idx = random.randint(0, len(stocks) - 1)
                stock = stocks[stock_idx]

                # Lấy kích thước của kho
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size

                # Kiểm tra điều kiện đặt ngang
                if stock_w >= prod_w and stock_h >= prod_h:
                    pos_x = random.randint(0, stock_w - prod_w)
                    pos_y = random.randint(0, stock_h - prod_h)
                    if self._can_place_(stock, (pos_x, pos_y), prod_size):
                        break

                # Kiểm tra điều kiện đặt dọc
                if stock_w >= prod_h and stock_h >= prod_w:
                    pos_x = random.randint(0, stock_w - prod_h)
                    pos_y = random.randint(0, stock_h - prod_w)
                    if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                        prod_size = prod_size[::-1]  # Đảo chiều kích thước
                        break

            # Nếu tìm được vị trí phù hợp, thêm vào giải pháp
              if pos_x is not None and pos_y is not None and stock_idx is not None:
                solution.append({
                    "stock_idx": stock_idx,
                    "size": prod_size,
                    "position": (pos_x, pos_y),
                })

       return solution



    def _evaluate_fitness(self, solution, stocks):
        """Evaluate the fitness of a solution based on trim loss."""
        total_trim_loss = 0
        used_stocks = set()
        used_area_by_stock = {}
        for action in solution:
            stock_idx = action["stock_idx"]
            size = action["size"]
            position = action["position"]

            stock = stocks[stock_idx]
            pos_x, pos_y = position
            prod_w, prod_h = size

            if self._can_place_(stock, position, size):
                stock_area = np.sum(stock != -2)
                used_area = prod_w * prod_h
                trim_loss = max((stock_area - used_area) / stock_area, 0)  # Ensure non-negative
                total_trim_loss += trim_loss
                used_stocks.add(stock_idx)
                used_area_by_stock[stock_idx] = used_area_by_stock.get(stock_idx, 0) + used_area

        # Penalize excessive trim loss and reward better utilization
        penalty = len(used_stocks) * 0.5
        utilization_score = sum(
            (used_area_by_stock[s] / np.sum(stocks[s] != -2)) for s in used_stocks
        )
        return total_trim_loss + penalty - utilization_score

    def _crossover(self, parent1, parent2):
        """Perform crossover between two parents."""
        crossover_point = random.randint(0, len(parent1) - 1)

        def action_to_hashable(action):
            """Convert action dictionary to a hashable representation."""
            return (
                action["stock_idx"],
                tuple(action["size"]),  # Convert numpy array to tuple
                tuple(action["position"])  # Convert numpy array to tuple
            )

        # Convert actions in parent1[:crossover_point] to hashable format
        parent1_actions = set(action_to_hashable(a) for a in parent1[:crossover_point])

        # Build the child
        child = parent1[:crossover_point] + [
            action for action in parent2 if action_to_hashable(action) not in parent1_actions
        ]
        return child


    def _mutate(self, solution, products, stocks):
        """Đột biến giải pháp mà không sử dụng random."""
    # Tìm hành động gây lãng phí nhiều nhất
        max_trim_loss_idx = None
        max_trim_loss = -1

        for idx, action in enumerate(solution):
            stock_idx = action["stock_idx"]
            size = action["size"]
            position = action["position"]
            stock = stocks[stock_idx]

            if self._can_place_(stock, position, size):
                stock_area = np.sum(stock != -2)
                used_area = size[0] * size[1]
                trim_loss = max((stock_area - used_area) / stock_area, 0)

                if trim_loss > max_trim_loss:
                    max_trim_loss = trim_loss
                    max_trim_loss_idx = idx

        # Nếu không tìm được hành động nào để cải thiện, kết thúc
        if max_trim_loss_idx is None:
            return

        # Tìm sản phẩm chưa được sử dụng và cố gắng đặt nó vào tốt hơn
        for product in products:
            if product["quantity"] > 0:
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Tìm vị trí phù hợp nhất
                    best_position = None
                    min_waste = float("inf")

                    for pos_x in range(stock_w - product["size"][0] + 1):
                        for pos_y in range(stock_h - product["size"][1] + 1):
                            if self._can_place_(stock, (pos_x, pos_y), product["size"]):
                                # Tính toán lãng phí
                                waste = (stock_w * stock_h) - (product["size"][0] * product["size"][1])
                                if waste < min_waste:
                                    min_waste = waste
                                    best_position = (pos_x, pos_y)

                    # Nếu tìm được vị trí tốt hơn, thực hiện đột biến
                    if best_position:
                        solution[max_trim_loss_idx] = {
                            "stock_idx": stock_idx,
                            "size": product["size"],
                            "position": best_position,
                        }
                        return


    def _is_action_valid(self, action, products, stocks):
        """Check if an action is valid."""
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        stock = stocks[stock_idx]
        product_exists = any(
            np.array_equal(size, product["size"]) and product["quantity"] > 0
            for product in products
        )
        return product_exists and self._can_place_(stock, position, size)

    def _greedy_policy(self, observation, info):
        """
        Implements an optimized greedy policy to minimize trim loss.
        """
        list_prods = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True,
        )

        for prod in list_prods:
            if prod["quantity"] > 0:  # Only consider products with positive quantity
                prod_size = prod["size"]

                # Try to place the product in the best-fit stock
                best_placement = None
                min_waste = float("inf")  # Initialize the minimum waste

                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for orientation in [(prod_size[0], prod_size[1]), (prod_size[1], prod_size[0])]:
                        prod_w, prod_h = orientation
                        if stock_w >= prod_w and stock_h >= prod_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                        # Calculate the waste if placed here
                                        remaining_w = stock_w - (x + prod_w)
                                        remaining_h = stock_h - (y + prod_h)
                                        waste = remaining_w * remaining_h

                                        # Update the best placement if this is more optimal
                                        if waste < min_waste:
                                            min_waste = waste
                                            best_placement = {
                                                "stock_idx": stock_idx,
                                                "size": [prod_w, prod_h],
                                                "position": (x, y),
                                            }

                if best_placement:
                    return best_placement

        # If no valid placement is found, return an invalid action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _brute_force_policy(self, observation, info):
        """
        Implements a brute force policy to minimize trim loss.
        """
        best_placement = None
        min_trim_loss = float("inf")  # Minimum trim loss found

        for prod in observation["products"]:
            if prod["quantity"] > 0:  # Only consider products with positive quantity
                prod_size = prod["size"]

                # Evaluate all possible placements across all stocks
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Try both orientations of the product
                    for orientation in [(prod_size[0], prod_size[1]), (prod_size[1], prod_size[0])]:
                        prod_w, prod_h = orientation
                        if stock_w >= prod_w and stock_h >= prod_h:
                            # Check every valid position on the stock
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                        trim_loss = self._evaluate_fitness(
                                            [{"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": (x, y)}],
                                            observation["stocks"]
                                        )
                                        if trim_loss < min_trim_loss:
                                            min_trim_loss = trim_loss
                                            best_placement = {
                                                "stock_idx": stock_idx,
                                                "size": [prod_w, prod_h],
                                                "position": (x, y),
                                            }
                           

        return best_placement if best_placement else {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

