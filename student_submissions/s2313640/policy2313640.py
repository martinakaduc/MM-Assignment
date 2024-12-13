from policy import Policy
import numpy as np
import random
import math

class Policy2313640:
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = SimulatedAnnealing()
        elif policy_id == 2:
            self.policy = ColumnGeneration()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
    
class ColumnGeneration(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        sorted_prods = sorted(
            list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True
        )

        columns = self._initialize_columns(observation)

        while True:
            lp_solution = self._solve_lp_relaxation(columns, observation)

            reduced_costs = self._calculate_reduced_costs(
                columns, lp_solution, observation
            )

            if all(cost >= 0 for cost in reduced_costs):
                break  

            min_cost_idx = np.argmin(reduced_costs)

            if min_cost_idx >= 0 and min_cost_idx < len(observation["products"]):
                columns.append(self._generate_new_column(min_cost_idx, observation))
            else:
                break  

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                best_fit_stock = None
                best_fit_position = None
                min_waste = float("inf")  

                stock_areas = []
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue  

                    free_area = stock_w * stock_h - np.sum(stock != -1)
                    stock_areas.append((i, free_area, stock))

                stock_areas.sort(key=lambda x: x[1], reverse=True)

                for i, free_area, stock in stock_areas:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    position = self._find_first_position(stock, prod_size)
                    if position:
                        x, y = position
                        remaining_space = free_area - (prod_w * prod_h)

                        if remaining_space < min_waste:
                            min_waste = remaining_space
                            best_fit_stock = i
                            best_fit_position = (x, y)

                if best_fit_stock is not None and best_fit_position is not None:
                    stock_idx = best_fit_stock
                    pos_x, pos_y = best_fit_position
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def _initialize_columns(self, observation):
        columns = []
        for prod in observation["products"]:
            if prod["quantity"] > 0:
                columns.append(self._generate_initial_column(prod))
        return columns

    def _generate_initial_column(self, prod):
        return {"product": prod, "size": prod["size"]}

    def _generate_new_column(self, min_cost_idx, observation):
        prod = observation["products"][min_cost_idx]
        return {"product": prod, "size": prod["size"]}

    def _solve_lp_relaxation(self, columns, observation):
        lp_solution = {"objective_value": 0, "column_values": np.zeros(len(columns))}
        return lp_solution

    def _calculate_reduced_costs(self, columns, lp_solution, observation):
        reduced_costs = []
        for col in columns:
            reduced_cost = self._compute_reduced_cost(col, lp_solution, observation)
            reduced_costs.append(reduced_cost)
        return reduced_costs

    def _compute_reduced_cost(self, column, lp_solution, observation):
        product = column["product"]
        prod_size = product["size"]

        total_stock_area_before = 0
        for stock in observation["stocks"]:
            stock_w, stock_h = self._get_stock_size_(stock)
            total_stock_area_before += stock_w * stock_h

        total_stock_area_after = 0
        for stock in observation["stocks"]:
            stock_w, stock_h = self._get_stock_size_(stock)
            total_stock_area_after += stock_w * stock_h

        reduced_cost = total_stock_area_after - total_stock_area_before

        return reduced_cost

    def _find_first_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None

class SimulatedAnnealing(Policy):
    def __init__(self, initial_temperature=1000, cooling_rate=0.95, iterations=100):
        super().__init__()
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.iterations = iterations

    def get_action(self, observation, info):
        initial_action = self.FFD_action(observation, info)

        if initial_action is None:
            return self.get_default_action()

        optimized_action = self.run_simulated_annealing(initial_action, observation)

        return optimized_action

    def run_simulated_annealing(self, initial_action, observation):
        current_action = initial_action.copy()
        current_cost = self.calculate_cost(current_action, observation)
        best_action = current_action.copy()
        best_cost = current_cost
        temperature = self.initial_temperature

        for _ in range(self.iterations):
            neighbor_action = self.generate_neighbor(current_action, observation)

            if neighbor_action is None:
                continue  

            neighbor_cost = self.calculate_cost(neighbor_action, observation)
            delta_cost = neighbor_cost - current_cost

            if delta_cost < 0 or random.random() < math.exp(-delta_cost / temperature):
                current_action = neighbor_action.copy()
                current_cost = neighbor_cost

                if current_cost < best_cost:
                    best_action = neighbor_action.copy()
                    best_cost = current_cost

            temperature *= self.cooling_rate

            if temperature < 1e-3:
                break

        return best_action
    
    def FFD_action(self, observation, info):
        list_prods = observation["products"]

        list_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                placed = False
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    stock_idx = i
                                    placed = True
                                    break
                            if placed:
                                break

                    if not placed and stock_w >= prod_h and stock_h >= prod_w:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):  
                                    prod_size = prod_size[::-1]  
                                    pos_x, pos_y = x, y
                                    stock_idx = i
                                    placed = True
                                    break
                            if placed:
                                break

                    if placed:
                        break
                if placed:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def generate_neighbor(self, action, observation):
        neighbor = action.copy()

        candidate_stock_idx = random.randint(0, len(observation["stocks"]) - 1)
        neighbor["stock_idx"] = candidate_stock_idx

        prod_w, prod_h = neighbor["size"]
        prod_size = (prod_w, prod_h)

        stock = observation["stocks"][candidate_stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)

        placed = False

        if stock_w >= prod_w and stock_h >= prod_h:
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), prod_size):
                        neighbor["position"] = (x, y)
                        placed = True
                        return neighbor

        if not placed and prod_w != prod_h and stock_w >= prod_h and stock_h >= prod_w:
            rotated_size = (prod_h, prod_w)
            for x in range(stock_w - prod_h + 1):
                for y in range(stock_h - prod_w + 1):
                    if self._can_place_(stock, (x, y), rotated_size):
                        neighbor["size"] = rotated_size
                        neighbor["position"] = (x, y)
                        placed = True
                        return neighbor

        return None

    def is_valid_action(self, action, observation):
        stock = observation["stocks"][action["stock_idx"]]
        stock_w, stock_h = self._get_stock_size_(stock)
        pos_x, pos_y = action["position"]
        prod_w, prod_h = action["size"]

        if pos_x + prod_w > stock_w or pos_y + prod_h > stock_h:
            return False
        return self._can_place_(stock, (pos_x, pos_y), action["size"])

    def calculate_cost(self, action, observation):
        stock = observation["stocks"][action["stock_idx"]].copy() 
        pos_x, pos_y = action["position"]
        prod_w, prod_h = action["size"]

        if pos_x + prod_w > stock.shape[1] or pos_y + prod_h > stock.shape[0]:
            return float('inf') 

        if not self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
            return float('inf')  
        product_idx = action.get("product_idx", 0)

        stock[pos_y: pos_y + prod_h, pos_x: pos_x + prod_w] = product_idx
        remaining_empty_area = np.sum(stock == -1)

        return remaining_empty_area

    def get_default_action(self):
        return {"stock_idx": 0, "size": [1, 1], "position": (0, 0)}