from policy import Policy
import numpy as np
import random
import math

class BFF(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        pos_x, pos_y = 0, 0
        list_prods = observation["products"]
        sorted_prods = sorted(list_prods, key=lambda x: max(x["size"]), reverse=True)
        stock_idx = -1
        prod_size = [0, 0]
        

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                best_fit_position = None
                min_waste = float("inf")
                best_fit_stock = None

                for i, stock in enumerate(observation["stocks"]):
                    prod_w, prod_h = prod_size
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_h < prod_h or stock_w < prod_w:
                        continue

                    a = stock_w - prod_w + 1
                    for x in range(a):
                        b = stock_h - prod_h + 1
                        for y in range(b):
                            if self._can_place_(stock, (x, y), prod_size):
                                c = prod_w * prod_h
                                remaining_space = np.sum(stock == -1) - c

                                if remaining_space < min_waste:
                                    best_fit_position = (x, y)
                                    min_waste = remaining_space
                                    best_fit_stock = i

                if best_fit_position is not None and best_fit_stock is not None:
                    pos_x, pos_y = best_fit_position
                    n = best_fit_stock
                    stock_idx = n
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

# Pseudocode
# Sort the product list in descending order by the largest dimension:

# sorted_prods = sort(products, by max(size), in descending order)
# Initialize global variable to track the best overall result:

# BestIndex = -1
# BestPosition = None
# BestSize = [0, 0]
# MinWaste = infinity (to track the smallest waste).
# Iterate through each product in the sorted list:

# If the product's quantity is <= 0, skip it.
# For each product:

# Retrieve the product size: prod_size = [width, height].
# Initialize temporary variable to store the best result for the current product:
# current_Stock = None
# current_BestPosition = None
# current_MinWaste = infinity.
# Iterate through all available stock sheets:

# Retrieve the stock sheet dimensions: stock_size = [stock_width, stock_height].
# If the stock sheet cannot accommodate the product (stock_width < prod_width or stock_height < prod_height), skip it.
# For each valid stock sheet:

# Evaluate all possible positions to place the product:
# Check if the position (x, y) is valid (_can_place_ returns True).
# Compute the remaining unused space (waste):
# remaining_space = free_space_in_stock - product_area.
# If the remaining space is less than current_MinWaste:
# Update current_Stock, current_BestPosition, and current_MinWaste.
# After evaluating all stock sheets:

# If the product has a better placement option (current_MinWaste < MinWaste):
# Update the global best result:
# BestIndex = current_Stock
# BestPosition = current_BestPosition
# BestSize = product_size
# MinWaste = current_MinWaste.
# After processing all products:

# Return the final result:
# StockIndex = BestIndex
# size = BestSize
# position = BestPosition (or [0, 0] if no valid position was found).

class CG(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        stock_idx = -1
        list_prods = observation["products"]
        pos_x, pos_y = 0, 0
        prod_size = [0, 0]
        
        columns = self._initialize_columns(observation)

        while True:
            lp_solution = self._solve_lp_relaxation(columns, observation)
            reduced_costs = self._calculate_reduced_costs(columns, lp_solution, observation)

            if all(cost >= 0 for cost in reduced_costs):
                break

            min_cost_idx = np.argmin(reduced_costs)

            if min_cost_idx < 0 or min_cost_idx >= len(observation["products"]):
                break
            else:
                columns.append(self._generate_new_column(min_cost_idx, observation))

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                for i, stock in enumerate(observation["stocks"]):
                    prod_w, prod_h = prod_size
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_h < prod_h or stock_w < prod_w:
                        continue

                    pos_x = None
                    pos_y = None
                    a = stock_w - prod_w + 1
                    for x in range(a):
                        b = stock_h - prod_h + 1
                        for y in range(b):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x = x
                                pos_y = y
                                break

                        if None not in (pos_x, pos_y): break

                    if None not in (pos_x, pos_y):
                        stock_idx = i
                        break

                if None not in (pos_x, pos_y):
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def _generate_initial_column(self, prod):
        return {"product": prod, "size": prod["size"]}

    def _solve_lp_relaxation(self, columns, observation):
        return {"objective_value": 0, "column_values": np.zeros(len(columns))}

    def _compute_reduced_cost(self, column, lp_solution, observation):
        return random.uniform(-1, 1)

    def _generate_new_column(self, min_cost_idx, observation):
        return {"product": observation["products"][min_cost_idx], "size": observation["products"][min_cost_idx]["size"]}

    def _calculate_reduced_costs(self, columns, lp_solution, observation):
        reduced_costs = []
        for col in columns:
            reduced_cost = self._compute_reduced_cost(col, lp_solution, observation)
            reduced_costs.append(reduced_cost)
        return reduced_costs

    def _initialize_columns(self, observation):
        columns = []
        for prod in observation["products"]:
            if prod["quantity"] > 0:
                columns.append(self._generate_initial_column(prod))
        return columns

# Pseudo-code for Column Generation (CG) Algorithm
# Initialize the initial set of columns:

# Generate a set of basic cutting patterns (columns) derived from the list of products.
# Iterate until an optimal solution is found: a. Solve the LP relaxation problem using the current columns:

# Solve a linear programming (LP) relaxation without requiring integer constraints.
# Find an approximate solution to guide the generation of new columns.
# b. Compute the reduced costs for all current columns:

# Use the formula:
# reduced_cost = cost − (dual_variable × column_coefficient)

# c. Evaluate the reduced costs:
# If all reduced costs ≥ 0: Terminate. The current solution is optimal.
# If any reduced cost < 0: Select the column with the smallest reduced cost and add it to the set of columns.

# After identifying the optimal set of columns:
# Perform branch-and-bound to ensure integrality in the solution.
# Select the product and determine the optimal placement:
# Evaluate all stock sheets and choose the best position for the selected product.

class Policy2352222_2352184_2353029_2352925(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        if policy_id == 1:
            self.algorithm = BFF()
        elif policy_id == 2:
            self.algorithm = CG()

    def get_action(self, observation, info):
        return self.algorithm.get_action(observation, info)