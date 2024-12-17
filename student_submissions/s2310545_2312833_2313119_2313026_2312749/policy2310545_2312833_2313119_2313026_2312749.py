from policy import Policy
import numpy as np
from scipy.optimize import linprog


class Policy2310545_2312833_2313119_2313026_2312749(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.patterns = []
        self.dual_vals = None
        self.id = policy_id
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        if self.id == 1:
#   Algorithm: Best Fit Placement
#   Iterates through the available stocks to place products efficiently.
#   Products are selected one by one, and their possible placements are evaluated.
#
#   For each product, the algorithm checks all possible positions in the stock.
#   The placement is chosen to minimize the remaining unused space.
#
#   Criteria for placement:
#   - The product must fit within the stock dimensions.
#   - The placement is selected based on the smallest remaining space.
#
#   If a suitable placement is found, the product is placed in the current stock.
#   The process terminates as soon as a valid placement is determined.
            list_prods = observation["products"]
            stock_idx = -1
            prod_size = [0, 0]
            pos_x, pos_y = 0, 0
            
            for i,stock in enumerate(observation["stocks"]):
                flag = False
                stock_w, stock_h = self._get_stock_size_(stock)
                best_fit_stock_idx = -1
                best_fit_remaining_space = float('inf')  # Initialize with a large number
                best_fit_pos_x, best_fit_pos_y = None, None
                best_prod_size = prod_size
                for prod in list_prods:
                    
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        prod_w, prod_h = prod_size
                        if stock_w < prod_w or stock_h < prod_h:
                            continue
                        for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        remaining_space = (stock_w - (x + prod_w)) * (stock_h - (y + prod_h))
                                    
                                        # Check if this position leaves the least remaining space
                                        if remaining_space < best_fit_remaining_space:
                                            flag = True
                                            best_fit_remaining_space = remaining_space
                                            best_fit_stock_idx = i
                                            best_fit_pos_x, best_fit_pos_y = x, y
                                            best_prod_size = prod_size
                if (flag):
                    stock_idx = best_fit_stock_idx
                    pos_x, pos_y = best_fit_pos_x, best_fit_pos_y
                    prod_size = best_prod_size
                    break
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        elif self.id == 2:
# Algorithm: Cutting Stock with Dynamic Pattern Generation
# Iteratively generates and evaluates patterns to efficiently place products onto stocks.
# 
# Solves a linear program to minimize the number of stock pieces used.
# Patterns are updated dynamically based on the current solution and problem constraints.
# 
# Uses a 2D knapsack approach to generate new patterns.
# Evaluates dual values to create patterns that maximize placement value.
# 
# Criteria for Placement:
# - Products must fit within the stock dimensions.
# - Rotated placements are considered if they enable a fit.
# - The placement is selected based on the dual values and feasible patterns.
# 
# If a valid placement is found, the product is placed.
# The algorithm repeats until no further improvement is possible.
            list_prods = observation["products"]
            stocks = observation["stocks"]

            # Check for changes in the problem instance
            if len(self.patterns) != len(list_prods):
                self.patterns = []  # Reset patterns
                self._initialize_pattern(list_prods)

            if not self.patterns:
                self._initialize_pattern(list_prods)

            solution, dual_vals = self._solve_master_problem(list_prods)
            new_pattern, reduced_cost = self._solve_subproblem(list_prods, dual_vals)

            if reduced_cost >= -1e-7:
                action = self._select_action_from_solution(solution, list_prods, stocks)
            else:
                self.patterns.append(new_pattern)
                action = self.get_action(observation, info)

            return action
        pass

    # Student code here
    # You can add more functions if needed
    def _initialize_pattern(self, list_prods):
        for i in range(len(list_prods)):
            pattern = [0] * len(list_prods)
            pattern[i] = 1
            self.patterns.append(pattern)

    def _solve_master_problem(self, list_prods):
        num_patterns = len(self.patterns)
        A = np.array(self.patterns).T
        b = np.array([prod["quantity"] for prod in list_prods])
        c = np.ones(num_patterns)  # Minimize number of stock pieces used

        # Adjust b to match the number of rows in A
        if A.shape[0] > b.shape[0]:
            A = A[:b.shape[0], :]
        elif A.shape[0] < b.shape[0]:
            b = b[:A.shape[0]]

        res = linprog(c, A_eq=A, b_eq=b, method='highs')

        solution = res.x
        dual_vals = res.con

        return solution, dual_vals

    def _solve_subproblem(self, list_prods, dual_vals):
        # 2d knapsack
        n = len(list_prods)
        
        max_width = max(p["size"][0] for p in list_prods)
        max_height = max(p["size"][1] for p in list_prods)

        # Initialize a DP table (max_width + 1) x (max_height + 1)
        dp = [[0] * (max_height + 1) for _ in range(max_width + 1)]

        for i in range(n):
            prod_size = list_prods[i]["size"]
            prod_width, prod_height = prod_size
            value = dual_vals[i] * (prod_width * prod_height)

            # Update the DP table
            for w in range(max_width, prod_width - 1, -1):
                for h in range(max_height, prod_height - 1, -1):
                    dp[w][h] = max(dp[w][h], dp[w - prod_width][h - prod_height] + value)

        # Find the best value and corresponding pattern
        max_value = 0
        best_pattern = [0] * n
        for i in range(n):
            prod_size = list_prods[i]["size"]
            prod_width, prod_height = prod_size
            if dp[max_width][max_height] == dp[max_width - prod_width][max_height - prod_height] + dual_vals[i] * (prod_width * prod_height):
                best_pattern[i] = 1
                max_value += dual_vals[i] * (prod_width * prod_height)

        reduced_cost = 1 - max_value
        return best_pattern, reduced_cost




    def _select_action_from_solution(self, solution, list_prods, stocks):
        for i, stock in enumerate(stocks):
            for j in range(min(len(solution), len(list_prods))):  # Ensure index is within bounds
                if solution[j] > 0:
                    prod = list_prods[j]
                    prod_size = prod["size"]
                    for x in range(stock.shape[0] - prod_size[0] + 1):
                        for y in range(stock.shape[1] - prod_size[1] + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {"stock_idx": i, "size": prod_size, "position": (x, y)}
                            # Rotate piece
                            if self._can_place_(stock, (x, y), [prod_size[1], prod_size[0]]):
                                return {"stock_idx": i, "size": [prod_size[1], prod_size[0]], "position": (x, y)}
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}