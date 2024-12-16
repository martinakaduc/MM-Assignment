from policy import Policy
import numpy as np
import random

class Policy2213273_2312469_2311744_2310707_2212941(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            # decision = 1 mean dynamic program
            self.decision = 1;
        elif policy_id == 2:
            # decision = 2 mean simulated annealing
            self.decision = 2;
        self.INT_MAX = 2 ** 31 - 1 # max value of integer
        self.count = 0 # index of solution
        self.stocks = [] # list of stocks
        self.products = [] # list of products
        self.stocks_size = [] # list of stocks's size
        self.solution = [] # result
        self.numProd = 0;    

    def get_action(self, observation, info):
        # Initialize for simulated annealing algorithm
        if self.count == 0:
            # Initializing the stocks 
            self.stocks = list(stock.copy() for stock in observation["stocks"])
            # Initializing the products
            self.products = list(product.copy() for product in observation["products"]) 
            # Initializing the stocks_size
            self.stocks_size = [self._get_stock_size_(stock) for stock in self.stocks] 
            for product in observation["products"]:
                self.numProd += product["quantity"]
        # Policy 1st
        if self.decision == 1:
            # Initialize required variables
            list_prods = observation["products"]
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0

            # Sort products by area (largest first)
            sorted_prods = sorted(
                [p for p in list_prods if p["quantity"] > 0],
                key=lambda x: x["size"][0] * x["size"][1],
                reverse=True
            )

            if not sorted_prods:
                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

            # Get the first available product
            first_prod = sorted_prods[0]
            prod_size = first_prod["size"]
            prod_w, prod_h = prod_size

            # Find the best placement using DP
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
            
                if stock_w < prod_w or stock_h < prod_h:
                    continue

                # Create DP table
                dp = np.zeros((stock_h + 1, stock_w + 1), dtype=float)
            
                # Fill DP table
                for y in range(stock_h - prod_h + 1):
                    for x in range(stock_w - prod_w + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            dp[y + prod_h][x + prod_w] = dp[y][x] + (prod_w * prod_h)

                # Find the best position
                if np.max(dp) > 0:
                    max_pos = np.unravel_index(np.argmax(dp), dp.shape)
                    pos_y, pos_x = max_pos[0] - prod_h, max_pos[1] - prod_w
                    stock_idx = i
                    break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        # Policy 2nd
        else:
            if self.count == 0:
                self.solution = self._simulated_annealing_() # Solve the problem with simulated annealing algorithm
        
            self.count += 1 # Increase the count variable after each function call
            #return self.solution
            # Return the step for each function call
            sol = self.solution[self.count - 1] 
            if self.count == self.numProd:
                self.count = 0;
                self.numProd = 0;
            return {"stock_idx": sol[0], "size": sol[1], "position": sol[2]}
    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    # Check if products can be cut from stock
    def _can__place_(self, stock_idx, prod_size, decision = True):
        stock_w, stock_h = self.stocks_size[stock_idx]
        prod_w, prod_h = prod_size
        range_width = stock_w - prod_w + 1
        range_height = stock_h - prod_h + 1
        if not decision:
            range_width, range_height = range_height, range_width
        for x in range(range_width):
            for y in range(range_height):
                if decision and np.all(self.stocks[stock_idx][x : x + prod_w, y : y + prod_h] == -1):
                    return x, y
                if not decision and np.all(self.stocks[stock_idx][y : y + prod_w, x : x + prod_h] == -1):
                    return y, x
        return -1, -1

    # Cutting the product from stock if it is possible, otherwise restoring the product
    def _place_product_(self, stock, position, prod_size, value = 0):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        # value == 0 means the product is cut, while a value == -1 means the product is restored
        stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] = value

    # Finding a suitable stock to cut the product
    def _helper_(self, product):
        visited = [False] * len(self.stocks) # Marking the stock as used
        count_stock  = len(self.stocks) 
        prod_w, prod_h = product 
        # Iterating through the number of stocks
        while count_stock >= 0:
            min_val = self.INT_MAX
            idx = -1
            # Iterating through all the stocks to find the one that meets the requirements
            for stock_idx in range(len(self.stocks)):
                stock_w, stock_h = self.stocks_size[stock_idx]
                # Skip the stock if product's size exceeds the its size, if it has been visited
                if prod_w > stock_w or prod_h > stock_h or visited[stock_idx]:
                    continue
                if min_val > min(stock_w - prod_w, stock_h - prod_h):
                    min_val = min(stock_w - prod_w, stock_h - prod_h)
                    idx = stock_idx
            x, y = self._can__place_(idx, product)
            if x != -1 and y != -1:
                return [idx, product, [x, y]] # Return the position of the product in stock
            else:
                visited[idx] = True # If the product cannot be placed, mark this stock as visited
            count_stock -= 1 # Increase count_stock

        return [-1, product, (-1, -1)]
    
    # Swap products in steps
    def _swap_product_(self, pattern1, pattern2):
        # Extract information from patterns
        stock_idx1, prod1, pos1 = pattern1
        stock_idx2, prod2, pos2 = pattern2
        # Restore the previous stocks 
        self._place_product_(self.stocks[stock_idx1], pos1, prod1, -1)
        self._place_product_(self.stocks[stock_idx2], pos2, prod2, -1)
        
        x, y = self._can__place_(stock_idx2, prod1)
        # If the product1 can be cut from stock2, replace the pattern1 
        if x != -1 and y != -1:
            pattern1 = [stock_idx2, prod1, [x, y]]
        # Otherwise, call the helper function to assist with cutting product1
        else:
            pattern1 = self._helper_(prod1)
        # Cutting product1 from new stock
        self._place_product_(self.stocks[pattern1[0]], pattern1[2], pattern1[1])

        x, y = self._can__place_(stock_idx1, prod2)
        # If the product2 can be cut from stock1, replace the pattern2
        if x != -1 and y != -1:
            pattern2 = [stock_idx1, prod2, [x, y]]
        # Otherwise, call the helper function to assist with cutting product2
        else:
            pattern2 = self._helper_(prod2)
        # Cutting product2 from new stock
        self._place_product_(self.stocks[pattern2[0]], pattern2[2], pattern2[1])

        # Return the new patterns
        return pattern1, pattern2

    # Cut product1 in pattern1 from new stock
    def _move_product_(self, pattern1, pattern2):
        # Extract information from patterns
        stock_idx1, prod1, pos1 = pattern1
        stock_idx2, prod2, pos2 = pattern2

        # Restore the stocks
        self._place_product_(self.stocks[stock_idx1], pos1, prod1, -1)

        x, y = self._can__place_(stock_idx2, prod1)
        # If the product1 can be cut from stock2, replace the pattern1
        if x != -1 and y != -1 and stock_idx1 !=stock_idx2:
            pattern1 = [stock_idx2, prod1, [x, y]]
        # Otherwise, call the helper function to assist with cutting product1
        else:
            pattern1 = self._helper_(prod1)
        # Cutting product1 from new stock
        self._place_product_(self.stocks[pattern1[0]], pattern1[2], pattern1[1])

        # Return new patterns
        return pattern1, pattern2

    # Initializing new neighbor for current solution
    def _choose_neighbor_(self, solution):
        neighbor = list(sol.copy() for sol in solution)

        # Random number of times to apply an operator
        num_operation = random.randint(1, 2)
        for _ in range(num_operation):
            # Random 2 step from current solution
            idx1, idx2 = random.randint(0, len(neighbor) - 1), random.randint(0, len(neighbor) - 1)
            # Select an operation to perform randomly
            operation = random.randint(1, 2)
            tmp1, tmp2 = [], []
            # If operation == 1, swap products in steps
            if operation == 1 and idx1 != idx2:
                tmp1, tmp2 = self._swap_product_(neighbor[idx1], neighbor[idx2])
            # Otherwise, move product1 to stock2
            else:
                tmp1, tmp2 = self._move_product_(neighbor[idx1], neighbor[idx2])
            for idx in range(len(neighbor[idx1])):
                neighbor[idx1][idx] = tmp1[idx]
                neighbor[idx2][idx] = tmp2[idx]
        
        # Return neighbor of solution
        return neighbor

    # Compute the fitness of solution
    def _compute_fitness_(self, solution):
        # Create a list to track whether each stock (material) is used or not
        is_used = [False] * len(self.stocks)
        # Variable to store the total wasted area of the stocks
        stocks_area_wasted = 0

        # Mark stocks as used based on the solution
        for sol in solution:
            is_used[sol[0]] = True
        # Calculate the total wasted area for all used stocks
        for idx in range(len(self.stocks)):
            stock_w, stock_h = self.stocks_size[idx]  # Get the size of the stock
            if is_used[idx]:  # Only consider used stocks
                stocks_area_wasted += np.sum(self.stocks[idx] == -1)  # Count wasted area (cells marked as -1)

        # Return the negative of the total wasted area (minimizing waste is the goal)
        while stocks_area_wasted > 100:
            stocks_area_wasted /= 10
        return -stocks_area_wasted

    # Initialize solution
    def _initialize_sol_(self, num_walk = 20, k = 3):
        # Initialize solution as a list
        sol = []
        # Loop through each product in the list of products
        for prod in self.products:
            stock_idx = -1
            prod_w, prod_h = prod["size"]
            stock_flag = [k] * len(self.stocks)
            # For each product quantity, try to cut them
            for _ in range(prod["quantity"]):
                placed = False # Flag to indicate if the product has been placed
                while not placed:
                    min_val = self.INT_MAX
                    # If stock_idx is not valid or the current stock cannot hold more products
                    if stock_idx == -1 or stock_flag[stock_idx] <= 0:
                        for idx in range(len(self.stocks)):
                            stock_w, stock_h = self.stocks_size[idx]
                            # Skip stock if it is too small or already has no space left
                            if stock_w < prod_w or stock_h < prod_h or stock_flag[idx] <= 0:
                                continue
                            # If the stock can fit the product, calculate the remaining space difference.
                            if min(stock_w - prod_w, stock_h - prod_h) < min_val:
                                min_val = min(stock_w - prod_w, stock_h - prod_h)
                                stock_idx = idx
                    # Check if the product can be cut in the chosen stock
                    x, y = self._can__place_(stock_idx, prod["size"])
                    if x == -1 or y == -1:
                        # If the product cannot be placed, mark the stock as unavailable and reset stock_idx
                        stock_flag[stock_idx] = 0
                        stock_idx = -1
                    else:
                        # Cut the product from the stock and update the solution list
                        self._place_product_(self.stocks[stock_idx], (x, y), prod["size"])
                        sol.append([stock_idx, prod["size"], [x, y]])
                        stock_flag[stock_idx] -= 1
                        placed = True
        # Perform a number of random walks to improve the solution (local search for better placements)
        for _ in range(num_walk):
            sol = self._choose_neighbor_(sol)
        # Return the first solution
        return sol

    # Simulated annealing algorithm
        # Simulated annealing algorithm
    def _simulated_annealing_(self, T=1000, t=1.0, rate=1.02, num_walk=20, k=5):
        # Initialize first solution
        solution = self._initialize_sol_(num_walk, k)
        # Loop until temperature T decreases to the minimum threshold t
        while T > t:
            # Create a copy of the current stocks to revert if the solution is not accepted
            dummy_stocks = list(stock.copy() for stock in self.stocks)
            # Choose a neighboring solution based on the current solution
            neighbor = self._choose_neighbor_(solution)
            # Compute the fitness score of the current solution and the neighboring solution
            curr_score = self._compute_fitness_(solution)
            neighbor_score = self._compute_fitness_(neighbor)
            # Calculate the acceptance probability
            delta = neighbor_score - curr_score
            acceptance_prob = np.exp(delta / T)
            # Determine whether to accept the neighbor solution
            if delta > 0 or random.uniform(0, 1) <= acceptance_prob:
                solution = neighbor
            else:
                # Revert the stocks to their previous state if neighbor is rejected
                self.stocks = list(stock.copy() for stock in dummy_stocks)
            # Decrease the temperature gradually
            T /= rate
        # Return the best solution found
        return solution

    def evaluate(self, stocks, is_used):
        area_wasted = 0.0
        total_area = 0.0
        for i in range(len(stocks)):
            if is_used[i]:
                stock_w, stock_h = self.stocks_size[i]
                total_area += stock_w * stock_h
                area_wasted += np.sum(stocks[i] == -1)

        return area_wasted, area_wasted / total_area