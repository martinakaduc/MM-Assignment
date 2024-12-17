from policy import Policy
import numpy as np
from scipy.optimize import linprog



class Policy2212940_2212497_2212272_2212288_2212982(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id
        if self.policy_id == 1:
            pass
        elif self.policy_id == 2:
            self.patterns = None # store cutting patterns
            self.demand = None # store demand of products
            self.optimal_flag = False # flag to check if the optimal solution is found
            self.optimal_solution = None # store the optimal solution
            self.list_prods = None # store the list of products
            self.list_stocks = None # store the list of stocks
            self.observation = None # store the observation

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self._BFD(observation, info)
        elif self.policy_id == 2:
            return self._step_column_generation(observation, info)
    # Student code here
    # You can add more functions if needed

#===================Policy 1===================#
    """
    Best Fit Decreasing (BFD) algorithm

    Args:
        observation (dict): The observation of the environment
        info (dict): Additional information
    
    Returns:
        dict: The action to take

    1. Sort the products by size in descending order
    2. For each product, find the best fit stock
    3. Return the action to place the product in the best fit stock
    
    """
    def _BFD(self, observation, info):
        # Sort the products by size in descending order
        list_prods = sorted(
            observation["products"],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )

        for idx, prod in enumerate(list_prods):
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                the_best_fit = {
                    "stock_idx": -1,
                    "size": prod_size,
                    "position": (-1, -1),
                    "waste": float("inf")
                }

                # Find the best fit stock
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    waste = self._get_stock_size_waste(stock, prod_w, prod_h)
                                    if waste < the_best_fit["waste"]:
                                        the_best_fit["waste"] = waste
                                        the_best_fit["stock_idx"] = i
                                        the_best_fit["position"] = (x, y)
                                        the_best_fit["size"] = prod_size


                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    waste = self._get_stock_size_waste(stock, prod_h, prod_w)
                                    if waste < the_best_fit["waste"]:
                                        the_best_fit["waste"] = waste
                                        the_best_fit["stock_idx"] = i
                                        the_best_fit["position"] = (x, y)
                                        the_best_fit["size"] = prod_size[::-1]


                if the_best_fit["stock_idx"] != -1:
                    return the_best_fit

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


    def _get_stock_size_waste(self, stock, prod_w, prod_h):
        # Calculate the remaining area in the stock after placing the product
        product_area = prod_w * prod_h
        stock_area_remaining = np.sum(stock == -1) - product_area
        return stock_area_remaining




#===================Policy 2===================#

    def _column_generation(self, observation, info):

        self.observation = observation

        self.list_prods = sorted(
            observation["products"],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )

        self.list_stocks = observation["stocks"]
        self.demand = [prod["quantity"] for prod in self.list_prods]

        # initial patterns
        self.patterns = self._generate_cutting_pattern()

        max_counter = 100
        counter =  0
        res = []
        while not self.optimal_flag and counter < max_counter:
            # Solve the master problem
            dual_value, res = self._master_problem_solver(self.list_prods, self.list_stocks)
            # Solve the subproblem
            have_new_pattern = self.subproblem_solver(dual_value)

            if not have_new_pattern:
                self.optimal_flag = True
            counter += 1

        # After solving the column generation problem, solve the integer problem
        self._interger_problem_solver(res)
        return self.optimal_solution



    def _interger_problem_solver(self, res):
        """
        Solve the integer problem
        """
        
        # Round the result to the nearest integer
        res = [round(i) for i in res]
        self.optimal_solution = [self.patterns[i] for i in range(len(res)) if res[i] > 0]

        temp = [] # store the number of products in each pattern
        for i in range(len(self.demand)):
            temp.append(sum(self.optimal_solution[j]["products"][i] for j in range(len(self.optimal_solution))))


        # Check if the optimus solution is feasible
        # Here we will check if the number of products in each pattern is enough to meet the demand
        # If it more than the demand, we will remove the excessive products
        for i, j in enumerate(self.demand):
            tmp = 0
            if j < temp[i]:
                tmp = temp[i] - j
                for idx, pattern in enumerate(self.optimal_solution):
                    if pattern["products"][i] > 0 and tmp > 0:
                        if tmp > pattern["products"][i]:
                            tmp -= pattern["products"][i]
                            pattern["waste"] += pattern["products"][i] * self.list_prods[i]["size"][0] * self.list_prods[i]["size"][1]
                            pattern["products"][i] = 0
                        else:
                            pattern["waste"] += tmp * self.list_prods[i]["size"][0] * self.list_prods[i]["size"][1]
                            pattern["products"][i] -= tmp
                            break
        

        # Here we will store the number of missing products
        lack = [] # variable to store the number of missing products
        for i, j in enumerate(self.demand):
            if j > temp[i]:
                lack.append((i, j - temp[i]))
        
        # If there are missing products, we will add them to the stock
        self._process_adding_pattern(lack)


    def _process_adding_pattern(self, lack):
        """
        Function to add the missing products to the stock
        
        Initial, We will implement process to add products to the stock
        After that, we will implement process to add the missing products to the stock use the best fit algorithm
        """
        list_stocks_copy = [stock.copy() for stock in self.list_stocks]
        list_optimal_solution = [pattern.copy() for pattern in self.optimal_solution]

        # add products to the stock
        for pattern in list_optimal_solution:
                stock = list_stocks_copy[pattern["stock_idx"]]
                stock_w, stock_h = self._get_stock_size_(stock)
                for prod_idx, quality in enumerate(pattern["products"]):
                    if quality > 0:
                        for i in range(quality):
                            prod_size = self.list_prods[prod_idx]["size"]
                            prod_w, prod_h = prod_size
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break

                            if pos_x is None and pos_y is None:
                                for x in range(stock_w - prod_h + 1):
                                    for y in range(stock_h - prod_w + 1):
                                        if self._can_place_(stock, (x, y), prod_size[::-1]):
                                            pos_x, pos_y = x, y
                                            prod_size = prod_size[::-1]
                                            break
                                    if pos_x is not None and pos_y is not None:
                                        break

                            if pos_x is not None and pos_y is not None:
                                self._cutting_stock(stock, prod_size, (pos_x, pos_y))
                                
                       
        # add missing products to the stock
        for i, j in lack:
            if j > 0:
                for z in range(j):
                    the_best_fit = {
                        "stock_idx": -1,
                        "size": [0, 0],
                        "position": (-1, -1),
                        "waste": float("inf")
                    }
                    # Find the best fit stock
                    for idx, stock in enumerate(list_stocks_copy):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        current_waste = self._calculate_waste(stock)
                        product_area = self.list_prods[i]["size"][0] * self.list_prods[i]["size"][1]
                        prod_size = self.list_prods[i]["size"]
                        prod_w, prod_h = prod_size
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                        if pos_x is None and pos_y is None:
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        pos_x, pos_y = x, y
                                        prod_size = prod_size[::-1]
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break

                        if pos_x is not None and pos_y is not None:
                            waste = current_waste - product_area
                            if waste < the_best_fit["waste"]:
                                the_best_fit["waste"] = waste
                                the_best_fit["stock_idx"] = idx
                                the_best_fit["position"] = (pos_x, pos_y)
                                the_best_fit["size"] = prod_size    

                    # Place the product in the best fit stock
                    # If the pattern is found, add the product to the pattern
                    # Else, create a new pattern
                    if the_best_fit["stock_idx"] != -1:
                        stock = list_stocks_copy[the_best_fit["stock_idx"]]
                        stock_w, stock_h = self._get_stock_size_(stock)
                        # Add the product to the stock
                        self._cutting_stock(stock, the_best_fit["size"], the_best_fit["position"])
                        # Calculate the waste of the stock
                        waste = self._calculate_waste(stock)
                        # Check if the pattern of the stock is already exist
                        existing_pattern = next((pattern for pattern in list_optimal_solution if pattern["stock_idx"] == the_best_fit["stock_idx"]), None)
                        # If the pattern is already exist, add the product to the pattern
                        if existing_pattern:
                            existing_pattern["products"][i] += 1
                            existing_pattern["waste"] = waste
                        else: # Else, create a new pattern
                            pattern = {
                                "stock_idx": the_best_fit["stock_idx"],
                                "products": [0] * len(self.list_prods),
                                "waste": waste
                            }
                            pattern["products"][i] += 1
                            list_optimal_solution.append(pattern)

        # Update the optimal solution
        self.optimal_solution = list_optimal_solution
                        

    def _master_problem_solver(self, list_prods, list_stocks):
        """
        Solve the master problem
        """

        # Define the objective function
        c = np.array([pattern['waste'] for pattern in self.patterns])

        # Define the constraints
        A = np.array([pattern['products'] for pattern in self.patterns]).T
        b = np.array(self.demand)
        A_eq = -A
        b_eq = -b

        # Define the bounds
        bounds = [(0, 1) for _ in range(len(self.patterns))]

        num_patterns = len(self.patterns)
        num_stocks = len(set(pattern['stock_idx'] for pattern in self.patterns))  # Unique stock count

        # Define the constraints
        # Initialization of A_ub and b_ub
        A_ub = np.zeros((num_stocks, num_patterns))
        b_ub = np.ones(num_stocks)
        
        # Each stock can be used at most once
        for i, pattern in enumerate(self.patterns):
            stock_idx = pattern['stock_idx']
            A_ub[stock_idx, i] = 1

        # Solve the linear programming problem
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        # Return the result
        if res.success:
            return res.slack, res.x
        else:
            print("No solution found")


    def subproblem_solver(self, dual_value):
        """
        Solve the subproblem

        We will placed the product in the stock
        If the waste of the stock is less than the sum of the dual value and the product of the quantity and dual value
        We will add the pattern to the optimal solution
        """
        list_prods_copy = [prod.copy() for prod in self.list_prods]
        list_stocks_copy = [stock.copy() for stock in self.list_stocks]
        
        best_pattern = None
        min_reduced_cost = float("inf")

        # For each stock, find the best pattern
        for idx, stock in enumerate(list_stocks_copy):
            stock_w, stock_h = self._get_stock_size_(stock)

            pattern = {
                "stock_idx": idx,
                "products": [0] * len(list_prods_copy),
                "waste": stock_w * stock_h
            }

            for prod_idx, prod in enumerate(list_prods_copy):
                prod_w, prod_h = prod["size"]
                quantity = prod["quantity"]
                if quantity > 0:
                    for i in range(quantity):
                        prod_size = prod["size"]
                        pos_x, pos_y = None, None
                        if prod_w > stock_w or prod_h > stock_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod["size"]):
                                        pos_x, pos_y = x, y
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break
                        if pos_x is None and pos_y is None:
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod["size"][::-1]):
                                        pos_x, pos_y = x, y
                                        prod_size= prod_size[::-1]
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break
                        if pos_x is not None and pos_y is not None:
                            pattern["products"][prod_idx] += 1
                            self._cutting_stock(stock, prod_size, (pos_x, pos_y))

            # Calculate the waste of the stock
            pattern["waste"] = self._calculate_waste(stock)
            reduced_cost = pattern["waste"] - sum(
                dual_value[i] * pattern["products"][i] for i in range(len(list_prods_copy))
            )

            # If the waste of the stock is less than the sum of the dual value and the product of the quantity and dual value
            if reduced_cost < min_reduced_cost:
                best_pattern = pattern
                min_reduced_cost = reduced_cost

        # If the best pattern is found, add the pattern to the optimal solution
        if best_pattern and min_reduced_cost < 0 and best_pattern not in self.patterns:
            print("best_pattern add: ",best_pattern)
            self.patterns.append(best_pattern)
            return True
        return False




    def _generate_cutting_pattern(self):
        """
        Function to generate initial cutting patterns

        We will use the BFD algorithm and FFD algorithm to generate the initial solution
        """
        patterns = [] # store the cutting patterns

        list_prods_copy = [prod.copy() for prod in self.list_prods]
        list_stocks_copy = [stock.copy() for stock in self.list_stocks]
        
        # Use the BFD algorithm to generate the initial solution
        initial_solutions = self._BFD_for_generatoin_collumn(list_prods_copy, list_stocks_copy)

        # Generate the cutting patterns
        for i, stock in enumerate(list_stocks_copy):
            pattern = {
                "stock_idx" : i,
                "products" : [0] * len(list_prods_copy),
                "waste" : 0
            }
            cutted_stock = 0
            for solution in initial_solutions:
                if solution["stock_idx"] == i:
                    pattern["products"][solution["product_idx"]] += 1
                    cutted_stock += 1

            if cutted_stock > 0:
                pattern["waste"] = self._calculate_waste(stock)

            patterns.append(pattern) # add the pattern to the list of patterns


        # use the FFD algorithm to generate the initial solution
        list_prods_copy2 = [prod.copy() for prod in self.list_prods]
        list_stocks_copy2 = [stock.copy() for stock in self.list_stocks]
        initial_solutions2 = self._FFD(list_prods_copy2, list_stocks_copy2)

        # Generate the cutting patterns
        for i, stock in enumerate(list_stocks_copy2):
            pattern = {
                "stock_idx" : i,
                "products" : [0] * len(list_prods_copy2),
                "waste" : 0
            }
            cutted_stock = 0
            for solution in initial_solutions2:
                if solution["stock_idx"] == i:
                    pattern["products"][solution["product_idx"]] += 1
                    cutted_stock += 1

            if cutted_stock > 0:
                pattern["waste"] = self._calculate_waste(stock)

            patterns.append(pattern) # add the pattern to the list of patterns

        return patterns




    def _FFD(self, list_prods_copy, list_stocks_copy):
        """
        First Fit Decreasing (FFD) algorithm
        """

        solution = [] # store the solution

        while True:
            placed = False
            for idx, prod in enumerate(list_prods_copy):
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    for i, stock in enumerate(list_stocks_copy):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size

                        if stock_w >= prod_w and stock_h >= prod_h:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        break

                                if pos_x is not None and pos_y is not None:
                                    break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break

                        if stock_w >= prod_h and stock_h >= prod_w:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        prod_size = prod_size[::-1]
                                        pos_x, pos_y = x, y
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break

                    if pos_x is not None and pos_y is not None:
                        solution.append({"stock_idx": stock_idx, "product_idx": idx, "position": (pos_x, pos_y)})
                        list_prods_copy[idx]["quantity"] -= 1
                        self._cutting_stock(stock, prod_size, (pos_x, pos_y))
                        placed = True
                        break

            # If no product can be placed or all products have been placed, return the solution
            if not placed or np.sum([prod["quantity"] for prod in list_prods_copy]) == 0:
                return solution

    def _BFD_for_generatoin_collumn(self, list_prods_copy, list_stocks_copy):
        """
        Best Fit Decreasing (BFD) algorithm
        """

        # Here we will use the BFD algorithm to generate the initial solution
        solution = []
        while True:
            placed = False
            for idx, prod in enumerate(list_prods_copy):
                if prod["quantity"] > 0:
                    prod_size = prod["size"]

                    the_best_fit = {
                        "stock_idx": -1,
                        "size": prod_size,
                        "position": (-1, -1),
                        "waste": float("inf")
                    }

                    # Find the best fit stock
                    for i, stock in enumerate(list_stocks_copy):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size

                        # Check if the stock can contain the product
                        if stock_w >= prod_w and stock_h >= prod_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        waste = self._get_stock_size_waste(stock, prod_w, prod_h)
                                        if waste < the_best_fit["waste"]:
                                            the_best_fit["waste"] = waste
                                            the_best_fit["stock_idx"] = i
                                            the_best_fit["position"] = (x, y)
                                            the_best_fit["size"] = prod_size

                        # Check if the stock can contain the product when rotated 90 degrees
                        if stock_w >= prod_h and stock_h >= prod_w:
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        waste = self._get_stock_size_waste(stock, prod_h, prod_w)
                                        if waste < the_best_fit["waste"]:
                                            the_best_fit["waste"] = waste
                                            the_best_fit["stock_idx"] = i
                                            the_best_fit["position"] = (x, y)
                                            the_best_fit["size"] = prod_size[::-1]

                    # Place the product in the best fit stock
                    if the_best_fit["stock_idx"] != -1:
                        solution.append({
                            "stock_idx": the_best_fit["stock_idx"],
                            "product_idx": idx,
                            "position": the_best_fit["position"]
                        })
                        list_prods_copy[idx]["quantity"] -= 1
                        self._cutting_stock(list_stocks_copy[the_best_fit["stock_idx"]], the_best_fit["size"], the_best_fit["position"])
                        placed = True
                        break

            # If no product can be placed or all products have been placed, return the solution
            if not placed or np.sum([prod["quantity"] for prod in list_prods_copy]) == 0:
                return solution


    def _cutting_stock(self, stock, prod_size, position):
        """
        Function implementing the cutting stock problem
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        x, y = position
        if (x >= 0 and y >= 0 and x + prod_w <= stock_w and y + prod_h <= stock_h):
            if np.all(stock[x : x + prod_w, y : y + prod_h] == -1):
                stock[x : x + prod_w, y : y + prod_h] = 1
                return True


    def _calculate_waste(self, stock):
        """
        Function to calculate the waste of the stock
        """
        return np.sum(stock == -1)


    def _step_column_generation(self, observation, info):
        """
        Function to generate the action and return the action to take for env

        Args:
            observation (dict): The observation of the environment
            info (dict): Additional information
        
        Returns:
            dict: The action to take

        1. Check if the optimal solution is not found, solve the column generation problem
        2. For each pattern in the optimal solution, find position to place the product
        3. Return the action to place the product in the stock
        """

        # Check if the optimal solution is not found, solve the column generation problem
        if self.optimal_solution is None:
            self._column_generation(observation, info)

        # For each pattern in the optimal solution, find position to place the product
        for pattern in self.optimal_solution:
            stock = observation["stocks"][pattern["stock_idx"]]
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod_idx, quality in enumerate(pattern["products"]):
                if quality > 0:
                    prod_size = self.list_prods[prod_idx]["size"]
                    prod_w, prod_h = prod_size
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break

                    if pos_x is None and pos_y is None:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    pos_x, pos_y = x, y
                                    prod_size = prod_size[::-1]
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                    # return the action to place the product in the stock
                    if pos_x is not None and pos_y is not None:
                        pattern["products"][prod_idx] -= 1
                        return {"stock_idx": pattern["stock_idx"], "size": prod_size, "position": (pos_x, pos_y)}


 
    

    
    