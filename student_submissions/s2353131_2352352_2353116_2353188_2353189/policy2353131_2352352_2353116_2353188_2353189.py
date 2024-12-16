from policy import Policy
import numpy as np

class Policy2353131_2352352_2353116_2353188_2353189(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id
        if policy_id == 1:
            # Branch and Bound
            self.initialize = True
            self.min_wasted_space = float("inf")
            self.best_solution = []
            self.current_solution = []
            self.cnt = 0
            self.stock_idx = 0
            self.flag = True
            self.sorted_stocks = []
            self.stock_indices = []
            self.stock_counter = 0
            self.first_get_action = True
            self.indexed_stocks = []
        elif policy_id == 2:
            # Best Fit
            self.prod_area = 0

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            # Branch and Bound
            list_prods = observation["products"]
            list_stocks = observation["stocks"]
        
            if self.first_get_action:
                self.first_get_action = False
                self.indexed_stocks = [(i, stock) for i, stock in enumerate(list_stocks)]
                self.indexed_stocks = sorted(
                    self.indexed_stocks,
                    key=lambda s: np.sum(s[1] != -2),
                    reverse=True
                )
                self.sorted_stocks = [item[1] for item in self.indexed_stocks]
                self.stock_indices = [item[0] for item in self.indexed_stocks]

            if (self.stock_counter == len(self.sorted_stocks)):
                self.stock_counter = 0
            stock = self.sorted_stocks[self.stock_counter]
            stock_idx = self.stock_indices[self.stock_counter]
            list_prods = list(list_prods)
            list_prods.sort(key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)
            if (self.initialize):
                self.flag = True
                self.branch_and_bound(list_prods, stock, 0, stock_idx)
            
            self.initialize = False
            self.current_solution = []
            if (len(self.best_solution) == 0):
                temp = {"stock_idx": stock_idx, "size": (0, 0), "position": (0, 0)}
                stock_idx += 1
                self.cnt = 0
                self.min_wasted_space = float("inf")
                self.initialize = True
                self.stock_counter += 1
                return temp
            if self.cnt < len(self.best_solution):
                temp = self.best_solution[self.cnt]
                self.cnt += 1
                if self.cnt == len(self.best_solution):
                    self.cnt = 0
                    self.best_solution = []
                    self.stock_counter += 1
                    self.min_wasted_space = float("inf")
                    self.initialize = True
                return temp
            
        elif self.policy_id == 2:
            # Best Fit
            list_prods = observation["products"]
            used_stocks = []
            optimal_waste = float("inf")
            sorted_prods = sorted(
                list_prods,
                key=lambda prod: prod["size"][0] * prod["size"][1],
                reverse=True
            )

            prod_size = [0, 0]
            stock_idx = -1
            x, y = 0, 0

            for prod in sorted_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size
                        if stock_w >= prod_w and stock_h >= prod_h:
                            stock_idx, x, y, optimal_waste = self.find_stock(i, stock, prod_size, optimal_waste, 0) 
                            if x is not None and y is not None:
                                break 

                        if stock_w >= prod_h and stock_h >= prod_w:
                            stock_idx, x, y, optimal_waste = self.find_stock(i, stock, prod_size, optimal_waste, 1) 
                            if x is not None and y is not None:
                                prod_size = prod_size[::-1]
                                break

                        if x is not None and y is not None:
                            if stock_idx not in used_stocks:
                                used_stocks.append(stock_idx)

                    if x is not None and y is not None:
                        self.prod_area += prod_size[0] * prod_size[1]
                        break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}

    # Student code here
    # You can add more functions if needed
       
    # Helper functions for Branch and Bound
    def branch_and_bound(self, list_prods, stock, idx, stock_idx):
        if not self.flag:
            wasted_space = self.calculate_wasted_space(stock)
            if wasted_space < self.min_wasted_space:
                self.min_wasted_space = wasted_space
                self.best_solution = self.current_solution[:]
            return True
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_sizes = [prod["size"], prod["size"][::-1]]
                stock_w, stock_h = self._get_stock_size_(stock)
                
                for prod_size in prod_sizes:
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    
                    for i in range(stock_w - prod_w + 1):
                        for j in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (i, j), prod_size):
                                self.place_product(stock, (i, j), prod_size)
                                prod["quantity"] -= 1
                                self.current_solution.append({"stock_idx": stock_idx, "size": prod_size, "position": (i, j)})
                                if self.branch_and_bound(list_prods, stock, idx + 1, stock_idx):
                                    self.remove_product(stock, (i, j), prod_size)
                                    prod["quantity"] += 1
                                    return True
                                wasted_space = self.calculate_wasted_space(stock)
                                if wasted_space < self.min_wasted_space:
                                    self.min_wasted_space = wasted_space
                                    self.best_solution = self.current_solution[:]
                                self.current_solution.pop()
                                self.remove_product(stock, (i, j), prod_size)
                                prod["quantity"] += 1
                                self.flag = False
        return False
        
    def calculate_wasted_space(self, stock):
        return np.sum(stock == -1)
        
    def place_product(self, stock, position, prod_size):
        x, y = position
        w, h = prod_size
        stock[x : x + w, y : y + h] = 1
            
    def remove_product(self, stock, position, prod_size):
        x, y = position
        w, h = prod_size
        stock[x : x + w, y : y + h] = -1

    #Helper function for Best Fit
    def find_stock(self, idx, stock, prod_size, optimal_waste, rotated):
        x, y = None, None
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for i in range(stock_w - (prod_h if rotated else prod_w) + 1):
            for j in range(stock_h - (prod_w if rotated else prod_h) + 1):
                if self._can_place_(stock, (i, j), (prod_size[::-1] if rotated else prod_size)):
                    waste = ((stock_w - (i + prod_w)) * stock_h + (stock_h - (j + prod_h)) * stock_w)
                    if waste < optimal_waste:
                        x, y = i, j
                        optimal_waste = waste
                        if rotated: prod_size = prod_size[::-1]
                        break

            if x is not None and y is not None:
                return idx, x, y, optimal_waste

        return idx, x, y, optimal_waste
            
            
        
