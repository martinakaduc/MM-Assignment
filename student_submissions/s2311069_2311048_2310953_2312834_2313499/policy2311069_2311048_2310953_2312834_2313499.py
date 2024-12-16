from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2311069_2311048_2310953_2312834_2313499(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = bestfit()
        elif policy_id == 2:
            self.policy = column()
    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

class bestfit(Policy):
    def __init__(self):
        self.sorted_prods = None
        self.sorted_stocks = None
        self.stock_area_used = None
        self.can_place_prod = None

    def get_action(self, observation, info):
        if info["filled_ratio"] == 0.0:
            prod_average_area = sum(product["size"][0]*product["size"][1] for product in observation["products"]) / len(observation["products"])
            prod_average_quantity = sum(product["quantity"] for product in observation["products"]) / len(observation["products"])
            self.sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1], reverse=True)
            self.sorted_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            stock_average_area = sum(self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for _, stock in self.sorted_stocks) / len(self.sorted_stocks)
            if prod_average_area * prod_average_quantity < stock_average_area:
                self.sorted_stocks = self.sorted_stocks[::-1]
            self.stock_area_used = np.zeros(len(self.sorted_stocks))
            self.can_place_prod = np.ones((len(self.sorted_stocks), len(self.sorted_prods)), dtype=bool)
        for prod_idx,prod in enumerate(self.sorted_prods):
            if prod["quantity"] == 0:
                continue
            prod_w, prod_h = prod["size"]
            for stock_idx, stock in self.sorted_stocks:
                area_not_sufficient = self.stock_area_used[stock_idx] + prod_w * prod_h > self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]
                if area_not_sufficient or not self.can_place_prod[stock_idx, prod_idx]:
                    continue
                stock_w, stock_h = self._get_stock_size_(stock)

                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                            x_pos, y_pos = x, y
                            self.stock_area_used[stock_idx] += prod_w * prod_h
                            return {"stock_idx": stock_idx, "size": (prod_w,prod_h), "position": (x_pos, y_pos)}
                for x in range(stock_w - prod_h + 1):
                    for y in range(stock_h - prod_w + 1):
                        if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                            x_pos, y_pos = x, y
                            self.stock_area_used[stock_idx] += prod_w * prod_h
                            return {"stock_idx": stock_idx, "size": [prod_h,prod_w], "position": (x_pos, y_pos)}
                self.can_place_prod[stock_idx, prod_idx] = False

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}   

class column(Policy):
    def __init__(self):
        self.current_patterns = []
        self.selected_pattern = None
        self.sorted_prods = None
        self.sorted_stocks = None
        self.num_products = 0
        self.dual_values = None
        self.master_solution = None
        self.current_stock_size = None
        self.prod_idx = 0
        self.can_place_prod = None
        self.stock_area_used = None

    def _solve_master_problem(self, demands_vector):
        num_patterns = len(self.current_patterns)
        cvec = np.ones(num_patterns)

        A_eq = np.zeros((self.num_products, num_patterns))
        for j, pattern in enumerate(self.current_patterns):
            A_eq[:, j] = pattern

        b_eq = demands_vector
        result = linprog(cvec, A_eq=A_eq, b_eq=b_eq, method="highs", bounds=(0, None))

        self.master_solution = result["x"]
        self.dual_values = result["eqlin"]["marginals"]


    def create_intial_pattern(self):
        for i in range(self.num_products):
            if self.sorted_prods[i]["quantity"] == 0:
                continue
            counts = np.zeros(self.num_products, dtype=int)
            counts[i] = self.sorted_prods[i]["quantity"]
            pattern = counts
            self.current_patterns.append(pattern)

    def solve_subproblem(self, dual_values = None):
        cvec = dual_values
        if dual_values is None:
            cvec = np.ones(self.num_products)
        cvec *= -1
        A_ub = np.zeros((1, self.num_products))
        b_ub = [self.current_stock_size[0]*self.current_stock_size[1]]
        bounds = [(0, prod["quantity"]) for prod in self.sorted_prods]
        A_ub[0, :] = [prod["size"][0] * prod["size"][1]for prod in self.sorted_prods]
        result = linprog(cvec, A_ub=A_ub, b_ub=b_ub, method="highs", bounds=bounds,integrality=True)
        return np.int64(result["x"])

    def return_action(self):
        for self.prod_idx in range(self.prod_idx,self.num_products):
            count = min(self.selected_pattern[self.prod_idx], self.sorted_prods[self.prod_idx]["quantity"])
            if count == 0:
                continue
            product = self.sorted_prods[self.prod_idx]
            prod_w, prod_h = product["size"]
            for stock_idx,stock in self.sorted_stocks:
                area_not_sufficient = self.stock_area_used[stock_idx] + prod_w * prod_h > self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]
                if area_not_sufficient or not self.can_place_prod[stock_idx, self.prod_idx]:
                    continue
                stock_w, stock_h = self._get_stock_size_(stock)
                for x_pos in range(stock_w - prod_w + 1):
                    for y_pos in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x_pos, y_pos), (prod_w, prod_h)):
                            self.selected_pattern[self.prod_idx] -= 1
                            self.stock_area_used[stock_idx] += prod_w * prod_h
                            return {"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": (x_pos, y_pos)}
                for x_pos in range(stock_w - prod_h + 1):
                    for y_pos in range(stock_h - prod_w + 1):
                        if self._can_place_(stock, (x_pos, y_pos), (prod_h, prod_w)):
                            self.selected_pattern[self.prod_idx] -= 1
                            self.stock_area_used[stock_idx] += prod_w * prod_h
                            return {"stock_idx": stock_idx, "size": [prod_h, prod_w], "position": (x_pos, y_pos)}
                self.can_place_prod[stock_idx, self.prod_idx] = False
        self.selected_pattern = None
        self.current_patterns = []
        self.prod_idx = 0
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            
   
    def get_action(self, observation, info):
        if(self.selected_pattern is not None):
            return self.return_action()
        if info["filled_ratio"] == 0.0:
            prod_average_area = sum(product["size"][0]*product["size"][1] for product in observation["products"]) / len(observation["products"])
            prod_average_quantity = sum(product["quantity"] for product in observation["products"]) / len(observation["products"])
            self.sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1], reverse=True)
            self.sorted_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            stock_average_area = sum(self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for _, stock in self.sorted_stocks) / len(self.sorted_stocks)
            if prod_average_area * prod_average_quantity < stock_average_area:
                self.sorted_stocks = self.sorted_stocks[::-1]
            self.stock_area_used = np.zeros(len(self.sorted_stocks))
            self.num_products = len(self.sorted_prods)   
            self.prod_area = sum([prod["size"][0] * prod["size"][1] * prod["quantity"] for prod in self.sorted_prods])
            self.can_place_prod = np.ones((len(self.sorted_stocks), len(self.sorted_prods)), dtype=bool)
        
        if self.selected_pattern == None:
            self.current_stock_size = self._get_stock_size_(self.sorted_stocks[49][1])
            self.create_intial_pattern()
            demands_vector = np.array([prod["quantity"] for prod in self.sorted_prods])
            for _ in range(100):
                self._solve_master_problem(demands_vector)
                new_pattern = self.solve_subproblem(self.dual_values)
                if(np.any(np.all(self.current_patterns == new_pattern, axis=1))):
                    break
                self.current_patterns.append(new_pattern)
            self._solve_master_problem(demands_vector)
            x = self.master_solution
            pattern_idx = np.argmax(x)
            self.selected_pattern = self.current_patterns[pattern_idx]
            # print(self.current_patterns)
            # print(self.selected_pattern) 
            return self.return_action()