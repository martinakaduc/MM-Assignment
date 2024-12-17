import numpy as np
from scipy.optimize import linprog
from policy import Policy

class Policy2313342_2310569_2313237_2311159_2310497(Policy):
    def __init__(self, policy_id=1):
        self.policy_id = policy_id # policy_id = 1: Column Generation, policy_id = 2: First Fit Decreasing Height (FFDH)
        self.init_cut_patterns = None
        self.required_quantities = None
        self.rotated_dims = None

    def get_action(self, observation, info):
        assert self.policy_id in [1,2], "Policy ID must be 1 or 2"
        if self.policy_id == 1:
            return self._column_generation_get_action_(observation, info)
        elif self.policy_id == 2:
            return self._FFDH_get_action_(observation, info)


    def _FFDH_get_action_(self, observation, info):
        stocks = observation["stocks"]
        products = sorted(
            observation["products"], key=lambda x: x["size"].prod(), reverse=True
        )

        for prod in products:
            size = prod["size"]
            quantity = prod["quantity"]

            if quantity > 0:
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for x in range(stock_w - size[0] + 1):
                        for y in range(stock_h - size[1] + 1):
                            if self._can_place_(stock, (x, y), size):
                                return {"stock_idx": stock_idx, "size": size, "position": (x, y)}

                    for x in range(stock_w - size[1] + 1):
                        for y in range(stock_h - size[0] + 1):
                            if self._can_place_(stock, (x, y), size[::-1]):
                                return {"stock_idx": stock_idx, "size": size[::-1], "position": (x, y)}

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}


    def _column_generation_get_action_(self, observation, info):
        stocks = observation["stocks"]
        items = sorted(
            observation["products"], key=lambda x: x["size"].prod(), reverse=True 
        )

        self._get_data_(items)

        cut_solution = self._solve_column_generation_(stocks)

        return self._decide_action_(stocks, cut_solution)

    def _get_data_(self, items):
        self.required_quantities = np.array([prod["quantity"] for prod in items])
        
        self.item_dims = np.array([prod["size"] for prod in items])
        
        total_items = len(self.item_dims)
        
        self.init_cut_patterns = np.eye(total_items, dtype=int)
        
        self.rotated_dims = np.array([size[::-1] for size in self.item_dims])

    def _solve_column_generation_(self, stocks):
        active_patterns = self.init_cut_patterns

        while True:
            dual_values = self._solve_lp_problem__(active_patterns)

            is_new_pattern, next_pattern = self._find_new_pattern_(dual_values, stocks)

            if not is_new_pattern:
                break

            active_patterns = np.column_stack((active_patterns, next_pattern))

        optimal_stock_count, optimal_solution = self._solve_ilp_problem_(active_patterns)

        return {"cut_patterns": active_patterns,"minimal_stock": optimal_stock_count,"optimal_numbers": optimal_solution}

    def _solve_lp_problem__(self, active_patterns):
        num_vars = active_patterns.shape[1]

        coeff = np.ones(num_vars)

        A_ub = -active_patterns

        b_ub = -self.required_quantities

        bounds = [(0, None) for i in range(num_vars)]

        res = linprog(c=coeff, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res.success:
            return res.slack
        
        else:
            raise ValueError("Could not solve the Linear programming problem.")

    def _solve_ilp_problem_(self, active_patterns):
        num_vars = active_patterns.shape[1]
        coeff = np.ones(num_vars) 
        A_ub = -active_patterns 
        b_ub = -self.required_quantities 
        bounds = [(0, None) for i in range(num_vars)] 

        res = linprog(c=coeff, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res.success:
            alloc_vars = np.round(res.x).astype(int)
            obj_val = alloc_vars.sum()
            return obj_val, alloc_vars
        else:
            raise ValueError("Could not solve the Integer Linear programming problem.")

    def _find_new_pattern_(self, dual_values, stocks):
        num_vars = len(self.item_dims)
        c = dual_values - 1
        A_ub = [np.minimum(self.item_dims[:, 0], self.item_dims[:, 1]),np.maximum(self.item_dims[:, 0], self.item_dims[:, 1])]
        b_ub = [100,100]

        bounds = [(0, None) for i in range(num_vars)] 
        res = linprog(c = c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if res.success:
            decision_vars = np.round(res.x).astype(int)
            obj_val = 1 - dual_values @ decision_vars
            return (True,decision_vars) if obj_val < 0 else (False,None)
        else:
            raise ValueError("Could not find new pattern.")

    def _decide_action_(self, stocks, cut_solution):

        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            for pattern, quantity in zip(cut_solution["cut_patterns"].T, cut_solution["optimal_numbers"]):
                if quantity > 0:
                    for prod_idx, prod_count in enumerate(pattern):
                        if prod_count > 0:
                            size = self.item_dims[prod_idx]
                            if stock_w >= size[0] and stock_h >= size[1]:
                                x, y = self._find_first_fit_(stock, size)
                                if x is not None and y is not None:
                                    return {"stock_idx": stock_idx, "size": size, "position": (x, y)}

                            rotated_size = size[::-1]
                            if stock_w >= rotated_size[0] and stock_h >= rotated_size[1]:
                                x, y = self._find_first_fit_(stock, rotated_size)
                                if x is not None and y is not None:
                                    return {"stock_idx": stock_idx, "size": rotated_size, "position": (x, y)}

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _find_first_fit_(self, stock, size):
        stock_w, stock_h = self._get_stock_size_(stock)
        for x in range(stock_w - size[0] + 1):
            for y in range(stock_h - size[1] + 1):
                if self._can_place_(stock, (x, y), size):
                    return x, y
        return None, None
    
