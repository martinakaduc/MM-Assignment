import numpy as np
from scipy.optimize import linprog
from policy import Policy

class Policy2310393_2311514_2310273_2311428(Policy):
    def __init__(self, policy_id=1, stock_width=50, stock_height=50):
        self.policy_id = policy_id # policy_id = 1: Bottom Left Fill (BLF), policy_id = 2: Column Generation
        self.stock_width = stock_width
        self.stock_height = stock_height
        self.initial_cut_patterns = None
        self.required_quantities = None
        self.rotated_dimensions = None

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._blf_get_action_(observation, info)
        else:
            return self._column_generation_get_action_(observation, info)

    def _blf_get_action_(self, observation, info):
        list_prods = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )

        observation["products"] = tuple(list_prods)

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Iterate to find the bottom-left placement
                    pos_x, pos_y = None, None
                    best_x, best_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                if best_x is None or y > best_y or (y == best_y and x < best_x):
                                    best_x, best_y = x, y

                    if best_x is not None and best_y is not None:
                        stock_idx = i
                        pos_x, pos_y = best_x, best_y
                        break

                    # Check rotated placement
                    if stock_w >= prod_h and stock_h >= prod_w:
                        best_x, best_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    if best_x is None or y > best_y or (y == best_y and x < best_x):
                                        best_x, best_y = x, y
                        if best_x is not None and best_y is not None:
                            prod_size = prod_size[::-1]
                            stock_idx = i
                            pos_x, pos_y = best_x, best_y
                            break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def _column_generation_get_action_(self, observation, info):
        stocks = observation["stocks"]
        items = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )

        observation["products"] = items

        self._prepare_data_(items)

        # Solve with column generation
        cut_solution = self._process_column_generation_(stocks)

        # Choose an action based on the solution
        move = self._determine_action_(stocks, cut_solution)
        return move

    def _prepare_data_(self, items):
        self.required_quantities = np.array([prod["quantity"] for prod in items])
        self.item_dimensions = np.array([prod["size"] for prod in items])
        total_items = len(self.item_dimensions)
        self.initial_cut_patterns = np.eye(total_items, dtype=int)
        self.rotated_dimensions = np.array([[size[1], size[0]] for size in self.item_dimensions])

    def _process_column_generation_(self, stocks):
        is_new_pattern = True
        next_pattern = None
        active_patterns = self.initial_cut_patterns

        while is_new_pattern:
            if next_pattern is not None:
                active_patterns = np.column_stack((active_patterns, next_pattern))

            dual_values = self._solve_lp_problem__(active_patterns)
            is_new_pattern, next_pattern = self._find_new_pattern_(dual_values, stocks)

        optimal_stock_count, optimal_solution = self._solve_ip_problem_(active_patterns)
        return {"cut_patterns": active_patterns, "minimal_stock": optimal_stock_count, "optimal_numbers": optimal_solution}

    def _solve_lp_problem__(self, active_patterns):
        num_vars = active_patterns.shape[1]
        c = np.ones(num_vars)
        A = -active_patterns
        b = -self.required_quantities
        bounds = [(0, None) for _ in range(num_vars)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        if res.success:
            return res.slack
        else:
            raise ValueError("Linear programming problem could not be solved.")

    def _solve_ip_problem_(self, active_patterns):
        num_vars = active_patterns.shape[1]
        c = np.ones(num_vars)
        A = -active_patterns
        b = -self.required_quantities
        bounds = [(0, None) for _ in range(num_vars)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        if res.success:
            allocation_vars = np.round(res.x).astype(int) # Round to integers
            obj_val = allocation_vars.sum()
            return obj_val, allocation_vars
        else:
            raise ValueError("Integer programming problem could not be solved.")

    def _find_new_pattern_(self, dual_values, stocks):
        num_vars = len(self.item_dimensions)
        c = dual_values - 1
        A = [
            np.minimum(self.item_dimensions[:, 0], self.item_dimensions[:, 1]),
            np.maximum(self.item_dimensions[:, 0], self.item_dimensions[:, 1])
        ]
        b = [self.stock_width, self.stock_height]
        bounds = [(0, None) for _ in range(num_vars)]
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        if res.success:
            decision_vars = np.round(res.x).astype(int) # Round to integers
            obj_val = 1 - dual_values @ decision_vars
            if obj_val < 0:
                return True, decision_vars
            else:
                return False, None
        else:
            raise ValueError("Pattern finding problem could not be solved.")

    def _determine_action_(self, stocks, cut_solution):
        stock_idx = 0
        while stock_idx < len(stocks):
            stock = stocks[stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)

            for pattern, qty in zip(cut_solution["cut_patterns"].T, cut_solution["optimal_numbers"]):
                if qty > 0:
                    for prod_idx, prod_count in enumerate(pattern):
                        if prod_count > 0:
                            prod_dim = self.item_dimensions[prod_idx]

                            if stock_w >= prod_dim[0] and stock_h >= prod_dim[1]:
                                for x_pos in range(stock_w - prod_dim[0] + 1):
                                    for y_pos in range(stock_h - prod_dim[1] + 1):
                                        if self._can_place_(stock, (x_pos, y_pos), prod_dim):
                                            return {
                                                "stock_idx": stock_idx,
                                                "size": prod_dim,
                                                "position": (x_pos, y_pos)
                                            }

                            rotated_dim = [prod_dim[1], prod_dim[0]]
                            if stock_w >= rotated_dim[0] and stock_h >= rotated_dim[1]:
                                for x_pos in range(stock_w - rotated_dim[0] + 1):
                                    for y_pos in range(stock_h - rotated_dim[1] + 1):
                                        if self._can_place_(stock, (x_pos, y_pos), rotated_dim):
                                            return {
                                                "stock_idx": stock_idx,
                                                "size": rotated_dim,
                                                "position": (x_pos, y_pos)
                                            }
            stock_idx += 1

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
