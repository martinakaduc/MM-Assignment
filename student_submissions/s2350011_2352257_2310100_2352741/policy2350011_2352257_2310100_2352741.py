from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2350011_2352257_2310100_2352741(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.FFD(observation, info) # First Fit Decreasing
        elif self.policy_id == 2:
            return self.BAP(observation, info) # Branch and Price

    def FFD(self, current_state, metadata):
        products = sorted(
            current_state["products"],
            key=lambda item: (item["size"][0] * item["size"][1], max(item["size"])),
            reverse=True
        )
        stocks = current_state["stocks"]
        remainings = [item for item in products if item["quantity"] > 0]

        for product in remainings:
            pWidth, pLength = product["size"]
            for sIndex, stock in enumerate(stocks):
                sWidth, sLength = self._get_stock_size_(stock)
                if sLength >= pLength and sWidth >= pWidth:
                    for startWidth in range(sWidth - pWidth + 1):
                        for startLength in range(sLength - pLength + 1):
                            if self._can_place_(stock, (startWidth, startLength), (pWidth, pLength)):
                                return {
                                    "stock_idx": sIndex,
                                    "size": (pWidth, pLength),
                                    "position": (startWidth, startLength),
                                }
        return None

    def BAP(self, current_state, metadata):
        def initialize_base_patterns(state):
            products = state['products']
            n = len(products)
            patterns = []
            for i, p in enumerate(products):
                if p['quantity'] > 0:
                    vec = np.zeros(n, dtype=int)
                    vec[i] = 1
                    patterns.append(vec)
            return patterns

        def solve_master_problem(patterns, state):
            products = state['products']
            num_products = len(products)
            num_patterns = len(patterns)
            if num_patterns == 0:
                return np.zeros(num_products), np.inf, np.array([])
            c = np.ones(num_patterns)
            A_ub = []
            b_ub = []
            for i, prod in enumerate(products):
                A_ub.append([-pt[i] for pt in patterns])
                b_ub.append(-prod['quantity'])
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(0, None)] * num_patterns
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                duals = np.zeros(num_products)
                obj_val = np.inf
                sol = np.zeros(num_patterns)
            else:
                duals = -res.ineqlin.marginals
                obj_val = res.fun
                sol = res.x
            return duals, obj_val, sol

        def find_new_pattern(dual_values, state):
            products = state['products']
            num_products = len(products)
            if not state['stocks']:
                return np.zeros(num_products, dtype=int)
            max_width, _ = self._get_stock_size_(state['stocks'][0])
            dp = np.zeros(max_width + 1)
            backtrack = [None] * (max_width + 1)
            product_sizes = [p['size'][0] for p in products]
            for prod_idx, (size, val) in enumerate(zip(product_sizes, dual_values)):
                for w in range(max_width, size - 1, -1):
                    cand_val = dp[w - size] + val
                    if cand_val > dp[w]:
                        dp[w] = cand_val
                        backtrack[w] = prod_idx
            new_pattern = np.zeros(num_products, dtype=int)
            cur_w = max_width
            while cur_w > 0 and backtrack[cur_w] is not None:
                chosen = backtrack[cur_w]
                new_pattern[chosen] += 1
                cur_w -= product_sizes[chosen]
            return new_pattern

        def locate_placement(pattern, state):
            products = state['products']
            stocks = state['stocks']
            needed = np.where(pattern > 0)[0]
            if needed.size == 0:
                return None
            prod_idx = needed[0]
            product = products[prod_idx]
            pw, ph = product['size']
            for s_idx, stock in enumerate(stocks):
                sw, sh = self._get_stock_size_(stock)
                if sw >= pw and sh >= ph:
                    limit_x = sw - pw + 1
                    limit_y = sh - ph + 1
                    for x_coord in range(limit_x):
                        for y_coord in range(limit_y):
                            if self._can_place_(stock, (x_coord, y_coord), (pw, ph)):
                                return {"stock_idx": s_idx, "size": [pw, ph], "position": (x_coord, y_coord)}
            return None

        patterns = initialize_base_patterns(current_state)
        if not patterns:
            return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}
        max_iterations = metadata.get("max_iterations", 100)
        for _ in range(max_iterations):
            dual_vals, obj_val, solution = solve_master_problem(patterns, current_state)
            if np.isinf(obj_val):
                break
            new_pattern = find_new_pattern(dual_vals, current_state)
            if new_pattern.sum() == 0:
                break
            patterns.append(new_pattern)
            if np.allclose(solution, np.round(solution)):
                break
        if solution.size == 0 or solution.max() <= 0:
            return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}
        for chosen_pattern, usage_count in zip(patterns, solution):
            if usage_count > 0:
                placement = locate_placement(chosen_pattern, current_state)
                if placement is not None:
                    return placement
        return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}