import numpy as np
from scipy.optimize import linprog
from policy import Policy


class Policy2350008_2352123_2353047_2353302_2353344(Policy):
    def __init__(self, policy_id=1, stock_width=50, stock_height=50):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.stock_width = stock_width
        self.stock_height = stock_height
        self.initial_cut_patterns = None
        self.required_quantities = None
        self.rotated_dimensions = None
        self.previous_solution_sub = None
        self.subproblem_cache = {}
        self.new_pattern_cache = {}
        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.first_fit_decreasing(observation)
        elif self.policy_id == 2:
            return self.column_generation(observation, info)

    def first_fit_decreasing(self, observation):
        products = list(observation["products"]) 
        products.sort(key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        for prod in products:
            if prod["quantity"] <= 0:
                continue
            prod_size = prod["size"]
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                for size in [prod_size, prod_size[::-1]]:
                    prod_w, prod_h = size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    pos_to_place = self.best_pos(size, stock)
                    if pos_to_place:    
                        return {"stock_idx": stock_idx, "size": size, "position": pos_to_place}
                    
                
    def column_generation(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        patterns = []
        for product in products:
            pattern = np.zeros(len(products)) 
            for idx, prod in enumerate(products):
                if np.array_equal(prod["size"], product["size"]) and prod["quantity"] == product["quantity"]:
                    pattern[idx] = 1
                    break
            patterns.append(pattern)

        demands = np.array([product["quantity"] for product in products])
        while True:
            c = np.ones(len(patterns))
            A_eq = np.array(patterns).T
            b_eq = demands
            bounds = [(0, None)] * len(patterns)

            res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if res.success:
                shadow_prices = res.get("slack", None)
                if shadow_prices is None:
                    raise ValueError("Shadow prices are not available in the solution.")
            else:
                return self.first_fit_decreasing(observation)

            new_pattern, new_trim_loss = self.solve_pricing_problem(stocks, products, shadow_prices)
            if new_trim_loss >= 0:
                break
            patterns.append(new_pattern)
        solution = res.x
        actions = self.generate_actions_from_solution(solution, patterns, products, stocks)
        return actions

    def solve_pricing_problem(self, stocks, products, shadow_prices):
        best_pattern = None
        best_trim_loss = float("inf")
        for stock_idx, stock in enumerate(stocks):
            pattern = np.zeros(len(products))
            trim_loss = 0
            if trim_loss < best_trim_loss:
                best_pattern = pattern
                best_trim_loss = trim_loss

        return best_pattern, best_trim_loss

    def generate_actions_from_solution(self, solution, patterns, products, stocks):
        for pattern_idx, count in enumerate(solution):
            if count > 0:
                selected_pattern = patterns[pattern_idx]
                for product_idx, quantity in enumerate(selected_pattern):
                    if quantity > 0:
                        product = products[product_idx]
                        prod_size = product["size"]

                        for stock_idx, stock in enumerate(stocks):
                            for size in [prod_size, prod_size[::-1]]:
                                position = self.best_pos(size, stock)
                                if position:
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": size,
                                        "position": position
                                    }

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
    def best_pos(self, size, stock):
        prod_w, prod_h = size
        stock_w, stock_h = self._get_stock_size_(stock)
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), size):
                    return (x, y)
    # Student code here
    # You can add more functions if needed