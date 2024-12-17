from policy import Policy
from scipy.optimize import linprog
import numpy as np

class Policy2310402_2350029_2352797_2352660(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.first_fit_decreasing_method(observation, info)
        elif self.policy_id == 2:
            return self.branch_and_price_method(observation, info)

    def first_fit_decreasing_method(self, observation, info):
        active_prods = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0], 
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )
        stocks = observation["stocks"]

        for prod in active_prods:
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                if stock_w < prod_w or stock_h < prod_h:
                    continue
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}
        return None

    def branch_and_price_method(self, observation, info):
        def check_first_place(observation):
            temps = []
            for idx, product in enumerate(observation['products']):
                if product['quantity'] > 0:
                    pattern = np.zeros(len(observation['products']), dtype=int)
                    pattern[idx] = 1
                    temps.append(pattern)
            return temps

        def solve_rmp(temps, observation):
            num_temps = len(temps)
            num_products = len(observation['products'])

            minimize_stock = np.ones(num_temps)

            constraint = np.zeros((num_products, num_temps))
            for i in range(num_products):
                for j in range(num_temps):
                    constraint[i, j] = temps[j][i]

            bine = [prod['quantity'] for prod in observation['products']]

            bounds = [(0, None)] * num_temps

            result_temp = linprog(minimize_stock, A_ub=-constraint, b_ub=-np.array(bine), bounds=bounds, method='highs')

            if result_temp.success:
                dual_values = result_temp.slack
                result = result_temp.x
                return dual_values, result
            else:
                return None, None

        def solve(dual_values, observation):
            max_size = observation['stocks'][0].shape[0]
            dp = [0] * (max_size + 1)
            backtrack = [None] * (max_size + 1)

            for idx, product in enumerate(observation['products']):
                size = product['size'][0]
                value = dual_values[idx]
                for j in range(max_size, size - 1, -1):
                    if dp[j] < dp[j - size] + value:
                        dp[j] = dp[j - size] + value
                        backtrack[j] = idx

            new_pattern = np.zeros(len(observation['products']), dtype=int)
            current_size = max_size
            while current_size > 0 and backtrack[current_size] is not None:
                idx = backtrack[current_size]
                new_pattern[idx] += 1
                current_size -= observation['products'][idx]['size'][0]

            return new_pattern

        def find_place(pattern, observation):
            for prod_idx, quantity in enumerate(pattern):
                if quantity > 0:
                    product = observation['products'][prod_idx]
                    size = product["size"]
                    for stock_idx, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = size
                        if stock_w >= prod_w and stock_h >= prod_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), size):
                                        return {"stock_idx": stock_idx, "size": size, "position": (x, y)}
            return None

        temps = check_first_place(observation)
        max_is = 100
        i = 0

        while i < max_is:
            i += 1
            dual_values, result = solve_rmp(temps, observation)
            if dual_values is None:
                break

            new_pattern = solve(dual_values, observation)
            if not new_pattern.any():
                break

            temps.append(new_pattern)

            if all(value.is_integer() for value in result):
                break

        for pattern, count in zip(temps, result):
            if count > 0:
                action = find_place(pattern, observation)
                if action:
                    return action
        return None, None