from policy import Policy
import numpy as np
import random

class Policy2353149_2352618_2353134_2352861_2353119(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy_get_action = self.simulated_annealing_get_action
        elif policy_id == 2:
            self.policy_get_action = self.dynamic_programming_get_action

    def get_action(self, observation, info):
        """
        Unified method to call the appropriate policy's action.
        """
        return self.policy_get_action(observation, info)

    def simulated_annealing_get_action(self, observation, info):
        """
        Choose an action using simulated annealing.
        """
        list_prods = sorted(
            observation["products"], key=lambda x: -x["size"][0] * x["size"][1]
        )

        best_solution = None
        best_wasted_space = float("inf")

        def calculate_wasted_space(stock, remaining_prods):
            stock_area = np.sum(stock == -1)
            used_area = sum(prod["size"][0] * prod["size"][1] for prod in remaining_prods)
            return stock_area - used_area

        def simulated_annealing(stock_idx, stock, remaining_prods):
            nonlocal best_solution, best_wasted_space
            T = 1000
            cooling_rate = 0.95
            max_iterations = 5

            current_solution = []
            current_stock = stock.copy()
            current_wasted_space = calculate_wasted_space(current_stock, remaining_prods)

            for _ in range(max_iterations):
                if not remaining_prods:
                    break

                prod = random.choice(remaining_prods)
                prod_w, prod_h = prod["size"]

                stock_w, stock_h = self._get_stock_size_(current_stock)

                valid_positions = [
                    (x, y)
                    for x in range(stock_w - prod_w + 1)
                    for y in range(stock_h - prod_h + 1)
                    if self._can_place_(current_stock, (x, y), (prod_w, prod_h))
                ]

                if valid_positions:
                    pos_x, pos_y = random.choice(valid_positions)
                    new_stock = current_stock.copy()
                    new_stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] = 1
                    new_remaining_prods = [
                        p for p in remaining_prods
                        if not (p["size"][0] == prod["size"][0] and p["size"][1] == prod["size"][1])
                    ]
                    new_wasted_space = calculate_wasted_space(new_stock, new_remaining_prods)

                    delta = new_wasted_space - current_wasted_space
                    acceptance_prob = np.exp(-delta / T) if delta > 0 else 1

                    if random.random() < acceptance_prob:
                        current_solution.append((stock_idx, (pos_x, pos_y), prod["size"]))
                        current_stock = new_stock
                        current_wasted_space = new_wasted_space
                        remaining_prods = new_remaining_prods

                        if current_wasted_space < best_wasted_space:
                            best_solution = current_solution.copy()
                            best_wasted_space = current_wasted_space

                T *= cooling_rate

        for i, stock in enumerate(observation["stocks"]):
            stock = np.array(stock, dtype=np.int32)
            if stock.size == 0:
                continue
            simulated_annealing(i, stock, list_prods)

        if best_solution:
            best_action = best_solution[0]
            return {"stock_idx": best_action[0], "size": best_action[2], "position": best_action[1]}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def dynamic_programming_get_action(self, observation, info):
        """
        Choose an action using a dynamic programming algorithm.
        """
        list_prods = sorted(
            observation["products"], key=lambda x: -x["size"][0] * x["size"][1]
        )

        dp_cache = {}

        def dp(stock_idx, remaining_prods):
            cache_key = (stock_idx, tuple([tuple(prod["size"]) for prod in remaining_prods]))
            if cache_key in dp_cache:
                return dp_cache[cache_key]

            if not remaining_prods:
                return 0, []

            if stock_idx >= len(observation["stocks"]):
                return float("inf"), []

            stock = np.array(observation["stocks"][stock_idx], dtype=np.int32)
            stock_area = np.sum(stock == -1)
            stock_w, stock_h = self._get_stock_size_(stock)

            best_wasted_space = float("inf")
            best_solution = []

            prod = remaining_prods[0]
            prod_w, prod_h = prod["size"]

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                        new_stock = stock.copy()
                        new_stock[x : x + prod_w, y : y + prod_h] = 1
                        new_wasted_space = stock_area - (prod_w * prod_h)

                        rec_wasted_space, rec_solution = dp(
                            stock_idx,
                            remaining_prods[1:],
                        )

                        total_wasted_space = new_wasted_space + rec_wasted_space

                        if total_wasted_space < best_wasted_space:
                            best_wasted_space = total_wasted_space
                            best_solution = [(stock_idx, (x, y), prod["size"])] + rec_solution

            skip_wasted_space, skip_solution = dp(stock_idx + 1, remaining_prods)

            if skip_wasted_space < best_wasted_space:
                best_wasted_space = skip_wasted_space
                best_solution = skip_solution

            dp_cache[cache_key] = (best_wasted_space, best_solution)
            return best_wasted_space, best_solution

        _, best_solution = dp(0, list_prods)

        if best_solution:
            best_action = best_solution[0]
            return {"stock_idx": best_action[0], "size": best_action[2], "position": best_action[1]}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
