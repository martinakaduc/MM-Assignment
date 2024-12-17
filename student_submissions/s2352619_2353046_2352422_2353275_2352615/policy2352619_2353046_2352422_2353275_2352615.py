from policy import Policy
import numpy as np

class Policy2352619_2353046_2352422_2353275_2352615(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def _get_state_key(self, observation, info):
        return str(observation)

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._GreedyDP_action(observation, info)
        else:
            return self._Backtracking_action(observation, info)

    def _Backtracking_action(self, observation, info):
        list_prods = observation["products"]

        stock_idx = -1
        pos_x, pos_y = 0, 0
        selected_size = None

        def backtrack(stock, prod_size, x, y, visited):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            if self._can_place_(stock, (x, y), prod_size):
                return (x, y)

            for nx in range(stock_w - prod_w + 1):
                for ny in range(stock_h - prod_h + 1):
                    if (nx, ny) not in visited and self._can_place_(stock, (nx, ny), prod_size):
                        visited.add((nx, ny))
                        result = backtrack(stock, prod_size, nx, ny, visited)
                        if result:
                            return result
                        visited.remove((nx, ny))
            return None

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    visited = set()
                    result = backtrack(stock, prod_size, 0, 0, visited)
                    if result:
                        pos_x, pos_y = result
                        stock_idx = i
                        selected_size = prod_size
                        break

                    rotated_size = prod_size[::-1]
                    visited = set()
                    result = backtrack(stock, rotated_size, 0, 0, visited)
                    if result:
                        pos_x, pos_y = result
                        stock_idx = i
                        selected_size = rotated_size
                        break

                if stock_idx != -1:
                    break

        return {"stock_idx": stock_idx, "size": selected_size, "position": (pos_x, pos_y)}

    def _GreedyDP_action(self, observation, info):
        list_prods = observation["products"]

        stock_idx = -1
        pos_x, pos_y = 0, 0
        selected_size = None

        def dp(stock, prod_size):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            dp_table = [[False] * stock_h for _ in range(stock_w)]

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), prod_size):
                        dp_table[x][y] = True

            for x in range(stock_w):
                for y in range(stock_h):
                    if dp_table[x][y]:
                        return (x, y)

            return None

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    result = dp(stock, prod_size)
                    if result:
                        pos_x, pos_y = result
                        stock_idx = i
                        selected_size = prod_size
                        break

                    rotated_size = prod_size[::-1]
                    result = dp(stock, rotated_size)
                    if result:
                        pos_x, pos_y = result
                        stock_idx = i
                        selected_size = rotated_size
                        break

                if stock_idx != -1:
                    break

        return {"stock_idx": stock_idx, "size": selected_size, "position": (pos_x, pos_y)}


