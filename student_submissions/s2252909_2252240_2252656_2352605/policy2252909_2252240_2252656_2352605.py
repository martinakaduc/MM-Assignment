import numpy as np
from policy import Policy, RandomPolicy

class policy2252909_2252240_2252656_2352605_FFD(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        stocks = [{"grid": stock, "placed": []} if not isinstance(stock, dict) else stock for stock in stocks]

        allocation = self.first_fit_decreasing(products, stocks)

        if allocation is None:
            print("First-Fit Decreasing Allocation failed, falling back to random policy.")
            return RandomPolicy().get_action(observation, info)

        for j, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock["grid"])

            for i, product in enumerate(products):
                if allocation[i, j] == 1:
                    prod_w, prod_h = product["size"]

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                self._place_(stock, (x, y), (prod_w, prod_h))
                                return {
                                    "stock_idx": j,
                                    "size": (prod_w, prod_h),
                                    "position": (x, y),
                                }

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def first_fit_decreasing(self, products, stocks):
        products = sorted(products, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        allocation = np.zeros((len(products), len(stocks)), dtype=int)

        for i, product in enumerate(products):
            prod_w, prod_h = product["size"]
            for j, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock["grid"])
                if prod_w <= stock_w and prod_h <= stock_h:
                    allocation[i, j] = 1
                    break

        return allocation

    def _get_stock_size_(self, stock):
        st_w = np.sum(np.any(stock != -2, axis=1))
        st_h = np.sum(np.any(stock != -2, axis=0))
        return st_w, st_h

    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        stock_w, stock_h = self._get_stock_size_(stock["grid"])

        if x + w > stock_w or y + h > stock_h:
            return False

        for placed in stock["placed"]:
            px, py, pw, ph = placed["position"][0], placed["position"][1], placed["size"][0], placed["size"][1]
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                return False

        return True

    def _place_(self, stock, position, size):
        if "placed" not in stock:
            stock["placed"] = []
        stock["placed"].append({"position": position, "size": size})

class policy2252909_2252240_2252656_2352605_BruteForce(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        stocks = [{"grid": stock, "placed": []} if not isinstance(stock, dict) else stock for stock in stocks]

        allocation = self.brute_force_allocation(products, stocks)

        if allocation is None:
            print("Brute Force Allocation failed, falling back to random policy.")
            return RandomPolicy().get_action(observation, info)

        for j, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock["grid"])

            for i, product in enumerate(products):
                if allocation[i, j] == 1:
                    prod_w, prod_h = product["size"]

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                self._place_(stock, (x, y), (prod_w, prod_h))
                                return {
                                    "stock_idx": j,
                                    "size": (prod_w, prod_h),
                                    "position": (x, y),
                                }

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def brute_force_allocation(self, products, stocks):
        num_products = len(products)
        num_stocks = len(stocks)
        best_allocation = None
        best_unused_area = float('inf')

        for allocation in np.ndindex((2,) * (num_products * num_stocks)):
            allocation = np.array(allocation).reshape((num_products, num_stocks))
            if np.all(np.sum(allocation, axis=1) == 1):
                unused_area = 0
                for j in range(num_stocks):
                    used_area = sum(products[i]["size"][0] * products[i]["size"][1] for i in range(num_products) if allocation[i, j] == 1)
                    stock_w, stock_h = self._get_stock_size_(stocks[j]["grid"])
                    stock_area = stock_w * stock_h
                    unused_area += max(0, stock_area - used_area)
                if unused_area < best_unused_area:
                    best_unused_area = unused_area
                    best_allocation = allocation

        return best_allocation

    def _get_stock_size_(self, stock):
        st_w = np.sum(np.any(stock != -2, axis=1))
        st_h = np.sum(np.any(stock != -2, axis=0))
        return st_w, st_h

    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        stock_w, stock_h = self._get_stock_size_(stock["grid"])

        if x + w > stock_w or y + h > stock_h:
            return False

        for placed in stock["placed"]:
            px, py, pw, ph = placed["position"][0], placed["position"][1], placed["size"][0], placed["size"][1]
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                return False

        return True

    def _place_(self, stock, position, size):
        if "placed" not in stock:
            stock["placed"] = []
        stock["placed"].append({"position": position, "size": size})

class policy2252909_2252240_2252656_2352605_DP(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        stocks = [{"grid": stock, "placed": []} if not isinstance(stock, dict) else stock for stock in stocks]

        allocation = self.dynamic_programming_allocation(products, stocks)

        if allocation is None:
            print("Dynamic Programming Allocation failed, falling back to random policy.")
            return RandomPolicy().get_action(observation, info)

        for j, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock["grid"])

            for i, product in enumerate(products):
                if allocation[i, j] == 1:
                    prod_w, prod_h = product["size"]

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                self._place_(stock, (x, y), (prod_w, prod_h))
                                return {
                                    "stock_idx": j,
                                    "size": (prod_w, prod_h),
                                    "position": (x, y),
                                }

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def dynamic_programming_allocation(self, products, stocks):
        num_products = len(products)
        num_stocks = len(stocks)
        dp = np.zeros((num_products + 1, num_stocks + 1), dtype=int)

        for i in range(1, num_products + 1):
            for j in range(1, num_stocks + 1):
                prod_w, prod_h = products[i - 1]["size"]
                stock_w, stock_h = self._get_stock_size_(stocks[j - 1]["grid"])
                if prod_w <= stock_w and prod_h <= stock_h:
                    dp[i, j] = max(dp[i - 1, j], dp[i - 1, j - 1] + prod_w * prod_h)
                else:
                    dp[i, j] = dp[i - 1, j]

        allocation = np.zeros((num_products, num_stocks), dtype=int)
        i, j = num_products, num_stocks
        while i > 0 and j > 0:
            if dp[i, j] != dp[i - 1, j]:
                allocation[i - 1, j - 1] = 1
                j -= 1
            i -= 1

        return allocation

    def _get_stock_size_(self, stock):
        st_w = np.sum(np.any(stock != -2, axis=1))
        st_h = np.sum(np.any(stock != -2, axis=0))
        return st_w, st_h

    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        stock_w, stock_h = self._get_stock_size_(stock["grid"])

        if x + w > stock_w or y + h > stock_h:
            return False

        for placed in stock["placed"]:
            px, py, pw, ph = placed["position"][0], placed["position"][1], placed["size"][0], placed["size"][1]
            if not (x + w <= px or px + pw <= x or y + h <= py or py + ph <= y):
                return False

        return True

    def _place_(self, stock, position, size):
        if "placed" not in stock:
            stock["placed"] = []
        stock["placed"].append({"position": position, "size": size})
