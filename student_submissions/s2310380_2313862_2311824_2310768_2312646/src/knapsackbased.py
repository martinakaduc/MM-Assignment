from policy import Policy
import numpy as np

class KnapsackBased(Policy):
    def _init_(self):
        pass

    def solve_knapsack(self, weights, values, capacity):
        n = len(values)
        dp = np.zeros((n + 1, capacity + 1), dtype=int)

        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]
        result = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                result.append(i - 1)
                w -= weights[i - 1]
    
        return dp[n][capacity], result

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        products = [prod for prod in list_prods if prod["quantity"] > 0]
        if not products:
            return {"stock_idx": -1, "size": 0, "position": (0, 0)}
        best_stock_idx, best_pos_x, best_pos_y = -1, 0, 0
        best_product_size = [0, 0]
        min_waste = float('inf')
        selected_product = None

        for i, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            capacity = stock_w * stock_h

            values = [prod["quantity"] for prod in products]
            weights = [prod["size"][0] * prod["size"][1] for prod in products]

            _, product_need = self.solve_knapsack(weights, values, capacity)

            for idx in product_need:
                prod = products[idx]
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                if(stock_w >= prod_w and stock_h >= prod_h):
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        used_area = np.sum(stock != -1)
                        remaining_area = stock_w * stock_h - used_area - (prod_size[0] * prod_size[1])
                        if(best_stock_idx == -1 and selected_product == None):
                            best_stock_idx = i
                            selected_product = prod
                        if remaining_area < min_waste:
                            min_waste = remaining_area
                            best_stock_idx = i
                            best_pos_x, best_pos_y = pos_x, pos_y
                            best_product_size = prod_size
                            selected_product = prod
                        
                if(stock_w >= prod_h and stock_h >= prod_w):
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                prod_size = prod_size[::-1]
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        used_area = np.sum(stock != -1)
                        remaining_area = stock_w * stock_h - used_area - (prod_size[0] * prod_size[1])
                        if(best_stock_idx == -1 and selected_product == None):
                            best_stock_idx = i
                            selected_product = prod
                        if remaining_area < min_waste:
                            min_waste = remaining_area
                            best_stock_idx = i
                            best_pos_x, best_pos_y = pos_x, pos_y
                            best_product_size = prod_size
                            selected_product = prod
            if pos_x is not None and pos_y is not None:
                break
                    
        return {"stock_idx": best_stock_idx, "size": best_product_size, "position": (best_pos_x, best_pos_y)}