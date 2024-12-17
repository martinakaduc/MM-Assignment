import numpy as np
from policy import Policy


class Policy2352967_2352758_2352968_2353139_2352495(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        if policy_id == 1:
            self.policy_name = "Guillotine Policy"
        elif policy_id == 2:
            self.policy_name = "Heuristic Best-Fit Policy"

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._guillotine_policy(observation, info)
        elif self.policy_id == 2:
            return self._GreedyBestFitDecreasing_policy(observation, info)

    def _guillotine_policy(self, observation, info):
        stocks = observation['stocks']
        products = observation['products']
        for product_idx, product in enumerate(products):
            if product['quantity'] <= 0:
                continue
            product_length, product_width = product['size']
            orientations = [
                (product_length, product_width),
                (product_width, product_length)
            ]
            for length, width in orientations:
                for stock_idx, stock in enumerate(stocks):
                    stock_length, stock_width = stock.shape
                    if stock_length >= length and stock_width >= width:
                        for i in range(stock_length - length + 1):
                            for j in range(stock_width - width + 1):
                                if self._can_place(stock, i, j, length, width):
                                    products[product_idx]['quantity'] -= 1
                                    return {
                                        'stock_idx': stock_idx,
                                        'size': (length, width),
                                        'position': (i, j)
                                    }
        return {"stock_idx": 0, "size": (1, 1), "position": (0, 0)}

    def _GreedyBestFitDecreasing_policy(self, observation, info):
        products = sorted(
            observation["products"],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )
        stocks = observation["stocks"]
        for product_idx, product in enumerate(products):
            if product["quantity"] > 0:
                prod_size = tuple(product["size"])
                for stock_idx, stock in enumerate(stocks):
                    best_position, best_orientation = self._find_best_fit(stock, prod_size)
                    if best_position is not None:
                        x, y = best_position
                        final_size = best_orientation
                        self._update_stock(stock, (x, y), final_size)
                        product["quantity"] -= 1
                        return {
                            "stock_idx": stock_idx,
                            "size": final_size,
                            "position": (x, y)
                        }
        return {"stock_idx": 0, "size": (1, 1), "position": (0, 0)}

    def _find_best_fit(self, stock, prod_size):
        stock_length, stock_width = stock.shape
        best_position = None
        best_orientation = None
        min_waste = float("inf")
        for orientation in [(prod_size[0], prod_size[1]), (prod_size[1], prod_size[0])]:
            length, width = orientation
            for i in range(stock_length - length + 1):
                for j in range(stock_width - width + 1):
                    if self._can_place(stock, i, j, length, width):
                        waste = self._calculate_waste(stock, (i, j), (length, width))
                        if waste < min_waste:
                            min_waste = waste
                            best_position = (i, j)
                            best_orientation = (length, width)
        return best_position, best_orientation

    def _calculate_waste(self, stock, position, size):
        x, y = position
        length, width = size
        stock_area = stock.shape[0] * stock.shape[1]
        used_area = np.sum(stock == -1) + (length * width)
        return stock_area - used_area

    def _update_stock(self, stock, position, prod_size):
        x, y = position
        length, width = prod_size
        stock[x:x + length, y:y + width] = -1

    def _can_place(self, stock, i, j, length, width):
        for x in range(i, i + length):
            for y in range(j, j + width):
                if stock[x, y] != -1:
                    return False
        return True
