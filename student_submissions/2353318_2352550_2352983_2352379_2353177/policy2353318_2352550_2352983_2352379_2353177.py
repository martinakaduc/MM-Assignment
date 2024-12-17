from policy import Policy
import numpy as np


class Policy2353318_2352550_2352983_2352379_2353177(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            products = observation["products"]
            stocks = observation["stocks"]

            for product in products:
                if product["quantity"] > 0:
                    prod_size = product["size"]
                    action = self._find_placement_in_stocks(prod_size, stocks)
                    if action is not None:
                        return action

            # If no valid placement is found
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        elif self.policy_id == 2:
            products = observation["products"]
            stocks = observation["stocks"]

            # Step 1: Select the largest product available
            selected_product = self.select_product(products)
            if not selected_product:
                return None

            product_idx, product = selected_product
            product_size = product["size"]

            # Step 2: Attempt allocation in stocks
            allocation = self.attempt_allocation(stocks, product_size)
            if allocation:
                return {
                    "stock_idx": allocation[0],
                    "size": product_size,
                    "position": allocation[1],
                }

            return None

    def _find_placement_in_stocks(self, prod_size, stocks):
        for stock_idx, stock in enumerate(stocks):
            placement = self._find_placement_in_stock(stock, prod_size)
            if placement is not None:
                pos_x, pos_y, size = placement
                return {
                    "stock_idx": stock_idx,
                    "size": size,
                    "position": (pos_x, pos_y),
                }

        return None

    def _find_placement_in_stock(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)

        # Check original orientation
        placement = self._find_position(stock, prod_size, stock_w, stock_h)
        if placement is not None:
            return placement + (prod_size,)

        # Check rotated orientation
        rotated_size = prod_size[::-1]
        placement = self._find_position(stock, rotated_size, stock_w, stock_h)
        if placement is not None:
            return placement + (rotated_size,)

        return None

    def _find_position(self, stock, prod_size, stock_w, stock_h):
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None
    def select_product(self, products):
        """
        Select the first product with a quantity greater than 0.
        """
        for product_idx, product in enumerate(products):
            if product["quantity"] > 0:
                return product_idx, product
        return None

    def attempt_allocation(self, stocks, product_size):
        """
        Try to allocate the product to any feasible position in the stocks.
        """
        for stock_idx, stock in enumerate(stocks):
            stock_width, stock_height = self.get_stock_dimensions(stock)

            # Tìm vị trí khả thi
            position = self.find_feasible_position(
                stock, stock_width, stock_height, product_size
            )
            if position:
                return stock_idx, position

        return None

    def find_feasible_position(self, stock, stock_width, stock_height, product_size):
        """
        Find the first feasible position in the stock where the product can fit.
        """
        for x in range(stock_width - product_size[0] + 1):
            for y in range(stock_height - product_size[1] + 1):
                if self.is_position_feasible(stock, x, y, product_size):
                    return (x, y)
        return None

    def get_stock_dimensions(self, stock):
        """
        Calculate the dimensions of the stock by counting non-empty rows and columns.
        """
        return np.sum(np.any(stock != -2, axis=1)), np.sum(np.any(stock != -2, axis=0))

    def is_position_feasible(self, stock, x, y, product_size):
        """
        Check if a product of the given size can be placed at position (x, y) in the stock.
        """
        return np.all(stock[x : x + product_size[0], y : y + product_size[1]] == -1)
