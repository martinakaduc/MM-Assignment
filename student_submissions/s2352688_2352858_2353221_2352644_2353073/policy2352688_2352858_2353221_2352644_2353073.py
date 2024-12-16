from policy import Policy
import numpy as np

class Policy2352688_2352858_2353221_2352644_2353073(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        
    def get_action(self, observation, info):
        # Pre-process data based on the policy ID
        products = observation["products"]
        stocks = observation["stocks"]

        if self.policy_id == 1:
            # Sort products for FFD
            products = sorted(products, key=lambda p: (-p["size"][0], -p["size"][1]))
            return self.first_fit_decreasing(observation, products, stocks)
        elif self.policy_id == 2:
            # Sort products and stocks for BFD
            products = sorted(products, key=lambda p: -(p["size"][0] * p["size"][1]))
            stocks_with_indices = list(enumerate(stocks))
            stocks_with_indices = sorted(stocks_with_indices, key=lambda p: (self._get_stock_size_(p[1])[0] * self._get_stock_size_(p[1])[1]))
            stocks = [stock for _, stock in stocks_with_indices]
            sorted_indices = [idx for idx, _ in stocks_with_indices]
            return self.best_fit_decreasing(observation, products, stocks, sorted_indices)

    def first_fit_decreasing(self, observation, products, stocks):
        for product in products:
            product_size = product["size"]
            
            product_quantity = product["quantity"]

            while product_quantity > 0:
                placed = False
                for stock_idx, stock in enumerate(stocks):
                    pos_x, pos_y = self._find_position(stock, product_size)
                    
                    if pos_x is not None and pos_y is not None:
                        product_quantity -= 1
                        placed = True
                        return {
                            "stock_idx": stock_idx,
                            "size": product_size,
                            "position": (pos_x, pos_y),
                        }
                        
                # Try rotating the product if it doesn't fit
                if not placed:
                    for stock_idx, stock in enumerate(stocks):
                        pos_x, pos_y = self._find_position(stock, product_size[::-1])
                        if pos_x is not None and pos_y is not None:
                            product_quantity -= 1
                            placed = True
                            return {
                                "stock_idx": stock_idx,
                                "size": product_size[::-1],
                                "position": (pos_x, pos_y),
                            }

                # If no stock can fit the product
                if not placed:
                    break

        return None

    def best_fit_decreasing(self, observation, products, stocks, sorted_indices):
        for stock_idx, stock in enumerate(stocks):
            for product in products:
                product_size = product["size"]
                product_quantity = product["quantity"]

                while product_quantity > 0:
                    placed = False

                    # Attempt to place the product in its original orientation
                    pos_x, pos_y = self._find_position(stock, product_size)
                    if pos_x is not None and pos_y is not None:
                        product_quantity -= 1
                        placed = True
                        return {
                            "stock_idx": sorted_indices[stock_idx],
                            "size": product_size,
                            "position": (pos_x, pos_y),
                        }

                    # Attempt to place the product in its rotated orientation
                    pos_x, pos_y = self._find_position(stock, product_size[::-1])
                    if pos_x is not None and pos_y is not None:
                        product_quantity -= 1
                        placed = True
                        return {
                            "stock_idx": sorted_indices[stock_idx],
                            "size": product_size[::-1],
                            "position": (pos_x, pos_y),
                        }

                    # Break if the product cannot be placed
                    if not placed:
                        break

        return None

    def _find_position(self, stock, size):
        stock_height, stock_width = stock.shape
        for i in range(stock_height - size[0] + 1):
            for j in range(stock_width - size[1] + 1):
                if self._can_place_(stock, (i, j), size):
                    return i, j
        return None, None