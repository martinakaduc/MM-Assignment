import numpy as np
from policy import Policy

class FFDHPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        # Sort products by decreasing height (or largest dimension)
        products = sorted(observation["products"], key=lambda p: max(p["size"]), reverse=True)

        stock_idx = -1
        pos_x, pos_y = None, None
        chosen_product = None

        for product in products:
            if product["quantity"] > 0:
                product_size = product["size"]

                # Iterate over all stocks to find the first fit
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = product_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        # Check for placement
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), product_size):
                                    pos_x, pos_y = x, y
                                    stock_idx = i
                                    chosen_product = product_size
                                    break
                            if pos_x is not None:
                                break

                    if pos_x is not None:
                        break

                if pos_x is not None:
                    break

        if pos_x is not None and pos_y is not None:
            self._update_product_quantity(observation, chosen_product)
            return {
                "stock_idx": stock_idx,
                "size": chosen_product,
                "position": (pos_x, pos_y),
            }
        else:
            print("FFDHPolicy: No valid placement found.")
            return None


class BFDHPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        # Sort products by decreasing height (or largest dimension)
        products = sorted(observation["products"], key=lambda p: max(p["size"]), reverse=True)

        stock_idx = -1
        pos_x, pos_y = None, None
        chosen_product = None
        min_waste = float('inf')  # Track minimum waste

        for product in products:
            if product["quantity"] > 0:
                product_size = product["size"]
                prod_w, prod_h = product_size

                # Iterate over all stocks to find the best fit
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w >= prod_w and stock_h >= prod_h:
                        # Check for placement
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), product_size):
                                    # Calculate waste (unused area after placing the product)
                                    waste = self._calculate_waste(stock, (x, y), product_size)
                                    if waste < min_waste:
                                        min_waste = waste
                                        pos_x, pos_y = x, y
                                        stock_idx = i
                                        chosen_product = product_size

                    if pos_x is not None:
                        break
                if pos_x is not None:
                    break

        if pos_x is not None and pos_y is not None:
            self._update_product_quantity(observation, chosen_product)
            return {
                "stock_idx": stock_idx,
                "size": chosen_product,
                "position": (pos_x, pos_y),
            }
        else:
            print("BFDHPolicy: No valid placement found.")
            return None

    def _calculate_waste(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        # Calculate the remaining unused space (waste) after placing the product
        used_area = np.sum(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] != -1)
        total_area = stock.size
        remaining_space = total_area - used_area
        return remaining_space

