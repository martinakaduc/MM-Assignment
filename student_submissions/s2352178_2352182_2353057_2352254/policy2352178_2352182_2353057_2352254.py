from policy import Policy
import numpy as np


class Policy2352178_2352182_2353057_2352254(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        pass

    def _get_stock_area_(self, stock):
        """Calculate the total initial usable area of the stock"""
        w, h = self._get_stock_size_(stock)
        return w * h

    def _sort_stocks(self, stocks):
        """Sort stocks by area in decreasing order"""
        return sorted(enumerate(stocks),
                      key=lambda x: self._get_stock_area_(x[1]),
                      reverse=True)

    def _sort_products(self, products):
        """Sort products by area in decreasing order"""
        valid_products = [p for p in products if p["quantity"] > 0]
        return sorted(valid_products,
                      key=lambda p: p["size"][0] * p["size"][1],
                      reverse=True)

    def _l1_l2_(self, a1, a2, b1, b2):
        l1 = (b2 // a2) * (b1 // a1) + ((b2 % a2) // a1) * (b1 // a2)
        l2 = (b1 // a2) * (b2 // a1) + ((b1 % a2) // a1) * (b2 // a2)
        if l1 >= l2:
            return False, True
        else:
            return True, False

    def _find_bottom_left_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w < prod_w or stock_h < prod_h:
            return None, None

        # Pre-calculate ranges for better performance
        x_range = stock_w - prod_w + 1
        y_range = range(stock_h - prod_h, -1, -1)  # Bottom to top

        # Vectorized check for each row
        for y in y_range:
            # Quick row check - skip if no valid positions possible
            if np.sum(stock[:, y:y + prod_h] == -1) < prod_w * prod_h:
                continue

            # Use vectorized operations to find valid x positions
            x_positions = np.where(stock[:x_range, y] == -1)[0]

            for x in x_positions:
                # Direct numpy check for the entire rectangle
                if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                    return x, y

        return None, None

    def get_action_first_fit(self, observation, info):
        sorted_stocks = self._sort_stocks(observation["stocks"])
        products = self._sort_products(observation["products"])

        if not products:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        for prod in products:
            if prod["quantity"] == 0:
                continue
            original_size = prod["size"]
            rotated_size = original_size[::-1]  # Swap width and height

            # Iterate through sorted stocks
            for stock_idx, stock in sorted_stocks:
                stock_w, stock_h = self._get_stock_size_(stock)

                # Compute l1 and l2 to choose best orientation
                a1, a2 = original_size
                b1, b2 = stock_w, stock_h

                # l1: original orientation
                l1 = (b2 // a2) * (b1 // a1) + ((b2 % a2) // a1) * (b1 // a2)
                # l2: rotated orientation
                l2 = (b1 // a2) * (b2 // a1) + ((b1 % a2) // a1) * (b2 // a2)

                if l1 >= l2:
                    # Try placing in original orientation first
                    size_list = [original_size, rotated_size]
                else:
                    # Try placing in rotated orientation first
                    size_list = [rotated_size, original_size]

                # Try both orientations, starting with the best one
                for size in size_list:
                    prod_w, prod_h = size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    pos_x, pos_y = self._find_bottom_left_position(stock, size)
                    if pos_x is not None:
                        # Place the product in this stock
                        self.current_stock = stock_idx
                        return {
                            "stock_idx": stock_idx,
                            "size": size,
                            "position": (pos_x, pos_y),
                        }
        # If no valid placement found, return a dummy action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    ##############################
    #
    #
    # Skyline method
    #
    #
    ##############################

    def _get_skyline_(self, stock, stock_w, stock_h):
        skyline = np.zeros(stock_w, dtype=int)
        for x in range(stock_w):
            column = stock[x, :stock_h]
            filled_indices = np.where(column != -1)[0]
            if filled_indices.size == 0:
                skyline[x] = 0
            else:
                skyline[x] = filled_indices.max() + 1
        return skyline

    def _find_position_in_stock_(self, stock, skyline, product_size, stock_w, stock_h):
        for rotate in self._l1_l2_(*product_size, stock_w, stock_h):
            prod_w, prod_h = (
                (product_size[1], product_size[0]) if rotate else product_size
            )

            if prod_w > stock_w or prod_h > stock_h:
                continue  # Skip if product doesn't fit in the stock dimensions

            # Slide the product along the skyline to find a valid position
            for x in range(stock_w - prod_w + 1):
                # Get the maximum height in the skyline over the product width
                max_height = np.max(skyline[x: x + prod_w])
                y = max_height
                if y + prod_h > stock_h:
                    continue  # Product exceeds stock boundaries vertically

                position = (x, y)
                if self._can_place_(stock, position, (prod_w, prod_h)):
                    return position, rotate  # Found a valid position

        return None, None  # No valid position found

    def get_action_skyline(self, observation, info):
        # Student code here
        products = observation["products"]
        stocks = observation["stocks"]

        # Get available products (products with quantity > 0)
        available_products = [
            p for p in products if p["quantity"] > 0
        ]

        if not available_products:
            # No products left to place
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        available_products.sort(
            key=lambda p: p["size"][0] * p["size"][1], reverse=True
        )

        # Try to place each product using the skyline method
        for product in available_products:
            product_size = product["size"]

            # Try to place the product in each stock
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                if stock_w == 0 or stock_h == 0:
                    continue  # Skip empty stocks

                # Compute the skyline for the current stock
                skyline = self._get_skyline_(stock, stock_w, stock_h)

                # Try to find a position in the stock to place the product
                position, rotation = self._find_position_in_stock_(
                    stock, skyline, product_size, stock_w, stock_h
                )

                if position is not None:
                    # Found a valid position to place the product
                    action = {
                        "stock_idx": stock_idx,
                        "size": product_size[::-1] if rotation else product_size,
                        "position": np.array(position),
                    }
                    return action

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.get_action_first_fit(observation, info)
        else:
            return self.get_action_skyline(observation, info)
