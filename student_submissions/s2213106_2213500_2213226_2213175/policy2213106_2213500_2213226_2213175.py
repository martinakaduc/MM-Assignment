from policy import Policy
import numpy as np
from scipy.optimize import linprog
import random
class Policy2213106_2213500_2213226_2213175(Policy):
    def __init__(self, policy_id=2, stocks=None, products=None):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.stocks = stocks if stocks is not None else []  # Stock data
        self.products = products if products is not None else []  # Product data


    def get_action(self, observation, info):
        if self.policy_id == 1:
            # Apply First-fit Algorithm
            action = self.Firstfitwithcalculate(observation, info)
        else:
            # Apply Dynamic Programming Algorithm
            action = self.dynamic_programming(observation, info)
        return action

    def Firstfitwithcalculate(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]

        # Sort products by area (largest first) to improve space utilization
        sorted_products = sorted(
            enumerate(products),
            key=lambda p: p[1]["size"][0] * p[1]["size"][1],
            reverse=True,
        )

        # Cache stock dimensions to avoid redundant computations
        stock_sizes = [self._get_stock_size_(stock) for stock in stocks]

        # Store the first placement as random
        placed_stocks = set()

        # Helper function to compute the combined metric
        def compute_metric(stock, position, prod_size, filled_ratio, trim_loss):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            # Remaining space after placement
            remaining_space = (stock_w * stock_h) - (prod_w * prod_h)

            # New filled ratio after placement
            new_filled_ratio = filled_ratio + (prod_w * prod_h) / (stock_w * stock_h)

            # Weighted metric for optimization
            return remaining_space + (1 - new_filled_ratio) + trim_loss

        # Try placing each product
        for prod_idx, product in sorted_products:
            prod_size = product["size"]
            quantity = product["quantity"]

            if quantity <= 0:
                continue  # Skip products with no quantity

            prod_width, prod_height = prod_size

            # Random placement for the first stock
            if len(placed_stocks) == 0:
                random_stock_idx = random.randint(0, len(stocks) - 1)
                stock = stocks[random_stock_idx]
                stock_width, stock_height = stock_sizes[random_stock_idx]

                for x in range(stock_width - prod_width + 1):
                    for y in range(stock_height - prod_height + 1):
                        position = (x, y)
                        if self._can_place_(stock, position, prod_size):
                            placed_stocks.add(random_stock_idx)
                            return {
                                "stock_idx": random_stock_idx,
                                "size": prod_size,
                                "position": position,
                            }

            # Metric-based placement for subsequent stocks
            best_action = None
            best_metric = float("inf")  # Combination of metrics to minimize

            for stock_idx, stock in enumerate(stocks):
                if stock_idx in placed_stocks:
                    continue  # Skip already placed stocks

                stock_width, stock_height = stock_sizes[stock_idx]

                # Skip stocks that are too small for this product
                if prod_width > stock_width or prod_height > stock_height:
                    continue

                # Try all valid positions in the stock
                for x in range(stock_width - prod_width + 1):
                    for y in range(stock_height - prod_height + 1):
                        position = (x, y)

                        # Check if the product can be placed at this position
                        if self._can_place_(stock, position, prod_size):
                            metric = compute_metric(
                                stock,
                                position,
                                prod_size,
                                info.get("filled_ratio", 0),
                                info.get("trim_loss", 1),
                            )

                            # Update the best action
                            if metric < best_metric:
                                best_metric = metric
                                best_action = {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": position,
                                }

            # If a valid placement is found, mark the stock as used and return the action
            if best_action:
                placed_stocks.add(best_action["stock_idx"])
                return best_action

        # If no valid placement is found, return an invalid action
        return {
            "stock_idx": -1,
            "size": (0, 0),
            "position": (0, 0),
        }







        
    def dynamic_programming(self,observation,info):
        list_prods = observation["products"]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Sort products by size for better table management
        sorted_prods = sorted(list_prods, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)

        # DP Table to store best fit solutions for products
        dp = {}

        # Helper function to compute the maximum remaining space for a given stock
        def max_remaining_space(stock, product_size):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = product_size

            if stock_w < prod_w or stock_h < prod_h:
                return None  # Product doesn't fit in the stock

            # Let's assume we only calculate the best fitting positions for the product
            best_remaining_space = stock_w * stock_h - (prod_w * prod_h)  # Simplified assumption (can be improved)
            return best_remaining_space

        # Try placing the products in the most efficient stock configuration using DP
        def dp_solver(stock_idx, remaining_space, prod_size):
            if (stock_idx, remaining_space) in dp:
                return dp[(stock_idx, remaining_space)]

            # Base case: No remaining space, placement impossible
            if remaining_space < 0:
                return None

            best_position = None
            stock = observation["stocks"][stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            # Try all possible positions in the stock
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), prod_size):
                        # Recursively calculate the remaining space after placement
                        remaining = max_remaining_space(stock, prod_size)
                        best_position = (x, y) if remaining is not None else None
                        if best_position:
                            dp[(stock_idx, remaining_space)] = best_position
                            return best_position

            return None

        # Loop through all sorted products
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Try placing the product in the best fitting stock using DP
                for i, stock in enumerate(observation["stocks"]):
                    remaining_space = self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]
                    pos = dp_solver(i, remaining_space, prod_size)

                    if pos is not None:
                        pos_x, pos_y = pos
                        stock_idx = i
                        break

                # If a valid placement is found, break the loop
                if stock_idx != -1:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
