# from policy import Policy


# class Policy2210xxx(Policy):
#     def __init__(self, policy_id=1):
#         assert policy_id in [1, 2], "Policy ID must be 1 or 2"

#         # Student code here
#         if policy_id == 1:
#             pass
#         elif policy_id == 2:
#             pass

#     def get_action(self, observation, info):
#         # Student code here
#         pass

#     # Student code here
#     # You can add more functions if needed
from policy import Policy
import random
import numpy as np

class Policy2352354_2352780_2352350_2352088_2352434(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Initialize the chosen policy
        self.policy_id = policy_id

        if self.policy_id == 1:
            # Algorithmic Art / Greedy Hybrid initialization
            pass
        elif self.policy_id == 2:
            # Column Generation Technique initialization
            pass

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._get_action_algorithmic_art(observation, info)
        elif self.policy_id == 2:
            return self._get_action_column_generation(observation, info)

    # Algorithm 1: Algorithmic Art / Greedy Hybrid
    def _get_action_algorithmic_art(self, observation, info):
        list_prods = observation["products"]

        # Step 1: Sort products by demand (from highest to lowest demand)
        list_prods = sorted(list_prods, key=lambda x: -x["quantity"])

        stock_idx, prod_size, pos_x, pos_y = -1, [0, 0], 0, 0

        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue  # Skip if the product is not needed

            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            # Step 2: Loop through each stock and try placing the product
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Step 3: Try to place product without rotation
                if stock_w >= prod_w and stock_h >= prod_h:
                    pos_x, pos_y = self._find_position(stock, prod_size)
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

                # Step 4: Try to place product with rotation (swap width and height)
                if stock_w >= prod_h and stock_h >= prod_w:
                    rotated_size = [prod_h, prod_w]
                    pos_x, pos_y = self._find_position(stock, rotated_size)
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        return {"stock_idx": stock_idx, "size": rotated_size, "position": (pos_x, pos_y)}

        # If no valid position found, return default action (this should ideally not happen)
        return {"stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}

    def _find_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Search for a valid position (this is where the algorithm by art comes in)
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y  # Found a valid position

        return None, None  # No valid position found

################################################################

    # Algorithm 2: Column Generation Technique
    def _get_action_column_generation(self, observation, info):
        products = sorted(
            observation["products"],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True,
        )  # Sort products by area in descending order
        stocks = observation["stocks"]

        # Step 1: Build candidate cutting patterns for each stock
        candidate_patterns = []
        for stock_idx, stock in enumerate(stocks):
            stock_patterns = self._generate_patterns(stock, products)
            candidate_patterns.append((stock_idx, stock_patterns))

        # Step 2: Select the best pattern (minimizing trim loss)
        best_stock_idx = -1
        best_pattern = None
        min_trim_loss = float("inf")

        for stock_idx, patterns in candidate_patterns:
            for pattern in patterns:
                trim_loss = self._calculate_trim_loss(stocks[stock_idx], pattern)
                if trim_loss < min_trim_loss:
                    min_trim_loss = trim_loss
                    best_stock_idx = stock_idx
                    best_pattern = pattern

        # Step 3: Return the selected action
        if best_pattern:
            return {
                "stock_idx": best_stock_idx,
                "size": best_pattern["prod_size"],
                "position": best_pattern["position"]
            }
        else:
            # No valid placement found; return default invalid action
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _generate_patterns(self, stock, products):
        """
        Generate feasible cutting patterns for a stock based on the products.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        patterns = []

        for prod in products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Add patterns for both orientations (normal and rotated)
                for orientation in [prod_size, prod_size[::-1]]:
                    self._add_patterns_if_valid(patterns, stock, stock_w, stock_h, orientation)

        return patterns

    def _add_patterns_if_valid(self, patterns, stock, stock_w, stock_h, prod_size):
        """
        Add valid patterns to the list using a more efficient approach.
        """
        prod_w, prod_h = prod_size
        if stock_w < prod_w or stock_h < prod_h:
            return

        # Use a vectorized approach to identify valid positions
        x_range = np.arange(stock_w - prod_w + 1)
        y_range = np.arange(stock_h - prod_h + 1)
        for x in x_range:
            for y in y_range:
                if self._can_place_(stock, (x, y), prod_size):
                    patterns.append({
                        "prod_size": prod_size,
                        "position": (x, y),
                    })

    def _calculate_trim_loss(self, stock, pattern):
        """
        Calculate trim loss for a stock after applying a pattern.
        """
        prod_w, prod_h = pattern["prod_size"]
        stock_w, stock_h = self._get_stock_size_(stock)

        used_area = prod_w * prod_h
        total_area = stock_w * stock_h
        trim_loss = (total_area - used_area) / total_area
        return trim_loss
