import random
import numpy as np
from policy import Policy

class Policy2211221_2252118_2252359_2252614_2252190(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.bfd_get_action(observation, info)
        if self.policy_id == 2:
            return self.hsa_get_action(observation, info)

    def bfd_get_action(self, observation, info):
        """Best-Fit Decreasing heuristic for placing items."""
        stocks = observation["stocks"]
        products = observation["products"]

        # Sort products by descending area
        sorted_products = sorted(products, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        for product in sorted_products:
            size = product["size"]
            prod_w, prod_h = size
            if product["quantity"] == 0:
                continue

            best_fit = None
            best_waste = float('inf')  # Minimum waste area

            # Search for the best fit stock
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_dimensions(stock)

                for rotated, (p_w, p_h) in enumerate([(prod_w, prod_h), (prod_h, prod_w)]):
                    if stock_w >= p_w and stock_h >= p_h:
                        for x in range(stock_w - p_w + 1):
                            for y in range(stock_h - p_h + 1):
                                if self._can_place(stock, (x, y), (p_w, p_h)):
                                    waste = (stock_w * stock_h) - (p_w * p_h)
                                    if waste < best_waste:
                                        best_waste = waste
                                        best_fit = {
                                            "stock_idx": stock_idx,
                                            "size": (p_w, p_h),
                                            "position": (x, y),
                                        }

            if best_fit:
                return best_fit

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def hsa_get_action(self, observation, info):
        """Heuristic Shelf Allocation for improved placement efficiency."""
        stocks = observation["stocks"]
        products = observation["products"]

        # Sort products by descending area
        sorted_products = sorted(products, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        # Maintain shelves for each stock
        shelves = [ [] for _ in range(len(stocks)) ]

        for product in sorted_products:
            size = product["size"]
            prod_w, prod_h = size
            if product["quantity"] == 0:
                continue

            best_fit = None
            best_waste = float('inf')

            # Iterate over stocks
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_dimensions(stock)
                
                # Check existing shelves in the stock
                for shelf_idx, shelf in enumerate(shelves[stock_idx]):
                    shelf_y, shelf_h = shelf  # Shelf position and height
                    shelf_w = stock_w

                    # Check both orientations of the product
                    for rotated, (p_w, p_h) in enumerate([(prod_w, prod_h), (prod_h, prod_w)]):
                        if p_h <= shelf_h and p_w <= shelf_w:  # Product fits in the shelf
                            for x in range(stock_w - p_w + 1):
                                if self._can_place(stock, (x, shelf_y), (p_w, p_h)):
                                    waste = (shelf_h * stock_w) - (p_w * p_h)
                                    if waste < best_waste:
                                        best_waste = waste
                                        best_fit = {
                                            "stock_idx": stock_idx,
                                            "shelf_idx": shelf_idx,
                                            "size": (p_w, p_h),
                                            "position": (x, shelf_y),
                                        }
                
                # Check entire stock area if no suitable shelf found
                if not best_fit:
                    for rotated, (p_w, p_h) in enumerate([(prod_w, prod_h), (prod_h, prod_w)]):
                        if stock_w >= p_w and stock_h >= p_h:
                            for x in range(stock_w - p_w + 1):
                                for y in range(stock_h - p_h + 1):
                                    if self._can_place(stock, (x, y), (p_w, p_h)):
                                        waste = (stock_w * stock_h) - (p_w * p_h)
                                        if waste < best_waste:
                                            best_waste = waste
                                            best_fit = {
                                                "stock_idx": stock_idx,
                                                "shelf_idx": len(shelves[stock_idx]),
                                                "size": (p_w, p_h),
                                                "position": (x, y),
                                            }

            if best_fit:
                # Update shelves if necessary
                stock_idx = best_fit["stock_idx"]
                shelf_idx = best_fit["shelf_idx"]
                if shelf_idx == len(shelves[stock_idx]):
                    # Add new shelf
                    shelves[stock_idx].append((best_fit["position"][1], best_fit["size"][1]))
                return best_fit

        # Return failure if no valid position found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _get_stock_dimensions(self, stock):
        """Retrieve the usable dimensions of the stock."""
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place(self, stock, position, size):
        """Check if the product can be placed at the given position in the stock."""
        x, y = position
        w, h = size
        return np.all(stock[x:x + w, y:y + h] == -1)