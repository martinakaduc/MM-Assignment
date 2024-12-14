import numpy as np
from policy import Policy

class Policy2213486_2313771_2313334_2212896_2313431(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.first_fit_decreasing(observation)
        elif self.policy_id == 2:
            return self.best_fit_decreasing_height(observation)

    def first_fit_decreasing(self, observation):
        """
        Implements the First Fit Decreasing algorithm.
        """	
        products = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],  # Sort by area (width * height)
            reverse=True
        )

        stocks = observation["stocks"]

        for product in products:
            prod_width, prod_height = product["size"]

            for stock_idx, stock in enumerate(stocks):
                stock_width, stock_height = self._get_stock_size_(stock)

                for x in range(stock_width - prod_width + 1):
                    for y in range(stock_height - prod_height + 1):
                        if self._can_place_(stock, (x, y), (prod_width, prod_height)):
                            return {
                                "stock_idx": stock_idx,
                                "size": (prod_width, prod_height),
                                "position": (x, y)
                            }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def best_fit_decreasing_height(self, observation):
        """
        Optimized implementation of the Best Fit Decreasing Height algorithm.
        """
        products = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],  # Sort by area
            reverse=True
        )

        stocks = observation["stocks"]
        stock_sizes = [self._get_stock_size_(stock) for stock in stocks]
        stock_free_areas = self.precompute_free_areas(stocks)

        for product in products:
            product_width, product_height = product["size"]
            product_area = product_width * product_height

            best_stock_idx = -1
            best_position = None
            best_orientation = None
            min_remaining_area = float('inf')

            for stock_idx, (stock, (stock_w, stock_h), free_area) in enumerate(zip(stocks, stock_sizes, stock_free_areas)):
                if free_area < product_area or not self.fits_in_stock((stock_w, stock_h), (product_width, product_height)):
                    continue

                for width, height in [(product_width, product_height), (product_height, product_width)]:
                    positions = self.get_available_positions(stock, width, height)
                    position, remaining_area = self.get_best_fit(stock, positions, width * height, free_area)

                    if position and remaining_area < min_remaining_area:
                        best_stock_idx = stock_idx
                        best_position = position
                        best_orientation = (width, height)
                        min_remaining_area = remaining_area

                    if min_remaining_area == 0:
                        break
                if min_remaining_area == 0:
                    break

            if best_stock_idx != -1:
                stock_free_areas[best_stock_idx] -= best_orientation[0] * best_orientation[1]
                return {
                    "stock_idx": best_stock_idx,
                    "size": best_orientation,
                    "position": best_position
                }

        return {
            "stock_idx": len(stocks),
            "size": products[0]["size"],
            "position": (0, 0)
        }

    def _get_stock_size_(self, stock):
        """
        Returns the dimensions of the stock (width, height).
        """
        return stock.shape[1], stock.shape[0]  # width, height

    def _can_place_(self, stock, position, size):
        """
        Check if the product can be placed at the given position in the stock.
        """
        x, y = position
        width, height = size

        # Check if the area is within bounds and empty (-1 means empty space)
        return np.all(stock[x:x + width, y:y + height] == -1)

    def _calculate_free_area(self, stock):
        """
        Calculates the free area in the stock, ignoring any gaps that may have been created
        by previous placements.
        """
        return np.sum(stock == -1)

    def precompute_free_areas(self, stocks):
        """
        Precomputes and returns a list of free areas for each stock.
        """
        return [self._calculate_free_area(stock) for stock in stocks]

    def fits_in_stock(self, stock_size, product_size):
        """
        Quickly checks if the product can fit in the stock dimensions.
        """
        stock_w, stock_h = stock_size
        width, height = product_size
        return (width <= stock_w and height <= stock_h) or (height <= stock_w and width <= stock_h)

    def get_available_positions(self, stock, width, height):
        """
        Returns a list of available positions in the stock for the given product dimensions.
        """
        stock_w, stock_h = stock.shape
        empty_mask = (stock == -1)  # -1 represents empty cells
        positions = []

        for y in range(stock_h - height + 1):
            for x in range(stock_w - width + 1):
                if np.all(empty_mask[x:x + width, y:y + height]):
                    positions.append((x, y))

        return positions

    def get_best_fit(self, stock, positions, product_area, free_area):
        """
        Returns the best position in the stock based on the given positions.
        """
        best_position = None
        min_remaining_area = float('inf')

        for x, y in positions:
            remaining_area = free_area - product_area
            if remaining_area < min_remaining_area:
                best_position = (x, y)
                min_remaining_area = remaining_area

                if min_remaining_area == 0:  # Perfect fit found
                    break

        return best_position, min_remaining_area

    # Student code here
    # You can add more functions if needed
