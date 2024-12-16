from policy import Policy
import numpy as np


class Policy2352600_2352726_2352621_2352972_2352522(Policy):
    def __init__(self, policy_id=1):
        """
        Initialize the policy with the specified heuristic method.
        :param policy_id: 1 for Combined Heuristics, 2 for First-Fit Decreasing (FFD)
        """
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        """
        Decide the action to take based on the selected heuristic.
        :param observation: Contains information about stocks and products.
        :param info: Additional information (if any).
        :return: A dictionary with the action details.
        """
        products = observation["products"]
        stocks = observation["stocks"]

        # Check if there are products to place
        if not any(prod["quantity"] > 0 for prod in products):
            return {"new_sheet": True}

        # Sort products by area in descending order
        sorted_products = sorted(
            [prod for prod in products if prod["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],
            reverse=True,
        )

        if self.policy_id == 1:
            # Test all heuristics combined
            return self.combined_heuristic(sorted_products, stocks)
        elif self.policy_id == 2:
            # Test only FFD
            return self.first_fit_decreasing(sorted_products, stocks)

    def combined_heuristic(self, products, stocks):
        """
        Combine multiple heuristics to improve placement efficiency.
        """
        for prod in products:
            prod_size = prod["size"]

            # 1. Try First-Fit Decreasing
            ffd_placement = self.first_fit_decreasing([prod], stocks)
            if ffd_placement:
                return ffd_placement

            # 2. Try Best-Fit Decreasing
            bfd_placement = self.best_fit_decreasing(prod["size"], stocks)
            if bfd_placement:
                return bfd_placement

            # 3. Try Gap Reuse
            gap_placement = self.gap_reuse_heuristic(prod["size"], stocks)
            if gap_placement:
                return gap_placement

        # If no placement found, open a new stock
        return {"new_sheet": True}

    def first_fit_decreasing(self, products, stocks):
        """
        First-Fit Decreasing heuristic for placing products into stocks.
        :param products: Sorted list of products by area in descending order.
        :param stocks: List of available stocks.
        :return: A dictionary with the action details.
        """
        for prod in products:
            prod_size = prod["size"]
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Try both orientations
                for rotation in [prod_size, prod_size[::-1]]:
                    for x in range(stock_w - rotation[0] + 1):
                        for y in range(stock_h - rotation[1] + 1):
                            if self._can_place_(stock, (x, y), rotation):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": rotation,
                                    "position": (x, y),
                                }
        return None

    def best_fit_decreasing(self, prod_size, stocks):
        """
        Best-Fit Decreasing heuristic for a single product.
        """
        best_fit = None
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Try both orientations
            for rotation in [prod_size, prod_size[::-1]]:
                for x in range(stock_w - rotation[0] + 1):
                    for y in range(stock_h - rotation[1] + 1):
                        if self._can_place_(stock, (x, y), rotation):
                            gap_area = self._calculate_gap_area(stock, (x, y), rotation)
                            if best_fit is None or gap_area < best_fit["gap_area"]:
                                best_fit = {
                                    "stock_idx": stock_idx,
                                    "size": rotation,
                                    "position": (x, y),
                                    "gap_area": gap_area,
                                }

        if best_fit:
            return {
                "stock_idx": best_fit["stock_idx"],
                "size": best_fit["size"],
                "position": best_fit["position"],
            }
        return None

    def gap_reuse_heuristic(self, prod_size, stocks):
        """
        Improved Gap Reuse heuristic to effectively utilize small gaps.
        """
        best_gap = None
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)

            for rotation in [prod_size, prod_size[::-1]]:
                for x in range(stock_w - rotation[0] + 1):
                    for y in range(stock_h - rotation[1] + 1):
                        if self._can_place_(stock, (x, y), rotation):
                            remaining_gap = self._calculate_gap_area(stock, (x, y), rotation)
                            if best_gap is None or remaining_gap < best_gap["remaining_gap"]:
                                best_gap = {
                                    "stock_idx": stock_idx,
                                    "size": rotation,
                                    "position": (x, y),
                                    "remaining_gap": remaining_gap,
                                }

        if best_gap:
            return {
                "stock_idx": best_gap["stock_idx"],
                "size": best_gap["size"],
                "position": best_gap["position"],
            }
        return None

    def _calculate_gap_area(self, stock, position, prod_size):
        """
        Calculate the gap area left after placing a product.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        x, y = position
        prod_w, prod_h = prod_size

        # Remaining space on the right and bottom
        right_gap = stock_w - (x + prod_w)
        bottom_gap = stock_h - (y + prod_h)

        # Total unused area around the product
        return right_gap * bottom_gap

    def _get_stock_size_(self, stock):
        """
        Extract the dimensions of a stock based on its non-negative regions.
        """
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, size):
        """
        Check if a product can be placed at a specific position in the stock.
        """
        x, y = position
        prod_w, prod_h = size
        return np.all(stock[x : x + prod_w, y : y + prod_h] == -1)
