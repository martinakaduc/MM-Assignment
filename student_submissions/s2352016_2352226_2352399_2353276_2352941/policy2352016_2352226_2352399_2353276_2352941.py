from policy import Policy
import numpy as np


class Policy2352016_2352226_2352399_2353276_2352941(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        self.policy = 0
        # Student code here
        if policy_id == 1:
            self.policy = 1
        elif policy_id == 2:
            self.policy = 2

        self.stocks = None
        self.initialize = True

        # Policy 2
        self.sort_count = 0  # Counter to track calls to `get_action2`
        self.sorted_stocks = []  # Cached sorted stocks
        self.cached_available_spaces = []  # Cached available spaces
        
    def get_action(self, observation, info):
        if self.policy == 1:
            return self.get_action1(observation, info)
        elif self.policy == 2:
            return self.get_action2(observation, info)

    

    def get_action1(self, observation, info):
        # Extract stocks and products
        stocks = observation["stocks"]
        products = observation["products"]

        # Sort products by area in descending order (largest first)
        sorted_prods = sorted(
            enumerate(products), 
            key=lambda p: p[1]["size"][0] * p[1]["size"][1], 
            reverse=True
        )

        best_stock_idx = -1
        best_prod_idx = -1
        best_prod_size = None
        best_prod_position = None
        min_waste = float("inf")

        # Traverse sorted products to find the best placement
        for prod_idx, prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_w, prod_h = prod["size"]

                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    waste = self._calculate_available_space_(stock) - prod_w * prod_h

                    # Check normal orientation
                    pos_x, pos_y = self._find_position(stock, (prod_w, prod_h))
                    if pos_x is not None and pos_y is not None:
                        # waste = (stock_w * stock_h) - (prod_w * prod_h)
                        if waste < min_waste:
                            best_stock_idx = stock_idx
                            best_prod_idx = prod_idx
                            best_prod_size = (prod_w, prod_h)
                            best_prod_position = (pos_x, pos_y)
                            min_waste = waste

                    # Check rotated orientation
                    pos_x, pos_y = self._find_position(stock, (prod_h, prod_w))
                    if pos_x is not None and pos_y is not None:
                        # waste = (stock_w * stock_h) - (prod_h * prod_w)
                        if waste < min_waste:
                            best_stock_idx = stock_idx
                            best_prod_idx = prod_idx
                            best_prod_size = (prod_h, prod_w)
                            best_prod_position = (pos_x, pos_y)
                            min_waste = waste

                if best_stock_idx != -1:
                    break  # Break if a valid position is found for the largest product

        # If a valid placement is found, return the action
        if best_stock_idx != -1:
            return {
                "stock_idx": best_stock_idx,
                "size": best_prod_size,
                "position": best_prod_position
            }

        # If no placement is possible, return default action
        return {"stock_idx": -1, "size": (0, 0), "position": (None, None)}

    def get_action2(self, observation, info):
        if info.get("filled_ratio") == 0.0:
            self.initialize = True

        if self.initialize == True:
            self.stocks = np.copy(observation["stocks"])
            self.initialize = False
        # Extract stocks and products
        stocks = self.stocks
        products = observation["products"]

        # Sort stocks by area (smallest first)
        sorted_stocks = sorted(
            enumerate(stocks),
            key=lambda s: self._calculate_available_space_(s[1])
        )

        # Sort products by area in descending order (largest first)
        sorted_prods = sorted(
            enumerate(products), 
            key=lambda p: p[1]["size"][0] * p[1]["size"][1], 
            reverse=True
        )

        # Traverse sorted stocks to find the best placement
        for stock_idx, stock in sorted_stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            best_prod_idx = -1
            best_prod_size = None
            best_prod_position = None
            min_waste = float("inf")  # Initialize waste with a large value

            # Traverse sorted products to find the best product for this stock
            for prod_idx, prod in sorted_prods:
                if prod["quantity"] > 0:  # Skip products with no quantity left
                    prod_w, prod_h = prod["size"]
                    waste = self._calculate_available_space_(stock) - prod_w * prod_h

                    # Check normal orientation
                    if stock_w >= prod_w and stock_h >= prod_h and waste >= 0:
                        # waste = (stock_w * stock_h) - (prod_w * prod_h)
                        pos_x, pos_y = self._find_position(stock, prod["size"])
                        if pos_x is not None and pos_y is not None and waste < min_waste:
                            best_prod_idx = prod_idx
                            best_prod_size = prod["size"]
                            best_prod_position = (pos_x, pos_y)
                            min_waste = waste

                    # Check rotated orientation
                    if stock_w >= prod_h and stock_h >= prod_w:
                        # waste = (stock_w * stock_h) - (prod_h * prod_w)
                        pos_x, pos_y = self._find_position(stock, prod["size"][::-1])
                        if pos_x is not None and pos_y is not None and waste < min_waste:
                            best_prod_idx = prod_idx
                            best_prod_size = prod["size"][::-1]
                            best_prod_position = (pos_x, pos_y)
                            min_waste = waste

            # If a valid product is found, return the action
            if best_prod_idx != -1 and best_prod_position is not None:
                # print(stock_idx, best_prod_size, best_prod_position)
                # print(stock)
                self._mark_filled_cells_(stock, best_prod_position, best_prod_size, best_prod_idx)
                return {
                    "stock_idx": stock_idx,
                    "size": best_prod_size,
                    "position": best_prod_position
                }

        # If no placement is possible, return default action
        return {"stock_idx": -1, "size": (0, 0), "position": (None, None)}

    def get_action3(self, observation, info):
        # Extract stocks and products
        stocks = observation["stocks"]
        products = observation["products"]

        # Precompute product areas and filter products with quantity > 0
        valid_products = [
            (idx, prod, prod["size"][0] * prod["size"][1])
            for idx, prod in enumerate(products) if prod["quantity"] > 0
        ]

        # Cache stock sizes and available spaces
        available_spaces = [self._calculate_available_space_(stock) for stock in stocks]

        # Maintain a cache for previously checked positions
        position_cache = {}

        best_stock_idx = -1
        best_prod_idx = -1
        best_prod_size = None
        best_prod_position = None
        max_utilization = 0

        # Traverse valid products and stocks to find the best placement
        for prod_idx, prod, prod_area in valid_products:
            prod_w, prod_h = prod["size"]

            for stock_idx, (stock, available_space) in enumerate(zip(stocks, available_spaces)):
                # Skip stocks with less available space than product area
                if available_space < prod_area:
                    continue

                # Check both normal and rotated orientations
                for orientation in [(prod_w, prod_h), (prod_h, prod_w)]:
                    cache_key = (stock_idx, orientation)
                    if cache_key not in position_cache:
                        pos_x, pos_y = self._find_position(stock, orientation)
                        position_cache[cache_key] = (pos_x, pos_y)
                    else:
                        pos_x, pos_y = position_cache[cache_key]

                    if pos_x is not None and pos_y is not None:
                        utilization = prod_area / available_space
                        if utilization > max_utilization:
                            best_stock_idx = stock_idx
                            best_prod_idx = prod_idx
                            best_prod_size = orientation
                            best_prod_position = (pos_x, pos_y)
                            max_utilization = utilization

                        # Early exit for the best possible utilization
                        if utilization == 1:
                            break

                # Stop checking further stocks if the best utilization is achieved
                if max_utilization == 1:
                    break

        # If a valid placement is found, return the action
        if best_stock_idx != -1:
            # Update available space dynamically
            used_space = best_prod_size[0] * best_prod_size[1]
            available_spaces[best_stock_idx] -= used_space

            return {
                "stock_idx": best_stock_idx,
                "size": best_prod_size,
                "position": best_prod_position
            }

        # If no placement is possible, return default action
        return {"stock_idx": -1, "size": (0, 0), "position": (None, None)}




    
    def _find_position(self, stock, prod_size):
        """
        Find the first available position in the stock for the given product size.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None, None
    
    def _calculate_available_space_(self, stock):
        available_space = np.sum(stock == -1)
        return available_space
    
    def _mark_filled_cells_(self, stock, position, prod_size, prod_idx):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        # Mark the cells with the product index
        stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] = prod_idx



    # Student code here
    # You can add more functions if needed
