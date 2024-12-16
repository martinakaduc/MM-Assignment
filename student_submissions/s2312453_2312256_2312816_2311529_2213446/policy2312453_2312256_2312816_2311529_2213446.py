from policy import Policy
import numpy as np
      
   
class Policy2312453_2312256_2312816_2311529_2213446(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2, 3], "Policy ID must be 1 or 2"
        # Student code here
        if policy_id == 1:
            self.policy_id = policy_id
            pass
        elif policy_id == 2:
            self.policy_id = policy_id
            pass
        elif policy_id == 3:
            self.policy_id = policy_id
            pass

    def get_action(self, observation, info):

        if self.policy_id == 1:
            list_prods = observation["products"]  # List of products
            best_fit = None
            best_stock_idx = -1
            best_position = None
            best_size = None

            sorted_products = sorted(
            (prod for prod in list_prods if prod["quantity"] > 0),
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True,
            )

            # Loop through products and find the best fit
            for prod in sorted_products:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    stock_idx = -1
                    pos_x, pos_y = None, None
                    best_unused_area = float('inf')  # To track the best fit (minimize unused area)

                    # Loop through all stocks to find the best position for the product
                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)  # Get stock dimensions
                        prod_w, prod_h = prod_size

                        # Check the stock without rotation
                        if stock_w >= prod_w and stock_h >= prod_h:
                            position = self._find_best_position(stock, prod_size)
                            if position:
                                unused_area = (stock_w * stock_h) - (prod_w * prod_h)
                                if unused_area < best_unused_area:
                                    best_unused_area = unused_area
                                    best_stock_idx = i
                                    best_position = position
                                    best_size = prod_size

                        if stock_w >= prod_h and stock_h >= prod_w:
                            position = self._find_best_position(stock, prod_size[::-1])
                            if position:
                                unused_area = (stock_w * stock_h) - (prod_h * prod_w)
                                if unused_area < best_unused_area:
                                    best_unused_area = unused_area
                                    best_stock_idx = i
                                    best_position = position
                                    best_size = prod_size[::-1]


                    if best_position is not None:
                        break

            return {"stock_idx": best_stock_idx, "size": best_size, "position": best_position}
        
        elif self.policy_id == 2:
            # Another heuristic: prioritize filling up the largest stock
            list_prods = observation["products"]
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0

            sorted_products = sorted(
            (prod for prod in list_prods if prod["quantity"] > 0),
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True,
            )

            for stock_idx, stock in sorted(enumerate(observation["stocks"]), key=lambda s: np.sum(s[1] != -2), reverse=True):
                stock_w, stock_h = self._get_stock_size_(stock)
                for prod in sorted_products:
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                            for x in range(stock_w - prod_size[0] + 1):
                                for y in range(stock_h - prod_size[1] + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
            # If no valid action is found
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        elif self.policy_id == 3:
            list_prods = observation["products"]  # List of products
            best_stock_idx = -1
            best_position = None
            best_size = None
            best_unused_area = float('inf')  # To track the best fit (minimize unused area)

            # Sort products by area (largest first)
            sorted_products = sorted(
                (prod for prod in list_prods if prod["quantity"] > 0),
                key=lambda p: p["size"][0] * p["size"][1],
                reverse=True,
            )

            # Filter stocks that can fit the largest product
            possible_stocks = [
                (i, stock) for i, stock in enumerate(observation["stocks"])
                if self._get_stock_size_(stock)[0] >= sorted_products[0]["size"][0] and
                self._get_stock_size_(stock)[1] >= sorted_products[0]["size"][1]
            ]

            # Sort stocks by available area (best fit) from largest to smallest
            possible_stocks.sort(
                key=lambda s: np.sum(s[1] != -2), reverse=True
            )

            for prod in sorted_products:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    stock_idx = -1
                    pos_x, pos_y = None, None

                    # Loop through all possible stocks to find the best position for the product
                    for i, stock in possible_stocks:
                        stock_w, stock_h = self._get_stock_size_(stock)  # Get stock dimensions
                        prod_w, prod_h = prod_size

                        # Check the stock without rotation
                        if stock_w >= prod_w and stock_h >= prod_h:
                            position = self._find_best_position(stock, prod_size)
                            if position:
                                unused_area = (stock_w * stock_h) - (prod_w * prod_h)
                                if unused_area < best_unused_area:
                                    best_unused_area = unused_area
                                    best_stock_idx = i
                                    best_position = position
                                    best_size = prod_size

                        # Check the stock with rotation
                        if stock_w >= prod_h and stock_h >= prod_w:
                            position = self._find_best_position(stock, prod_size[::-1])
                            if position:
                                unused_area = (stock_w * stock_h) - (prod_h * prod_w)
                                if unused_area < best_unused_area:
                                    best_unused_area = unused_area
                                    best_stock_idx = i
                                    best_position = position
                                    best_size = prod_size[::-1]

                    # If a best position has been found, break out of the product loop
                    if best_position is not None:
                        break

            return {"stock_idx": best_stock_idx, "size": best_size, "position": best_position}
    

    def _find_best_position(self, stock, prod_size):

        prod_w, prod_h = prod_size
        stock_w, stock_h = self._get_stock_size_(stock)


        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)