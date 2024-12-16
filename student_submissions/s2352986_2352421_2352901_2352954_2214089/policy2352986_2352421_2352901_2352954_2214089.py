from policy import Policy
import numpy as np


class Policy2352986_2352421_2352901_2352954_2214089(Policy):  
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        if policy_id == 1:
            self.policy_function = self.split_fit
        elif policy_id == 2:
            self.policy_function = self.gurel_algorithm

    def get_action(self, observation, info):    
        action = self.policy_function(observation, info) 
        return action

    def split_fit(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # Group products based on width vs height
        group_1 = [prod for prod in list_prods if prod["quantity"] > 0 and prod["size"][0] > prod["size"][1]]  
        group_2 = [prod for prod in list_prods if prod["quantity"] > 0 and prod["size"][0] <= prod["size"][1]] 

        # Sort products in each group by width and then height (descending order)
        group_1 = sorted(group_1, key=lambda p: (p["size"][0], p["size"][1]), reverse=True)
        group_2 = sorted(group_2, key=lambda p: (p["size"][0], p["size"][1]), reverse=True)

        # Process groups sequentially
        for group in [group_1, group_2]:
            for prod in group:
                prod_size = prod["size"]

                # Try to place this product in one of the stocks
                for i, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Check placement without rotation
                    placement = self._find_placement(stock, prod_size, stock_w, stock_h)
                    if placement is not None:
                        pos_x, pos_y = placement
                        return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}

                    # Check placement with rotation
                    placement = self._find_placement(stock, prod_size[::-1], stock_w, stock_h)
                    if placement is not None:
                        pos_x, pos_y = placement
                        return {"stock_idx": i, "size": prod_size[::-1], "position": (pos_x, pos_y)}

        return None  

    def gurel_algorithm(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # Sort products by area 
        sorted_prods = sorted(
            [prod for prod in list_prods if prod["quantity"] > 0],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )

        # Iterate through products
        for prod in sorted_prods:
            prod_size = prod["size"]

            # Check stock for placement
            for i, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size

                # Check placement without rotation
                placement = self._find_placement(stock, prod_size, stock_w, stock_h)
                if placement is not None:
                    pos_x, pos_y = placement
                    return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}

                # Check placement with rotation
                placement = self._find_placement(stock, prod_size[::-1], stock_w, stock_h)
                if placement is not None:
                    pos_x, pos_y = placement
                    return {"stock_idx": i, "size": prod_size[::-1], "position": (pos_x, pos_y)}

        return None  

    def _find_placement(self, stock, prod_size, stock_w, stock_h):
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None

