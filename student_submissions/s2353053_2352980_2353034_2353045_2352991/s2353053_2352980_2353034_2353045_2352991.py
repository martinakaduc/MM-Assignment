from policy import Policy
import numpy as np


class Policy2353053_2352980_2353034_2353045_2352991(Policy):
    action_id = None
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.action_id = policy_id
        # Student code here
        if policy_id == 1:
            print("Policy Heuristic")
            
        if policy_id == 2:
            print("Policy Column Generation")
            super().__init__()
            self.patterns = []  
            self.demands = []  
            self.stocks = []  
            
            
    def get_action(self, observation, info):
        if self.action_id == 1:
            return self.get_action_1(observation, info)
        elif self.action_id == 2:
            return self.get_action_2(observation, info)
        
    #HEURISTIC POLICY
    def get_action_1(self, observation, info):
        # Student code here
        products = observation["products"]
        storted_products = sorted(products, key = lambda p : p["size"][0] * p["size"][1], reverse=True)
        for product in storted_products:
            if product["quantity"] > 0:
                prod_size = product["size"]
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                        for x in range(stock_w - prod_size[0] + 1):
                            for y in range(stock_h - prod_size[1] + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)
                                    }
                                    
        return {
            "stock_idx": -1,
            "size": (0, 0),
            "position": (0, 0)
        }
    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        return np.all(stock[position[0]:position[0]+prod_size[0], position[1]:position[1]+prod_size[1]] == -1)

    
    
    #COLUMN GENERATION POLICY
    def initialize(self, observation):
        self.stocks = observation["stocks"]
        self.demands = [prod["quantity"] for prod in observation["products"]]
        product_sizes = [prod["size"] for prod in observation["products"]]

        for i, size in enumerate(product_sizes):
            self.patterns.append({
                "product_idx": i,
                "size": size,
                "quantity": 1,
                "positions": [],
                "stock_idx": -1
            })

    def get_action_2(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        sorted_prods = sorted(
            list_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True
        )

        new_pattern = None
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    best_position = None
                    min_waste = float("inf")

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                waste = self._calculate_waste(stock, (x, y), prod_size)
                                if waste < min_waste:
                                    best_position = (x, y)
                                    min_waste = waste

                    if best_position:
                        new_pattern = {
                            "product_idx": next(
                                i
                                for i, p in enumerate(list_prods)
                                if p["quantity"] == prod["quantity"]
                                and np.array_equal(p["size"], prod["size"])
                            ),
                            "size": prod_size,
                            "quantity": 1,
                            "position": best_position,  # Sử dụng 'best_position' trực tiếp
                            "stock_idx": stock_idx,
                        }
                        break

            if new_pattern:
                break

        if not new_pattern:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        self.patterns.append(new_pattern)
        return new_pattern


    def _calculate_waste(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        occupied_area = prod_w * prod_h

        total_area = stock.shape[0] * stock.shape[1]

        remaining_area = total_area - occupied_area
        return remaining_area

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        if (pos_x + prod_w > stock.shape[0]) or (pos_y + prod_h > stock.shape[1]):
            return False
        return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)