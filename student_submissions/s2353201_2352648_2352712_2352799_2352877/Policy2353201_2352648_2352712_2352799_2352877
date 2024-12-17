import numpy as np
from policy import Policy

class Policy2353201_2352648_2352712_2352799_2352877(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        if policy_id == 1:
            self._get_action_impl = self.greedy
        elif policy_id == 2:
            self._get_action_impl = self.BLF

    def get_action(self, observation, info):
        return self._get_action_impl(observation, info)
    
    def greedy(self, observation, info):
      # Sort products by area (width * height) in descending order
        list_prods = sorted(
            observation["products"], 
            key=lambda prod: (
                -(prod["size"][0] * prod["size"][1]), 
                -min(prod["size"]),                  
                max(prod["size"])                  
            )
        )
        stocks_with_efficiency = [
            (idx, self._get_stock_efficiency_(stock), stock) 
            for idx, stock in enumerate(observation["stocks"])
        ]
        stocks_with_efficiency.sort(key=lambda x: x[1]) 

        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue

            prod_size = prod["size"]
            rotated_size = prod_size[::-1]
            for idx, _, stock in stocks_with_efficiency:
                stock_w, stock_h = self._get_stock_size_(stock)
                for size in (prod_size, rotated_size):
                    prod_w, prod_h = size
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos = self._find_placement_1(stock, prod_w, prod_h)
                        if pos:  
                            return {"stock_idx": idx, "size": size, "position": pos}
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_placement_1(self, stock, prod_w, prod_h):
        stock_w, stock_h = self._get_stock_size_(stock)
        best_position = None
        best_score = float('inf')
    
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    score = (stock_w - (x + prod_w)) + (stock_h - (y + prod_h)) * 0.5
                
                    if score < best_score:
                        best_score = score
                        best_position = (x, y)
    
        return best_position

    def _get_stock_efficiency_(self, stock):
        stock_w, stock_h = self._get_stock_size_(stock)

        used_spaces = np.sum(stock != -1)
        total_spaces = stock_w * stock_h
        free_spaces = total_spaces - used_spaces
        fragmentation_score = free_spaces / total_spaces
        edge_penalty = 0
        for x in range(stock_w):
            if np.all(stock[:, x] == -1):  
                edge_penalty += 1
        for y in range(stock_h):
            if np.all(stock[y, : ] == -1) :
                edge_penalty += 1

        min_dim = min(stock_w, stock_h)
        small_free_space_penalty = free_spaces / (min_dim ** 2)

        
        aspect_ratio = max(stock_w / stock_h, stock_h / stock_w)

    
        efficiency_score = (
            fragmentation_score * 0.3 +        
            edge_penalty * 0.3 +               
            small_free_space_penalty * 0.3 +   
            (1 / aspect_ratio) * 0.1           
        )

        return efficiency_score

    def BLF(self, observation, info):
        list_prods = sorted(observation["products"], key=lambda prod: (prod["size"][0] * prod["size"][1], prod["size"][0] / prod["size"][1]), reverse=True)
        stocks = sorted(enumerate(observation["stocks"]), 
                        key=lambda item: self._get_stock_size_(item[1])[0] * self._get_stock_size_(item[1])[1])

        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue
           
            prod_size = prod["size"]
            rotated_size = prod_size[::-1]

            for idx, stock in stocks:
                stock_w, stock_h = self._get_stock_size_(stock)

                for size in (prod_size, rotated_size):
                    prod_w, prod_h = size
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos = self._find_placement2(stock, prod_w, prod_h)
                        if pos:
                            return {"stock_idx": idx, "size": size, "position": pos}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_placement2(self, stock, prod_w, prod_h):
        stock_w, stock_h = self._get_stock_size_(stock)

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    return (x, y)
        return None


    
