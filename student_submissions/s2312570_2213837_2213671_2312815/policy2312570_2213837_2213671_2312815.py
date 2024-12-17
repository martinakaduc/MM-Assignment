from policy import Policy
import numpy as np


class Policy2312570_2213837_2213671_2312815(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id

        if policy_id == 1:
            self.placement_method = self._skyline
        elif policy_id == 2:
            self.placement_method = self._guillotine

    def get_action(self, observation, info):
        # Student code here
        list_prods = observation["products"]

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                orientations = [
                    prod_size, 
                    prod_size[::-1] 
                ]
                
                for curr_size in orientations:
                    for stock_idx, stock in enumerate(observation["stocks"]):
                        if (self._get_stock_size_(stock)[0] >= curr_size[0] and 
                            self._get_stock_size_(stock)[1] >= curr_size[1]):
                            
                            placement = self.placement_method(stock, curr_size)
                            
                            if placement:
                                return {
                                    "stock_idx": stock_idx, 
                                    "size": curr_size, 
                                    "position": placement
                                }
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    # Student code here
    # You can add more functions if needed
    def _skyline(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        skyline = np.zeros(stock_w, dtype=int)
        
        for x in range(stock_w):
            for y in range(stock_h):
                if stock[y, x] != -1:
                    skyline[x] = y + 1
                    break
        best_y = float('inf')
        best_x = -1
        
        for x in range(stock_w - prod_w + 1):
            max_y = max(skyline[x:x+prod_w])
        
            full_area_valid = True
            for dx in range(prod_w):
                if x+dx < stock_w and max_y + prod_h > stock_h:
                    full_area_valid = False
                    break
            if full_area_valid:
                placement_valid = all(
                    self._can_place_(stock, (x+dx, max_y), (1, prod_h)) 
                    for dx in range(prod_w)
                )
                
                if placement_valid:
                    if max_y < best_y:
                        best_y = max_y
                        best_x = x
        
        return (best_x, best_y) if best_x != -1 else None

    def _guillotine(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        best_placement = None
        min_waste = float('inf')

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    waste = self._calculate_local_waste(
                        stock, (x, y), (prod_w, prod_h)
                    )
                    
                    if waste < min_waste:
                        min_waste = waste
                        best_placement = (x, y)

        return best_placement

    def _calculate_local_waste(self, stock, position, prod_size):
        x, y = position
        width, height = prod_size
        adjacent_waste = 0
        
        right_space = np.sum(stock[y:y+height, x+width:] == -1)
    
        bottom_space = np.sum(stock[y+height:, x:x+width] == -1)
        
        adjacent_waste = right_space + bottom_space
        
        return adjacent_waste

