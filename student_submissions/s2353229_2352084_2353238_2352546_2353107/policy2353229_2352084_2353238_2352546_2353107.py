from policy import Policy
import numpy as np

class Policy2353229_2352084_2353238_2352546_2353107(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            super().__init__()

    def get_action(self, observation, info):
        # Student code here
        if  self.policy_id == 1:
            prods = observation["products"]

            sorted_prods = sorted(prods, key=lambda x: max(x["size"]), reverse=True)

            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0

            for prod in sorted_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]

                    best_stock = None
                    best_position = None
                    min = float("inf")  

                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size

                        if stock_w >= prod_w and stock_h >= prod_h:
                        
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):                                      
                                        temp = np.sum(stock == -1) - prod_w * prod_h

                                        if temp < min:
                                            min = temp
                                            best_stock = i
                                            best_position = (x, y)

                        if stock_w >= prod_h and stock_h >= prod_w:
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        temp = np.sum(stock == -1) - prod_h * prod_w

                                        if temp < min:
                                            prod_size = prod_size[::-1]
                                            min = temp
                                            best_stock = i
                                            best_position = (x, y)

                    
                    if best_stock is not None and best_position is not None:
                        stock_idx = best_stock
                        pos_x, pos_y = best_position
                        break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        
        elif self.policy_id == 2:
            prods = observation["products"]

            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0
            
            sorted_products = sorted(prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

            for prod in sorted_products:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
    
                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size

                
                        if (stock_w >= prod_w and stock_h >= prod_h) or (stock_w >= prod_h and stock_h >= prod_w):

                            cuts1 = self._subset_row_cut1(observation, i, prod_size)
                            cuts2 = self._subset_row_cut2(observation, i, prod_size[::-1])
                            
                            if cuts1 and cuts2:
                                row1, reduction1 = cuts1[0]
                                row2, reduction2 = cuts2[0]
                            
                                if reduction1 < reduction2:
                                    pos_x, pos_y = row1 
                                else:
                                    pos_x, pos_y = row2
                                    prod_size = prod_size[::-1]

                                stock_idx = i
                                break

                            elif cuts1:
                                row1, reduction1 = cuts1[0]
                                pos_x, pos_y = row1
                                stock_idx = i
                                break
                            elif cuts2:
                                row2, reduction2 = cuts2[0]
                                pos_x, pos_y = row2
                                prod_size = prod_size[::-1]
                                stock_idx = i
                                break

                    if pos_x is not None and pos_y is not None:
                        break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def _subset_row_cut1(self, observation, stock_idx, prod_size):
        stock = observation["stocks"][stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        possible_rows = []
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    possible_rows.append((x, y))

        cost = []
        for row in possible_rows:
            x, y = row
            reduction = self.reduction(stock, row, prod_size)
            cost.append((row, reduction))

    
        cuts = []
        for row, reduction in cost:
            if reduction != 0:  
                cuts.append((row, reduction))

        return cuts
    
    def _subset_row_cut2(self, observation, stock_idx, prod_size):

        stock = observation["stocks"][stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        possible_rows = []
        for x in range(stock_w - prod_h + 1):
            for y in range(stock_h - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    possible_rows.append((x, y))

        cost = []
        for row in possible_rows:
            x, y = row
            reduction = self.reduction(stock, row, prod_size)
            cost.append((row, reduction))

        cuts = []
        for row, reduction in cost:
            if reduction != 0:  
                cuts.append((row, reduction))

        return cuts


    def reduction(self, stock, row, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        x, y = row

        area_used = prod_w * prod_h
        leftover_area = (stock_w * stock_h) - area_used

        waste = leftover_area if leftover_area > prod_w * prod_h else 0

        return waste

        
    # Student code here
    # You can add more functions if needed
            