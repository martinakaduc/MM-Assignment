from policy import Policy
import numpy as np


class Policy2352291_2352268_2352328(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.mode = policy_id
        
        
        if policy_id == 1:
            self.filled_stock = []
            
        elif policy_id == 2:
            self.stock_fits = None  # Cache for sorted stock fits

    def get_action(self, observation, info):
        if self.mode == 1:
            return self.HBF_action(observation, info)
        elif self.mode == 2:
            return self.DP_action(observation, info)
    # Main function begin #################################
    
    # First algorithm begin
    def HBF_action(self, observation, info):
        list_prods = observation["products"]
        area = []
        sortedProd = []
        prod_size = [0, 0]
        pos_x, pos_y = 0, 0
        decision_array = []
        
        for prod in list_prods:
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            area.append((prod_w * prod_h, prod))
                
        area = np.array(area)
        sorted_area = area[area[:, 0].argsort()[::-1]]
        middle = (len(sorted_area)-1)/2
        sortedProd = sorted_area[:, 1]
        count = 0
            
        for prods in sortedProd:
            count += 1 #Plus 1 to indicate the product type
            if prods["quantity"] > 0:
                prod_size = prods["size"]
                prod_w, prod_h = prod_size
                if count <= middle: #First half
                    if prod_w >= prod_h:
                        if prod_w < 1.25*prod_h: #Check for nearly square product
                            decision_array = self.get_w_array(observation, prod_size)
                            if decision_array:
                                break        
                        else:
                            for i, stock in enumerate(observation["stocks"]):
                                stock_w, stock_h = self._get_stock_size_(stock)
                                # Generate possible placements
                                if stock_w >= prod_w and stock_h >= prod_h:
                                    pos_x, pos_y = None, None
                                    for x in range(stock_w - prod_w + 1):
                                        for y in range(stock_h - prod_h + 1):
                                            if self._can_place_(stock, (x, y), prod_size):
                                                pos_x, pos_y = x, y
                                                return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}
                                # Perform a rotation
                                if stock_w >= prod_h and stock_h >= prod_w:
                                    pos_x, pos_y = None, None
                                    for x in range(stock_w - prod_h + 1):
                                        for y in range(stock_h - prod_w + 1):
                                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                                pos_x, pos_y = x, y
                                                prod_size = prod_size[::-1]
                                                return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}    
                                        
                    else:
                        if prod_h < 1.25*prod_w:
                            decision_array = self.get_h_array(observation, prod_size)
                            if decision_array:
                                break        
                        else:
                            for i, stock in enumerate(observation["stocks"]):
                                stock_w, stock_h = self._get_stock_size_(stock)
                                # Generate possible placements
                                if stock_w >= prod_w and stock_h >= prod_h:
                                    pos_x, pos_y = None, None
                                    for x in range(stock_w - prod_w + 1):
                                        for y in range(stock_h - prod_h + 1):
                                            if self._can_place_(stock, (x, y), prod_size):
                                                pos_x, pos_y = x, y
                                                return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}
                                # Perform a rotation
                                if stock_w >= prod_h and stock_h >= prod_w:
                                    pos_x, pos_y = None, None
                                    for x in range(stock_w - prod_h + 1):
                                        for y in range(stock_h - prod_w + 1):
                                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                                pos_x, pos_y = x, y
                                                prod_size = prod_size[::-1]
                                                return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}   
                                                 
                else:  
                    for i in reversed(self.filled_stock):
                        curr_stock = observation["stocks"][i]
                        stock_w, stock_h = self._get_stock_size_(curr_stock)
                        # Generate possible placements
                        if stock_w >= prod_w and stock_h >= prod_h:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(curr_stock, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}
                        # Perform a rotation
                        if stock_w >= prod_h and stock_h >= prod_w:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(curr_stock, (x, y), prod_size[::-1]):
                                        pos_x, pos_y = x, y
                                        prod_size = prod_size[::-1]
                                        return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}    
                        
                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        # Generate possible placements
                        if stock_w >= prod_w and stock_h >= prod_h:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}
                        # Perform a rotation
                        if stock_w >= prod_h and stock_h >= prod_w:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        pos_x, pos_y = x, y
                                        prod_size = prod_size[::-1]
                                        return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}    
                        
                            
        
        decision_array.sort(key=lambda x: x[0])
        result = decision_array[0]
        self.add_filled_stock(result[1])
        return {"stock_idx": result[1], "size": prod_size, "position": result[2]}
    # First algorithm end   
        
    # Second algorithm begin
    def DP_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        stock_idx = -1
        prod_size = [0, 0]
        pos_x, pos_y = 0, 0
        product_sizes = []
        for prod in list_prods:
            if prod["quantity"] > 0:
                product_sizes.extend([prod["size"]] * prod["quantity"])
        if not product_sizes:
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        if self.stock_fits is None:
            self.update_stock_fits(stocks, product_sizes)
        for i, _ in self.stock_fits:
            stock = stocks[i]
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_w, prod_h = prod["size"]
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod["size"]):
                                prod_size = prod["size"]
                                pos_x, pos_y = x, y
                                stock_idx = i
                                return {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (pos_x, pos_y),
                                }
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod["size"][::-1]):
                                prod_size = prod["size"][::-1]
                                pos_x, pos_y = x, y
                                stock_idx = i
                                return {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (pos_x, pos_y),
                                }
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    # Second algorithm end
         
    # Main function end #################################
        
    #######################################################
    ################## HELPER METHOD  #####################
    #######################################################
    
    # Helper for first algorithm begin
    def add_filled_stock(self, stock_idx):
        if stock_idx not in self.filled_stock:
            self.filled_stock.append(stock_idx)  
    
    def get_w_array(self, observation, prod_size):
        w_array = []
        pos_w, pos_h = 0, 0
        prod_w, prod_h = prod_size
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            if stock_w >= prod_w and stock_h >= prod_h:
                pos_w, pos_h = None, None
                for w in range(stock_w - prod_w + 1):
                    for h in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (w, h), prod_size):
                            pos_w, pos_h = w, h
                            w_array.append(((stock_w-(pos_w+prod_w)), i, (pos_w, pos_h)))
                            break
                    if pos_w is not None and pos_h is not None:
                        break
                    else:
                        w_array.append((1000000000, i, (pos_w, pos_h)))
                        break
            else:
                w_array.append((1000000000, i, (pos_w, pos_h)))              
        return w_array
    
    
    def get_h_array(self, observation, prod_size):
        h_array = []
        pos_w, pos_h = 0, 0
        prod_w, prod_h = prod_size
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            if stock_w >= prod_w and stock_h >= prod_h:
                for w in range(stock_w - prod_w + 1):
                    for h in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (w, h), prod_size):
                            pos_w, pos_h = w, h
                            h_array.append(((stock_h-(pos_h+prod_h)), i, (pos_w, pos_h)))
                            break
                    if pos_w is not None and pos_h is not None:
                        break
                    else:
                        h_array.append((1000000000, i, (pos_w, pos_h)))
                        break     
            else:
                h_array.append((1000000000, i, (pos_w, pos_h)))              
        return h_array
    # Helper for first algorithm end
    
    
    # Helper for second algorithm begin
    def _knapsack_2d(self, stock_width, stock_height, products):
        dp = [[0] * (stock_height + 1) for _ in range(stock_width + 1)]

        for w in range(stock_width + 1):
            for h in range(stock_height + 1):
                for product in products:
                    pw, ph = product 
                    if pw <= w and ph <= h:
                        dp[w][h] = max(dp[w][h], dp[w - pw][h] + 1) 
                    if ph <= w and pw <= h:
                        dp[w][h] = max(dp[w][h], dp[w][h - ph] + 1) 

        return dp[stock_width][stock_height]
    
    
    def update_stock_fits(self, stocks, product_sizes):
        """
        Update the cached stock_fits based on the current stocks and product sizes.
        """
        self.stock_fits = [
            (i, self._knapsack_2d(*self._get_stock_size_(stock), product_sizes))
            for i, stock in enumerate(stocks)
        ]
        self.stock_fits.sort(key=lambda x: x[1], reverse=True)
    # Helper for second algorithm end