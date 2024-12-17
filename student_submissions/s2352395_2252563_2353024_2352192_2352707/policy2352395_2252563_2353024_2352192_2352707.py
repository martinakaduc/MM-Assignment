from policy import Policy
import numpy as np

class Policy2352395_2252563_2353024_2352192_2352707(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.used_stocks = []
        if policy_id == 1:
            self.policy_id = 1
            
        elif policy_id == 2:
            self.policy_id = 2

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            # Sort products by area in descending order
            list_prods = observation["products"]
            sorted_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            
            # Iterate over the sorted list of products
            for prod in sorted_prods:
                if prod["quantity"] > 0:    # Only consider products with quantity > 0
                    checked = [False] * len(observation["stocks"])  # Track which stocks have been checked
                    size = prod["size"]
                    prod_w, prod_h = size
                    fit = None

                    # Check used stocks first for potential placement
                    for i in self.used_stocks:
                        stock = observation["stocks"][i]
                        checked[i] = True   # Mark the stock as checked
                        placed = False
                        stock_w, stock_h = self._get_stock_size_(stock)
                        wasted = np.sum(stock == -1) - prod_w * prod_h  # Calculate wasted space if place at this stock

                        if (wasted >= 0 and (fit is None or wasted < fit["wasted"])):
                            # Try placing the product in its original orientation
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), size):
                                        fit = {"stock": i, "size": size, "x": x, "y": y, "wasted": wasted}
                                        placed = True
                                        break

                                if placed:
                                    break
                            
                            # If not placed, try rotated orientation
                            if not placed:
                                for x in range(stock_w - prod_h + 1):
                                    for y in range(stock_h - prod_w + 1):
                                        if self._can_place_(stock, (x, y), size[::-1]):
                                            fit = {"stock": i, "size": size[::-1], "x": x, "y": y, "wasted": wasted}
                                            placed = True
                                            break
                                    
                                    if placed:
                                        break


                    # If there is no placement in used stocks, check all unused stocks
                    if fit is None:
                        for i, stock in enumerate(observation["stocks"]):
                            if not checked[i]:  # Only consider unchecked stocks
                                placed = False
                                stock_w, stock_h = self._get_stock_size_(stock)
                                wasted = np.sum(stock == -1) - prod_w * prod_h  # Calculate wasted space if place at this stock

                                if (wasted >= 0 and (fit is None or wasted < fit["wasted"])):
                                    # Try placing the product in its original orientation
                                    for x in range(stock_w - prod_w + 1):
                                        for y in range(stock_h - prod_h + 1):
                                            if self._can_place_(stock, (x, y), size):
                                                self.used_stocks.append(i)
                                                fit = {"stock": i, "size": size, "x": x, "y": y, "wasted": wasted}
                                                placed = True
                                                break

                                        if placed:
                                            break
                                    
                                    # If not placed, try rotated orientation
                                    if not placed:
                                        for x in range(stock_w - prod_h + 1):
                                            for y in range(stock_h - prod_w + 1):
                                                if self._can_place_(stock, (x, y), size[::-1]):
                                                    self.used_stocks.append(i)
                                                    fit = {"stock": i, "size": size[::-1], "x": x, "y": y, "wasted": wasted}
                                                    placed = True
                                                    break
                                            
                                            if placed:
                                                break
                    
                    # If the product is placed, return the placement
                    if fit is not None:
                        # Calculate the remaining quantity of all products
                        rest = sum(prod["quantity"] for prod in sorted_prods)
                        
                        # If there is only one product left, reset used_stocks
                        if rest != 1:
                            return {"stock_idx": fit["stock"], "size": fit["size"], "position": (fit["x"], fit["y"])}

                        else:
                            self.used_stocks = []
                            return {"stock_idx": fit["stock"], "size": fit["size"], "position": (fit["x"], fit["y"])}    

            # If there is no placement for any product, return failure response
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        elif self.policy_id == 2:
            list_prods = observation["products"]

            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0
            area_official = 0
            index_official = 0 
            
            # Find the largest product
            for i in range(len(list_prods)):
                prod = list_prods[i]
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    area_ref = prod_w * prod_h
                    if area_ref > area_official:
                        area_official = area_ref
                        index_official = i
            
            best_prod = list_prods[index_official]     
            if best_prod["quantity"] > 0:
                prod_size = best_prod["size"]
                
                # Try to place the product
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    
                    # Try original orientation
                    best_pos = self._find_bottom_left_position(stock, stock_w, stock_h, prod_size)
                    
                    # If original orientation fails, try rotated
                    if best_pos is None:
                        prod_size = prod_size[::-1]  
                        best_pos = self._find_bottom_left_position(stock, stock_w, stock_h, prod_size)
                    
                    # If placement is found
                    if best_pos is not None:
                        stock_idx = i
                        pos_x, pos_y = best_pos
                        break
            
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
          
    def _find_bottom_left_position(self, stock, stock_w, stock_h, prod_size):
        prod_w, prod_h = prod_size
        
        # Check if the product fits in the stock
        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        
        return None

    # Student code here
    # You can add more functions if needed
