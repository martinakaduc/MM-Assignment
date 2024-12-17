from policy import Policy


class Policy2313257_2111682_2313101_2014717_1832026(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        super().__init__()
        self.policy_id = policy_id        

        self.stock_used_set = set()
        self.stock_area = 0
        self.product_area = 0
        self.prev_stock_used = None # Use to get prev stock use (maybe the best stock to use in the next time)

    def can_place_prev_stock(self, list_stocks, prod): 
        prod_size = prod["size"]
        is_place = False

        stock = list_stocks[self.prev_stock_used]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    # Calculate stock used percent
                    self.product_area += prod_w*prod_h
                    is_place = True
                if is_place == True:
                    break

            if is_place == True:
                break

                    
        if is_place == True: 
            return {
                "stock_idx": self.prev_stock_used, 
                "size": prod_size, 
                "position": (
                    x,
                    y
                )
            }
        return None

    def get_action(self, observation, info):
        """Main method to get cutting action based on selected policy"""
        list_prods = observation["products"]
        stocks = observation['stocks']


        # Create list of stocks
        list_stocks = list(stocks)

        # Sort products by width
        list_prods = sorted (list_prods, key = lambda x: x['size'][0], reverse=True)

        # Create an array of stock_size (stock contains no product)
        prod_size = [0, 0]
        best_fit = None
        is_place_prev = None

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                
                # If the previous stock enough space for current product, then cut from it
                if self.prev_stock_used != None:
                    is_place_prev = self.can_place_prev_stock(list_stocks, prod)
                if is_place_prev is not None:
                    break
                
                best_fit = {
                    "index": -1,
                    "x": 0,
                    "y": 0,
                    "waste": None,
                    "stock_area": 0,
                    "prod_area": 0
                }
                
                # Loop through all stocks
                for stock_index, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    is_place = False

                    # Prod is larger than stock in width or height
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                # Calculate stock used percent
                                temp_prod_area = self.product_area + prod_w*prod_h
                                if stock_index not in self.stock_used_set:
                                    temp_stock_area = self.stock_area + stock_w*stock_h
                                else:
                                    temp_stock_area = self.stock_area

                                waste = temp_stock_area - temp_prod_area

                                if best_fit["waste"] == None:
                                    best_fit['waste'] = waste
                                    best_fit["x"] = x
                                    best_fit["y"] = y
                                    best_fit["index"] = stock_index
                                    best_fit["stock_area"] = temp_stock_area
                                    best_fit['prod_area'] = temp_prod_area
                                elif waste < best_fit["waste"] and best_fit["waste"] != None:
                                    best_fit['waste'] = waste
                                    best_fit["x"] = x
                                    best_fit["y"] = y
                                    best_fit["index"] = stock_index
                                    best_fit["stock_area"] = temp_stock_area
                                    best_fit['prod_area'] = temp_prod_area
                                    is_place = True
                            if is_place == True:
                                break
                        if is_place == True:
                            break
            
                # Found best stock fit
                if best_fit["waste"] != None:
                    self.product_area = best_fit['prod_area']
                    self.stock_area = best_fit['stock_area']
                    if best_fit["index"] not in self.stock_used_set:
                        self.stock_used_set.add(best_fit["index"])
                    break
        
        # Place current product in the same stock with the previous product
        if is_place_prev is not None:
            self.prev_stock_used = is_place_prev["stock_idx"]
            return is_place_prev 
        
        # Mark the last stock used
        self.prev_stock_used = best_fit["index"]
        return {
            "stock_idx": best_fit["index"], 
            "size": prod_size, 
            "position": (
                best_fit["x"],
                best_fit["y"]
            )
        }