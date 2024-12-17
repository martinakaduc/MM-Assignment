from policy import Policy
import numpy as np
import scipy
from scipy.signal import correlate

class Policy2313293_2313950_2310280(Policy):
    # Initialize
    def __init__(self, policy_id = 1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.num = 1
        self.policy_id = policy_id
        self.sorted_product = []
        self.list_stock = []
        self.distribute = 0
        if policy_id == 1:
            self.strategy = self.strategy_1
        elif policy_id == 2:
            self.strategy = self.strategy_2
    
    ###############################################################
    # HELPER FUNCTION
    ###############################################################    
    # Return the area of the rectangle stock and product
    def _calculated_area_(self, height, width):
        return height * width
    
    ###############################################################
    # POLICY 1
    ###############################################################
    def strategy_1(self, observation, info):
        return self.select_and_place(observation, info)
        
    def select_and_place(self, observation, info):
        if len(self.sorted_product) == 0 or all([product["quantity"] == 0 for product in self.sorted_product]):
            list_prods = observation["products"]
            self.sorted_product = self.sort_products_by_max_dimension(list_prods)
            self.list_stock.clear()
            self.distribute = 0
        if len(self.list_stock) == 0:
            self.add_suitable_stock(observation["stocks"])

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        # restrict loop (which mean we first create strip with longer edge and higher edge first)
        # Choose final adding stock only cus other have been filled no more addable
        idx = len(self.list_stock)-1
        index = self.list_stock[idx]
        # #run 
        stock_w, stock_h = self._get_stock_size_(observation["stocks"][index])
        
        for prod in self.sorted_product:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                if stock_h >=stock_w:
                    pos_x, pos_y = None, None
                    pos_x, pos_y, prod_size = self.non_rotated_prod_place(observation["stocks"][index],prod)
                    if pos_x is not None and pos_y is not None:
                        stock_idx = index
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
                
                else:
                    pos_x, pos_y = None, None
                    pos_x, pos_y, prod_size = self.rotated_prod_place(observation["stocks"][index],prod)
                    if pos_x is not None and pos_y is not None:
                        stock_idx = index
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        # unrestrict  which mean we now filling the gap as much as posible to reduce trimloss
        for prod in self.sorted_product:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                if stock_w >= prod_w and stock_h >= prod_h:
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if observation["stocks"][index][x][y]!=-1 :
                                continue
                            if self._can_place_(observation["stocks"][index], (x, y), prod_size):
                                pos_x, pos_y = x, y
                                stock_idx = index
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

                if stock_w >= prod_h and stock_h >= prod_w:
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if observation["stocks"][index][x][y]!=-1 :
                                continue
                            if self._can_place_(observation["stocks"][index], (x, y), prod_size[::-1]):
                                prod_size = prod_size[::-1]
                                pos_x, pos_y = x, y
                                stock_idx = index
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        #if there still no where to place that mean we cannt fill prod with current stocks anymore
        # using defined funct to add new suitable stock and run a lastest version of this function
        self.add_suitable_stock(observation["stocks"])
        return self.select_and_place(observation, info)
    def recurisve(self, prod, stock, lower, upper):
        
        return None
    def non_rotated_prod_place(self, stock, prod):
        prod_size = prod["size"]
        prod_w, prod_h = prod_size
        stock_w, stock_h = self._get_stock_size_(stock)
        # if(prod_w > prod_h):
        if stock_w >= prod_w and stock_h >= prod_h:
            for x in range(stock_w - prod_w  ):
                for y in range(stock_h - prod_h):
                    if stock[x][y]!=-1 :
                        continue
                    if y-1 >=0 and x-1>=0:
                        #priority with longer edge
                        if prod_h < prod_w and x+prod_h < stock_w and y + prod_w < stock_h and self.constrains(stock, (x,y) ,prod_size[::-1]):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                return x, y, prod_size[::-1]
                        if self.constrains(stock, (x,y) ,prod_size):
                            if self._can_place_(stock, (x, y), prod_size):
                                return x, y, prod_size
                    else:
                        if self._can_place_(stock, (x, y), prod_size):
                            return x, y, prod_size
        # else:
        if stock_w >= prod_h and stock_h >= prod_w:
            for x in range(stock_w - prod_h ):
                for y in range(stock_h - prod_w ):
                    if stock[x][y]!=-1 :
                        continue
                    if y-1 >=0 and x-1>=0:
                        if self.constrains(stock, (x,y) ,prod_size[::-1]) :
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                return x, y, prod_size[::-1]
                    else:
                        if self._can_place_(stock, (x, y), prod_size[::-1]):
                            return x, y, prod_size[::-1]
        return None, None, None
    def rotated_prod_place(self, stock, prod):
        prod_size = prod["size"]
        prod_w, prod_h = prod_size
        stock_w, stock_h = self._get_stock_size_(stock)
        # if(prod_h > prod_w):
        for y in range(stock_h - prod_h  ):
            for x in range(stock_w - prod_w ):
                if stock[x][y]!=-1 :
                    continue
                if y-1 >=0 and x-1>=0:
                    if prod_h > prod_w and x+prod_h < stock_w and y + prod_w < stock_h and self.constrains(stock, (x,y) ,prod_size[::-1]):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                return x, y, prod_size[::-1]
                    if self.constrains(stock, (x,y) ,prod_size):
                        if self._can_place_(stock, (x, y), prod_size):
                            return x, y, prod_size
                else :
                    if self._can_place_(stock, (x, y), prod_size):
                        return x, y, prod_size
        # else:
        for y in range(stock_h - prod_w ):
            for x in range(stock_w - prod_h ):
                if stock[x][y]!=-1 :
                    continue
                if y-1 >=0 and x-1>=0:
                    if self.constrains(stock, (x,y) ,prod_size[::-1]) :
                        if self._can_place_(stock, (x, y), prod_size[::-1]):
                            return x, y, prod_size[::-1]
                else :
                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                        return x, y, prod_size[::-1]
        return None, None, None
    def constrains(self, stock , position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        return np.all(stock[pos_x : pos_x + prod_w, pos_y -1] != -1) and np.all(stock[pos_x -1, pos_y : pos_y + prod_h] != -1)
    def sort_products_by_max_dimension(self, products):
        # Extract the maximum dimension for sorting
        max_dims = [max(product["size"]) for product in products]
        # Sort products by max dimension in descending order
        sorted_indices = np.argsort(max_dims)[::-1]
        sorted_products = [products[i] for i in sorted_indices]
        return sorted_products
    def add_suitable_stock(self, stocks):
        # Find the largest area product with quantity > 0
        largest_product_area = 0
        trend = 0
        count = 0
        for product in self.sorted_product:
            if product["quantity"] > 0:
                area = self._calculated_area_(product["size"][0], product["size"][1])
                if area > largest_product_area:
                    largest_product_area = area
        sum_area = 0
        for product in self.sorted_product:
            if product["quantity"] > 0:
                prod_w, prod_h = product["size"] 
                temp = prod_w / prod_h if prod_w > prod_h else prod_h / prod_w
                area = self._calculated_area_(prod_w, prod_h)
                count+= product["quantity"]
                trend += temp*product["quantity"]
                sum_area += area*product["quantity"]
        average = trend / count 

        min_shape_dis =  None

        best_index = None
        best_loss = None
        # Iterate through stocks to find a near square stock larger than the largest product
        for i,stock in enumerate(stocks):
            width, height = self._get_stock_size_(stock)
            stock_area = width * height
            if stock_area > largest_product_area and abs(width - height) <= self.distribute:
                not_add_yet = True
                for index in self.list_stock:
                    if i == index:
                        not_add_yet = False
                        break
                if not_add_yet == False:
                    continue
                else :
                    propose = width/height if width > height else height/width
                    if best_loss is not None:
                        if min_shape_dis <= 0 and sum_area-stock_area <= 0:
                            if sum_area-stock_area > min_shape_dis and stock_area/sum_area < 1.5:
                                if propose-average < abs(best_loss):
                                    best_loss = propose-average
                                    min_shape_dis = sum_area-stock_area
                                    best_index = i
                        else:
                            if min_shape_dis >=0 and sum_area-stock_area <= 0:
                                best_loss = propose-average
                                min_shape_dis = sum_area-stock_area
                                best_index = i
                            else :
                                if min_shape_dis >=0 :
                                    if sum_area-stock_area < min_shape_dis:
                                        best_loss = propose-average
                                        min_shape_dis = sum_area-stock_area
                                        best_index = i
                    else:
                        min_shape_dis = sum_area-stock_area
                        best_loss = propose-average 
                        best_index = i
        if best_index is not None :
            self.list_stock.append(best_index)
            return   
        self.distribute+=1
        self.add_suitable_stock(stocks)
  
    
    ###############################################################
    # POLICY 2
    ###############################################################
    # Action the policy 2
    def strategy_2(self, observation, info):
        prods = observation["products"]
        stocks = observation["stocks"]
        sorted_prods = self.sort_products(prods)
        sorted_stocks_indices = self.sort_stocks(stocks)

        for stock_indices in sorted_stocks_indices:
            for prod in sorted_prods:
                if prod["quantity"] > 0:
                    prod_size = np.array(prod["size"])
                    prod_w, prod_h = prod_size
                    stock = stocks[stock_indices]
                    spaces = self.find_place(stock, prod_size)   
                    if spaces.size > 0:
                        x, y  = spaces[0]
                        return {
                            "stock_idx" : stock_indices,
                            "size"      : prod_size,
                            "position"  : (x,y),
                        }
                    else:
                        spaces = self.find_place(stock, np.flip(prod_size))
                        if spaces.size > 0:
                            x, y  = spaces[0]
                            return {
                                "stock_idx" : stock_indices,
                                "size"      : np.flip(prod_size),
                                "position"  : (x,y),
                            }

    # Find feasiable place to allocate the product
    def find_place(self, stock, prod_size):
        r_rows, r_cols = prod_size
        
        # Find the first rows and columns where there's a valid empty space (-2)
        rows = np.argmin(stock[:, 0] == -2)  # Find the first empty row
        cols = np.argmin(stock[0, :] == -2)  # Find the first empty column
        
        # If there are no empty spaces, adjust the index
        if stock[rows, 0] != -2: 
            rows = stock.shape[0]
        if stock[0, cols] != -2:
            cols = stock.shape[1]
        
        # Pad the stock grid with 1s (empty) and 0s (filled)
        stocker = np.pad(1 - (stock[:rows, :cols] == -1), 1, mode = "constant", constant_values = 1)

        # Define convolution kernels
        rect_kernel = np.ones((r_rows, r_cols), dtype = int)
        up_kernel = np.ones((1, r_cols), dtype = int)
        left_kernel = np.ones((r_rows, 1), dtype = int)

        # Check for valid placements using convolution
        condition1 = correlate(stocker, rect_kernel)[r_rows:-1, r_cols:-1] == 0  # Check rectangular area
        condition2 = correlate(stocker, up_kernel)[0:-2, r_cols:-1] >= 1         # Check upward continuity
        condition3 = correlate(stocker, left_kernel)[r_rows:-1, 0:-2] >= 1       # Check left continuity
        
        # Combine the results to find all valid placement locations
        result = condition1 & condition2 & condition3
        
        # Get the positions (corners) where the product can fit
        corners = np.argwhere(result)
        return corners
    
    # Return the non-ascending sorted products list
    def sort_products(self, prods):
        product_areas = np.array([self._calculated_area_(prod["size"][0], prod["size"][1]) for prod in prods])
        sorted_prods_indices = np.argsort(product_areas)[ : :-1]
        sorted_prods = []
        for i in sorted_prods_indices:
            sorted_prods.append(prods[i])
        return sorted_prods
    
    # Return the non-ascending sorted stock's indices list
    def sort_stocks(self, stocks):
        areas = np.array([self._calculated_area_(np.sum(np.any(stock != -2, axis=1)), np.sum(np.any(stock != -2, axis=0))) for stock in stocks])
        sorted_stocks_indices = np.argsort(areas)[ : :-1]
        return sorted_stocks_indices
    
    def get_action(self, observation, info):
        return self.strategy(observation, info)