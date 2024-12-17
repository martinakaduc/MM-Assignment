from policy import Policy
import numpy as np
import copy as cp
import time

#Member: Class CC01 - Group 26
#Lê Đỗ Minh Anh - 2252023
#Trần Đăng Khoa - 2252363

class Policy2252023_2252363(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.Policy = PolicyBFD()
        if policy_id == 2:
            self.Policy = First_Fit_Policy()
            
    def get_action(self, observation, info):
        return self.Policy.get_action(observation, info)
    
        
class PolicyBFD(Policy):
    def __init__(self):
        self.list_prods = []
        self.list_stocks = []
        self.action = []
        self.action_index = -1
        #performance
        self.start_time = 0
        self.end_time = 0
        self.used_area = 0
        self.total_area = 0
        self.used_stock = []
    
    #reset all variables after a testcase
    def reset(self):
        self.list_prods = []
        self.list_stocks = []
        self.action = [] #store all action in a list and drop each turn
        self.action_index = -1
        self.start_time = 0
        self.end_time = 0
        self.used_area = 0
        self.total_area = 0
        self.used_stock = []
        
    def get_action(self, observation, info):
        if (self.action_index == -1):
            self.start_time = time.time()
            self.action_index += 1
            self.obs = cp.deepcopy(observation) #need deepcopy to avoid error
            self.list_prods = self.obs["products"]
            self.list_prods = self.sorted_product(self.list_prods) #sort size descending
            self.list_stocks = self.obs["stocks"]
            #for stock in self.list_stocks:
            #    w, h = self._get_stock_size_(stock) #for performance
            #    self.total_area += w * h
            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0
            for prod_idx, prod in enumerate(self.list_prods):
                while prod["quantity"] > 0:
                    current_prod = cp.deepcopy(prod_idx)
                    prod_size = prod["size"]
                    #find the stock to start cutting
                    stock_idx, position, rotate = self.find_best_fit_stock(prod_size)
                    if stock_idx != -1 and position is not None:
                        #if stock_idx not in self.used_stock:
                        #    self.used_stock.append(stock_idx)
                        #    w, h = self._get_stock_size_(self.list_stocks[stock_idx])
                        #    self.used_area += w * h
                        pos_x, pos_y = position
                        if rotate:
                            prod_size = prod_size[::-1]
                        prod_w, prod_h = prod_size
                        self.place_product(stock_idx, pos_x, pos_y, prod_w, prod_h)
                        prod["quantity"] -= 1
                        new_action = {
                            'stock_idx': stock_idx,
                            'size': prod_size,
                            "position": (pos_x, pos_y)
                        }
                        self.action.append(new_action)
                        #after place the first product, try to place another product
                        # on the same stock to maximize the utility
                        while True:
                            if self.list_prods[current_prod]["quantity"] <= 0:
                                current_prod += 1
                                if current_prod >= len(self.list_prods):
                                    break
                                else:
                                    continue
                            prod_size = self.list_prods[current_prod]["size"]
                            prod_w, prod_h = prod_size
                            position, rotate = self.place_smaller(stock_idx, prod_w, prod_h)
                            if position is not None:
                                pos_x, pos_y = position
                                if rotate:
                                    prod_size = prod_size[::-1]
                                prod_w, prod_h = prod_size
                                self.place_product(stock_idx, pos_x, pos_y, prod_w, prod_h)
                                self.list_prods[current_prod]["quantity"] -= 1
                                new_action = {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (pos_x, pos_y)
                                }
                                self.action.append(new_action)
                            else:
                                current_prod += 1
                                if current_prod >= len(self.list_prods):
                                    break
        self.action_index += 1
        #drop the action each turn
        if (self.action_index - 1) >= len(self.action):
            self.end_time = time.time()
            #print("Running time: ", self.end_time - self.start_time)
            #print("Percent area used: ", self.used_area / self.total_area)
            self.reset()
            return {
                "stock_idx": -1,
                "size": [0, 0],
                "position": (0, 0)
            }
        else:
            return self.action[self.action_index - 1]
                            
                    
    def sorted_product(self, list_prod):
        return sorted(list_prod, key=lambda p: p["size"][0] * p["size"][1], reverse=True)
    
    def place_product(self, stock_idx, pos_x, pos_y, prod_w, prod_h):
        #place product on stock by changing the value to -2
        self.list_stocks[stock_idx][pos_x:pos_x + prod_w, pos_y:pos_y+prod_h] = -2
        
    def place_smaller(self, stock_idx, prod_w, prod_h):
        #find a place to place the next product on the same stock
        stock_w, stock_h = self._get_stock_size_(self.list_stocks[stock_idx])
        best_position = None
        best_fragmentation = float('inf')
        rotate = False
        #for not rotate
        if stock_w >= prod_w and stock_h >= prod_h:
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(self.list_stocks[stock_idx], (x, y), (prod_w, prod_h)):
                        fragmentation = (stock_w - prod_w) * stock_h + stock_w * (stock_h - prod_h) #optimize fragmentation
                        if (fragmentation < best_fragmentation):
                            best_position = (x, y)
                            rotate = False
                            best_fragmentation = fragmentation
        #for rotate   
        if stock_w >= prod_h and stock_h >= prod_w:
            for x in range(stock_w - prod_h + 1):
                for y in range(stock_h - prod_w + 1):
                    if self._can_place_(self.list_stocks[stock_idx], (x, y), (prod_h, prod_w)):
                        fragmentation = (stock_w - prod_h) * stock_h + stock_w * (stock_h - prod_w)
                        if (fragmentation < best_fragmentation):
                            best_position = (x, y)
                            rotate = True
                            best_fragmentation = fragmentation
        return best_position, rotate
    
    def find_best_fit_stock(self, size):
        #find the best fit stock by choosing the stock with lowest waste
        best_stock_index = -1
        best_remaining_area = float('inf')
        best_position = None
        rotate = False
        best_fragmentation = float('inf')
        for stock_index, stock in enumerate(self.list_stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = size
            #for not rotate
            if stock_w >= prod_w and stock_h >= prod_h:
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                            remaining_area = (stock_w * stock_h) - (prod_w * prod_h)
                            fragmentation = (stock_w - prod_w) * stock_h + stock_w * (stock_h - prod_h) #optimize fragmentation
                            if (remaining_area < best_remaining_area or
                                (remaining_area == best_remaining_area and fragmentation < best_fragmentation)):
                                best_stock_index = stock_index
                                best_remaining_area = remaining_area
                                best_position = (x, y)
                                best_fragmentation = fragmentation
                                rotate = False
            #for rotate
            if stock_w >= prod_h and stock_h >= prod_w:
                for x in range(stock_w - prod_h + 1):
                    for y in range(stock_h - prod_w + 1):
                        if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                            remaining_area = (stock_w * stock_h) - (prod_h * prod_w)
                            fragmentation = (stock_w - prod_h) * stock_h + stock_w * (stock_h - prod_w)
                            if (remaining_area < best_remaining_area or
                                (remaining_area == best_remaining_area and fragmentation < best_fragmentation)):
                                best_stock_index = stock_index
                                best_remaining_area = remaining_area
                                best_fragmentation = fragmentation
                                best_position = (x, y)
                                rotate = True
        return best_stock_index, best_position, rotate
    
class First_Fit_Policy(Policy):
    def __init__(self):
        self.total_time = 0
        
    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)

    def get_action(self, observation, info):
        start_time = time.time()  
        list_prods = observation["products"]
        stocks = observation["stocks"]
        # Iterate until there are no more products to place
        for prod in list_prods:
            if prod["quantity"] > 0:
                new_pattern, _ = self._find_first_fit(list_prods, stocks)
                if new_pattern is not None:
                    end_time = time.time() 
                    self.total_time += end_time - start_time
                    return new_pattern["action"]
                else: break
        end_time = time.time() 
        self.total_time += end_time - start_time
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_first_fit(self, list_prods, stocks):
        """
        Implements the first-fit decreasing algorithm:
        1. Sorts products by area in descending order.
        2. Tries to place each product in the first stock that fits.
        3. Returns the action for placing the product.
        """
        sorted_prods = sorted(list_prods,
            key=lambda prod: prod["size"][0] * prod["size"][1], #Sorting in decreasing order
            reverse=True,)

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Check normal orientation
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {
                                    "action": {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y),
                                    },
                                }, None

                    # Check rotated orientation
                    if stock_w >= prod_h and stock_h >= prod_w:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    return {
                                    "action": {
                                        "stock_idx": stock_idx,
                                        "size": prod_size[::-1],
                                        "position": (x, y),
                                    },
                                }, None
        return None, None  
    def evaluate(self):
        print(" - Total Time:", self.total_time, "s")        
            
