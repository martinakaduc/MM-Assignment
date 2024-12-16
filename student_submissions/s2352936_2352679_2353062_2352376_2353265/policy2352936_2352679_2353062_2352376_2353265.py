from policy import Policy
import numpy as np

class Policy2352936_2352679_2353062_2352376_2353265(Policy):

    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.using_FFD = True
        elif policy_id == 2:
            self.using_FFD = False

        self.first_action = True    # to check if first call get_action
        self.saved_stocks = []      # to save stocks
        self.prods = []             # to save product

        # for easier manage and sort the 2 list above
        self.used_stocks_idx = []
        self.not_used_stocks_idx = []
        self.is_used = []
        self.available_area = []
        self.prod_idx = []
        self.dist_prod_idx = []

        # to save info of product in stock
        self.prods_in_stocks_info = [] # info của các product trong stock

        # variables for tracking state
        self.current_stock_idx = 0
        self.current_product_idx = 0

    def get_action(self, observation, info):
        # assign stock and product list if first observation
        if (self.first_action):
            self.first_assignment(observation["stocks"], observation["products"])

            # do feasible cut
            check = self.init_cut()

            if not check:
                print("No solution")

            # improve the FFD until no step can improve
            if self.using_FFD:
                while self.optimize_FFD():
                    pass

            # end first action
            self.first_action = False

        # get the action
        return self.get_action_after_all()

    # Additional function

    # 3 function to assist sorting
    # to calculate area of stock
    def cal_stock_area(self, stock):
        stock_w, stock_h = self._get_stock_size_(stock)
        return stock_w * stock_h

    # to calculate perimeter of stock
    def cal_stock_half_peri(self, stock):
        stock_w, stock_h = self._get_stock_size_(stock)
        return stock_w + stock_h
    
    # to distribute product by its quantity
    def distribute_prod(self, products, products_idx):
        quantity = [product["quantity"] for product in products]
        result = []

        # sort for Best Fit
        if not self.using_FFD:
            # ! if quantity not 0, append to idx list  
            while any(qtt > 0 for qtt in quantity):
                for idx in products_idx:
                    if quantity[idx] > 0:
                        result.append(idx)
                        quantity[idx] -= 1

        else: # sort for First Fit
            for idx in products_idx:
                for _ in range(products[idx]["quantity"]):
                    result.append(idx)
        
        return result

    # to assign attributes at first action
    def first_assignment(self, stocks, products):
        # init stock, prod, avai area and is_used to check if a stock is used
        self.saved_stocks = [stock.copy() for stock in stocks]
        self.prods = products
        self.available_area = list(self.cal_stock_area(stock) for stock in stocks)
        self.is_used = [False] * len(stocks)
        self.prods_in_stocks_info = [[] for _ in range(len(stocks))]

        # to store idx of stock after sort
        self.not_used_stocks_idx = list(range(len(stocks)))

        # then sort it by decreasing area or increasing area based on FFD or BFD
        reverse_for_stock = False
        if self.using_FFD:
            reverse_for_stock = True
        self.not_used_stocks_idx.sort(key = lambda x : (self.available_area[x], self.cal_stock_half_peri(self.saved_stocks[x])), 
                                    reverse = reverse_for_stock) #!

        # to sort product_idx by decreasing larger size
        self.prod_idx = list(range(len(products)))
        self.prod_idx.sort(key = lambda x : (max(self.prods[x]["size"]), min(self.prods[x]["size"])),
                            reverse = True) #!
        
        # distribute product by its quantity
        self.dist_prod_idx = self.distribute_prod(products, self.prod_idx)
        
    # to check if can cut product from stock
    def cut_from_stock(self, product_idx, stock):
        prod = self.prods[product_idx]
        prod_w, prod_h = prod["size"]
        stock_w, stock_h = self._get_stock_size_(stock)

        if prod_w > max(stock_w, stock_h) or prod_h > max(stock_w, stock_h):
            return {"can_place" : False, "position" : (-1,-1), "isRotated" : False}

        for x in range(stock_w):
            for y in range(stock_h):
                if x + prod_w <= stock_w and y + prod_h <= stock_h:
                    if self._can_place_(stock, (x,y), [prod_w, prod_h]):
                        return {"can_place" : True, "position" : (x,y), "isRotated" : False}
                if x + prod_h <= stock_w and y + prod_w <= stock_h:
                    if self._can_place_(stock, (x,y), [prod_h, prod_w]):
                        return {"can_place" : True, "position" : (x,y), "isRotated" : True}

        return {"can_place" : False, "position" : (-1,-1), "isRotated" : False}

    # to choose suitable stock to cut for product
    def choosing_stock(self, product_idx):
        # scan through used_stock_idx first
        for stock_idx in self.used_stocks_idx:

            # check for enough space
            prod_w, prod_h =  self.prods[product_idx]["size"]
            if self.available_area[stock_idx] < prod_w * prod_h:
                continue

            res_dict = self.cut_from_stock(product_idx, self.saved_stocks[stock_idx])
            can_place = res_dict["can_place"]
            position = res_dict["position"]
            isRotated = res_dict["isRotated"]

            if can_place:
                return {"stock_idx" : stock_idx, "position" : position, "isRotated" : isRotated}
        
        # if no used stock can place, scan not_used
        for not_used_stock_idx in self.not_used_stocks_idx:
            res_dict = self.cut_from_stock(product_idx, self.saved_stocks[not_used_stock_idx])
            can_place = res_dict["can_place"]
            position = res_dict["position"]
            isRotated = res_dict["isRotated"]

            if can_place:
                return {"stock_idx" : not_used_stock_idx, "position" : position, "isRotated" : isRotated}
        
        # if no stock can place, no feasiblt solution
        return {"stock_idx" : -1, "position" : (-1, -1), "isRotated" : False}
    
    # choosing stock for best fit
    def choosing_stock_best_fit(self, product_idx):
        prod_w, prod_h = self.prods[product_idx]["size"]
        best_fit_stock = None
        best_fit_info = {"can_place": False, "position": (-1, -1), "isRotated": False}
        min_wasted_area = float('inf')  # to store min waste

        # check used stock
        for stock_idx in self.used_stocks_idx:
            if self.available_area[stock_idx] >= prod_w * prod_h:
                res_dict = self.cut_from_stock(product_idx, self.saved_stocks[stock_idx])
                if res_dict["can_place"]:
                    wasted_area = self.available_area[stock_idx] - (prod_w * prod_h)
                    if wasted_area < min_wasted_area:
                        min_wasted_area = wasted_area
                        best_fit_stock = stock_idx
                        best_fit_info = res_dict

        # check not_used_stock
        for stock_idx in self.not_used_stocks_idx:
            if self.available_area[stock_idx] >= prod_w * prod_h:
                res_dict = self.cut_from_stock(product_idx, self.saved_stocks[stock_idx])
                if res_dict["can_place"]:
                    wasted_area = self.available_area[stock_idx] - (prod_w * prod_h)
                    if wasted_area < min_wasted_area:
                        min_wasted_area = wasted_area
                        best_fit_stock = stock_idx
                        best_fit_info = res_dict

        # if can find, return
        if best_fit_stock is not None:
            return {
                "stock_idx": best_fit_stock,
                "position": best_fit_info["position"],
                "isRotated": best_fit_info["isRotated"]
            }

        # if can't find
        return {"stock_idx": -1, "position": (-1, -1), "isRotated": False}
        
    # to cut a product from a stock, same as step of teacher
    #! remember to check can cut before calling this method
    def cut_step(self, position, stock_idx, prod_idx, isRotated):
        x, y = position[0], position[1]
        prod_w, prod_h = 0, 0   

        # check for isRotated
        if isRotated:
            prod_h, prod_w = self.prods[prod_idx]["size"]
        else:
            prod_w, prod_h = self.prods[prod_idx]["size"]
        
        # cut
        self.saved_stocks[stock_idx][x : x + prod_w, y : y + prod_h] = prod_idx

        # update item info of the recently cut stock
        item_details = {"index" : prod_idx, "position" : (x, y), "rotation" : isRotated}
        self.prods_in_stocks_info[stock_idx].append(item_details)

        # update area, add in used list and sort the list
        self.available_area[stock_idx] -= (prod_w * prod_h)

        if not self.is_used[stock_idx]:
            # add to used list
            self.used_stocks_idx.append(stock_idx)
            self.is_used[stock_idx] = True

            # delete from unused list and resort it
            self.not_used_stocks_idx.remove(stock_idx)
            # self.not_used_stocks_idx.sort(key = lambda x : (self.available_area[x], self.cal_stock_half_peri(self.saved_stocks[x])), reverse = True)

        # sort the used_list after cut
        self.used_stocks_idx.sort(key = lambda x : self.available_area[x], reverse = True) #!

    # perform FFD to get initial cut
    def init_cut(self):
        # determine function to choose stock
        choosing_stock_ptr = None
        if self.using_FFD:
            choosing_stock_ptr = self.choosing_stock
        else:
            choosing_stock_ptr = self.choosing_stock_best_fit

        # cut all product
        for prod_idx in self.dist_prod_idx:
            res_dict = choosing_stock_ptr(prod_idx)
            stock_idx = res_dict["stock_idx"]
            position = res_dict["position"]
            isRotated = res_dict["isRotated"]
            
            # cut if can cut
            if stock_idx != -1:
                self.cut_step(position, stock_idx, prod_idx, isRotated)
            else:
                return False # this indicate no feasible solution
        
        # have solution
        return True

    """
        This section gives function to assist optimizing feasible solution
    """

    # to try recuting list of product to stock_idx
    def recut_to_stock(self, stock_idx, list_prod_on_stock):
        # to check if item can be recut to other stock
        check_for_item_can_recut = [False] * len(list_prod_on_stock)

        # cloning stock to test for recut
        clone_stock = [stock.copy() for stock in self.saved_stocks]

        # check if can recut all item with smaller total area used
        can_recut_all = False

        recut_item = []

        # total area of item to recut
        area_of_all_item_to_recut = self.cal_stock_area(self.saved_stocks[stock_idx]) - self.available_area[stock_idx]

        # try recuting into other used stocks
        for i, item in enumerate(list_prod_on_stock):
            for other_stock_idx in self.used_stocks_idx:
                prod_w, prod_h = self.prods[item["index"]]["size"]

                if other_stock_idx != stock_idx and self.available_area[other_stock_idx] >= prod_w * prod_h:  

                    # if can cut, perform cut on clone stock and update relevant data
                    res_dict = self.cut_from_stock(item["index"], clone_stock[other_stock_idx])
                    if res_dict["can_place"]:

                        # get necessary data for cutting
                        x, y = res_dict["position"]
                        prod_w, prod_h = self.prods[item["index"]]["size"]

                        if res_dict["isRotated"]:
                            prod_h, prod_w = self.prods[item["index"]]["size"]

                        # cut on cloning stock
                        clone_stock[other_stock_idx][x : x + prod_w, y : y + prod_h] = item["index"]

                        # set flag
                        check_for_item_can_recut[i] = True

                        # add info to recut
                        recut_item.append({
                            "index": item["index"],
                            "stock_idx": other_stock_idx,
                            "position": res_dict["position"],
                            "isRotated": res_dict["isRotated"]
                        })

                        # decrease area of item recut
                        area_of_all_item_to_recut -= prod_w * prod_h

                        # go to next item
                        break

        # to store info for product which is cut on not used stock
        recut_item_to_not_used = []

        # Try recuting into not-used stocks
        for not_used_stock_idx in self.not_used_stocks_idx:
            # just recut to stock with smaller area than the old one
            if self.cal_stock_area(self.saved_stocks[not_used_stock_idx]) >= self.cal_stock_area(self.saved_stocks[stock_idx]):
                break

            # just recut to stock with enough area to store all item
            if self.cal_stock_area(self.saved_stocks[not_used_stock_idx]) <= area_of_all_item_to_recut:
                continue

            for i, item in enumerate(list_prod_on_stock):
                # if not packed to other used_stock, pack to new stock
                if not check_for_item_can_recut[i]:

                    # if can cut, perform cut on clone stock and update relevant data
                    res_dict = self.cut_from_stock(item["index"], clone_stock[not_used_stock_idx])
                    if res_dict["can_place"]:

                        # get necessary data
                        x, y = res_dict["position"]
                        prod_w, prod_h = self.prods[item["index"]]["size"]

                        if res_dict["isRotated"]:
                            prod_h, prod_w = self.prods[item["index"]]["size"]

                        # cut on cloning stock
                        clone_stock[not_used_stock_idx][x : x + prod_w, y : y + prod_h] = item["index"]

                        # add neccessary info to recut
                        recut_item_to_not_used.append({
                            "index": item["index"],
                            "stock_idx": not_used_stock_idx,
                            "position": res_dict["position"],
                            "isRotated": res_dict["isRotated"]
                        })

                        # to check if all item can recut to this new stock
                        if i == len(list_prod_on_stock) - 1:
                            can_recut_all = True
                            
                    else: 
                        # if any item can't pack to a stock, reset that stock, recut info and try another stock
                        clone_stock[not_used_stock_idx][clone_stock[not_used_stock_idx] != -2] = -1
                        recut_item_to_not_used.clear()
                        break
            
            if can_recut_all:
                break
        
        # just true when can recut all (can optimize solution)
        if can_recut_all:
            recut_item = recut_item + recut_item_to_not_used
            return {"can_recut": True, "recut_item": recut_item}

        return{"can_recut" : False, "recut_item" : []}

    # do recut from old stock to other stock
    def do_recut(self, old_stock_idx, recut_item):
        # remove used stock from used list and clear all item placed on it
        self.is_used[old_stock_idx] = False
        self.available_area[old_stock_idx] = self.cal_stock_area(self.saved_stocks[old_stock_idx])
        self.used_stocks_idx.remove(old_stock_idx)
        self.saved_stocks[old_stock_idx][self.saved_stocks[old_stock_idx] != -2] = -1
        self.prods_in_stocks_info[old_stock_idx].clear()

        # loop through and recut item
        for item_dict in recut_item:
            stock_idx = item_dict["stock_idx"]
            item_idx = item_dict["index"]
            position = item_dict["position"]
            isRotated = item_dict["isRotated"]

            # perform cut, validity of the cut is verified in recut_to_stock already
            self.cut_step(position, stock_idx, item_idx, isRotated)

            """
                Don't need to check if new stock is used or not because cut_step will do that :D
            """


    # to optimize a feasible solution by FFD
    def optimize_FFD(self):
        # to check if a stage is optimize or not
        is_optimize = False

        # sort by increasing area to optimize
        self.not_used_stocks_idx.sort(key = lambda x : (self.available_area[x], self.cal_stock_half_peri(self.saved_stocks[x])))

        # sort used_stock by increasing total item area on that stock
        # self.used_stocks_idx.sort(key = lambda x : (self.cal_stock_area(self.saved_stocks[x]) - self.available_area[x]))
        self.used_stocks_idx.sort(key = lambda x : self.available_area[x])

        for used_stock_idx in self.used_stocks_idx:
            # try recut item from stock to other stock
            res_dict = self.recut_to_stock(used_stock_idx, self.prods_in_stocks_info[used_stock_idx])
            can_recut = res_dict["can_recut"]
            
            # if can recut, recut :D
            if can_recut:
                is_optimize = True
                self.do_recut(used_stock_idx, res_dict["recut_item"])            

        return is_optimize
    
    # to get the suitable action
    def get_action_after_all(self):
        while self.current_stock_idx < len(self.used_stocks_idx):
            stock_actual_idx = self.used_stocks_idx[self.current_stock_idx]
            stock_items = self.prods_in_stocks_info[stock_actual_idx]

            # If the current product index is within range of the stock items
            if self.current_product_idx < len(stock_items):
                item = stock_items[self.current_product_idx]
                self.current_product_idx += 1

                # Get the size of the product
                size = self.prods[item["index"]]["size"]

                # If rotation == True, swap the dimensions
                if item["rotation"]:
                    size = [size[1], size[0]]

                return {
                    "stock_idx": stock_actual_idx,
                    "size": size,
                    "position": item["position"]
                }

            # If all products in the current stock are processed, move to the next stock
            self.current_stock_idx += 1
            self.current_product_idx = 0

        # If all stocks and products are processed, reset all param and return default action
        self.first_action = True    # to check if first call get_action
        self.saved_stocks = []      # to save stock
        self.first_action_stocks = [] # to indicate first action
        self.prods = []             # to save product

        # for easier manage and sort the 2 list above
        self.used_stocks_idx = []
        self.not_used_stocks_idx = []
        self.is_used = []
        self.available_area = []
        self.prod_idx = []
        self.dist_prod_idx = []

        # to save info of product in stock
        self.prods_in_stocks_info = [] # info của các product trong stock

        # variables for tracking state
        self.current_stock_idx = 0
        self.current_product_idx = 0

        return {"stock_idx": -1, "size": [-1, -1], "position": (-1, -1)}
