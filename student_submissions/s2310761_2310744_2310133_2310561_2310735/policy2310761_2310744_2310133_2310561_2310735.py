from policy import Policy
import numpy as np

class ProdObj:
    def __init__(self, insize, indemand):
        self.size = insize      # list of 2 int
        self.demand = indemand  # int
    def __init__(self, indict): # indict is dict
        self.size = indict["size"]
        self.demand = indict["quantity"]
    def __lt__(self, other):
        return (self.size[0] * self.size[1] < other.size[0] * other.size[1])

class StockObj:
    def __init__(self, instock, original_index): # instock is numpy array
        self.arr = instock
        self.height = np.sum(np.any(instock != -2, axis=0))
        self.width = np.sum(np.any(instock != -2, axis=1))
        self.rmarea = self.height * self.width
        self.orgidx = original_index
    def __lt__(self, other):
        return (self.rmarea < other.rmarea)

class Policy2310761_2310744_2310133_2310561_2310735(Policy):    
    def __init__(self, policy_id):
        # initialize parameters for policy #1
        self.used_stocks = list()    # used stocks are kept in ascending order
        self.unused_stocks = list()  # while unused stocks are kept in descending order
        self.previous_rmarea = None # this var is used to store the prev remaining area of the first stock for the reset cond
        self.fstock = None  # this var is used to track the first used stock
        self.policy_id = policy_id
    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return FirstFitDecreasing(self, observation)
        elif self.policy_id == 2:
            return BestFitDecreasing(self, observation)

    # Student code here
    # You can add more functions if needed
def BestFitDecreasing(s, observation):
    # PREPARING DATA

    # Preparing prods

    list_prods = list() # list of list_objs
    for prod in observation["products"]:    # for (dict) in (tuple of dicts)
        prod_obj = ProdObj(prod)
        list_prods.append(prod_obj)
    list_prods.sort(reverse=True)   # sort the prods in descending order

    prod_size = [0, 0]
    stock_idx = -1
    pos_x, pos_y = 0, 0

    # Preparing stocks
    enum_stocks = enumerate(observation["stocks"])
    # these actions are executed only once
    if s.fstock == None:
        for i, stock in enum_stocks:
            stock_obj = StockObj(stock, i)
            s.unused_stocks.append(stock_obj)
        s.unused_stocks.sort(reverse=True)
        s.fstock = s.unused_stocks[0]
        #print("First stock's orgidx = ", s.fstock.orgidx)
    else:
        # DETECTING ENV RESET
        fidx = s.fstock.orgidx
        observed_stock = None
        for i, stock in enum_stocks:
            if i==fidx:
                observed_stock = stock
                break
        #print("Current FS rmarea = ", s.fstock.rmarea)
        if (observed_stock[0,0] == -1):
            # reset everything
            s.used_stocks.clear()
            s.unused_stocks.clear()
            s.fstock = None
            s.previous_rmarea = None
            #print("----- ENVIRONMENT RESET DETECTED! ----------------")
            enum_stocks = enumerate(observation["stocks"])
            # reset enumerate so counter i starts from 0
            for i, stock in enum_stocks:
                stock_obj = StockObj(stock, i)
                s.unused_stocks.append(stock_obj)
            s.unused_stocks.sort(reverse=True)
            s.fstock = s.unused_stocks[0]
            #print("First stock's orgidx = ", s.fstock.orgidx)

    # ACTUAL PLACING PRODUCTS INTO STOCKS

    # Ensure product has demand > 0
    for prod in list_prods:
        if prod.demand > 0:
            prod_size = prod.size
            # Iterate through used stocks
            pos_x = None
            pos_y = None
            best_fit_area = 10001
            best_fit_idx = -1
            for stock in s.used_stocks:
                stock_w = stock.width
                stock_h = stock.height
                prod_w, prod_h = prod.size
                if stock_w >= prod_w and stock_h >= prod_h:
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if s._can_place_(stock.arr, (x, y), prod_size) and stock.rmarea < best_fit_area:
                                best_fit_area = stock.rmarea
                                best_fit_idx = stock.orgidx
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is None and pos_y is None:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if s._can_place_(stock.arr, (x, y), prod_size[::-1]) and stock.rmarea < best_fit_area:
                                    prod_size = prod_size[::-1]
                                    best_fit_area = stock.rmarea
                                    best_fit_idx = stock.orgidx
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                    
                    if pos_x is not None and pos_y is not None:
                        stock_idx = best_fit_idx
                        stock.rmarea -= prod_w * prod_h
                        #s.used_stocks.sort()
                        break
            if pos_x is not None and pos_y is not None:
                # if stock is placed, stop
                break
            else:
                # if not, use a new stock, place the product into it and update rmarea
                new_stock = s.unused_stocks.pop(0)
                stock_idx = new_stock.orgidx
                pos_x = 0
                pos_y = 0
                s.used_stocks.append(new_stock)
                #s.used_stocks.sort()

    return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

def FirstFitDecreasing(s, observation):
    # prepare prods
    list_prods = list() # list of list_objs
    for prod in observation["products"]:    # for (dict) in (tuple of dicts)
        prod_obj = ProdObj(prod)
        list_prods.append(prod_obj)
    list_prods.sort(reverse=True)   # sort the prods in descending order

    prod_size = [0, 0]
    stock_idx = -1
    pos_x, pos_y = 0, 0

    # Pick a product that has quality > 0
    for prod in list_prods:
        if prod.demand > 0:
            prod_size = prod.size

            # Loop through all stocks
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = s._get_stock_size_(stock)
                prod_w, prod_h = prod_size
                if stock_w >= prod_w and stock_h >= prod_h:
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if s._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                if stock_w >= prod_h and stock_h >= prod_w:
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if s._can_place_(stock, (x, y), prod_size[::-1]):
                                prod_size = prod_size[::-1]
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

            if pos_x is not None and pos_y is not None:
                break

    return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}