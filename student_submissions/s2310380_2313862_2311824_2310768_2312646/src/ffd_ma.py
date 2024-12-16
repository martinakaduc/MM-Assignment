from policy import Policy
import numpy as np

class FFD_MA(Policy):
    def __init__(self):
        pass
    
    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        products = [prod for prod in list_prods if prod["quantity"] > 0]
        if not products:
            return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}
            
        products.sort(key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        for prod in products:
            prod_size = prod["size"]
            best_fit_stock_idx = -1
            best_pos_x, best_pos_y = None, None

            for attempt in range(len(stocks)): 
                for i, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                        if pos_x is not None and pos_y is not None:
                            best_fit_stock_idx = i
                            best_pos_x, best_pos_y = pos_x, pos_y
                            break

                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                        if pos_x is not None and pos_y is not None:
                            best_fit_stock_idx = i
                            best_pos_x, best_pos_y = pos_x, pos_y
                            break

                if best_pos_x is not None and best_pos_y is not None:
                    stock_idx = best_fit_stock_idx
                    pos_x, pos_y = best_pos_x, best_pos_y
                    break

            if pos_x is not None and pos_y is not None:
                break
                
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}