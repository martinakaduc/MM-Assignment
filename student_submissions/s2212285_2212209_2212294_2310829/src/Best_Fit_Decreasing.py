from policy import Policy
import numpy as np

class Best_Fit_Decreasing(Policy):
    def __init__(self):
        self.sorted_list_prod = []
        self.list_area_stocks = []

    def get_action(self, observation, info):
        list_prods = observation["products"]
        if not self.sorted_list_prod: 
            self.sorted_list_prod = sorted(
                list_prods,
                key=lambda prod: prod["size"][0] * prod["size"][1],
                reverse=True
            )
        if not self.list_area_stocks:
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                self.list_area_stocks.extend([(i, stock, stock_w*stock_h)])
        self.list_area_stocks = sorted(
            self.list_area_stocks,
            key=lambda prod: prod[2],
            reverse=False
        )
        best_action = None
        index = 0
        for prod in self.sorted_list_prod:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                for i,stock, area in self.list_area_stocks:
                    if area >= prod_size[0] * prod_size[1]:
                        for size in [prod_size, prod_size[::-1]]:  # Kiểm tra xoay 90 độ
                            pos_x, pos_y = self._find_position(stock, size)
                            if pos_x is not None and pos_y is not None:
                                best_action = {
                                    "stock_idx": i,
                                    "size": size,
                                    "position": (pos_x, pos_y),
                                }
                                self.list_area_stocks.remove((i,stock,area))
                                self.list_area_stocks.extend([(i,stock, area - prod_size[0]*prod_size[1])])
                                if index == len(self.sorted_list_prod) - 1 and prod["quantity"] == 1:
                                    print("DONE")
                                    self.list_area_stocks = []
                                    self.sorted_list_prod = []
                                break
                    if best_action is not None:
                        break
                if best_action is not None:
                    break                                         
            index+=1
        return best_action

    def _find_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y

        return None, None