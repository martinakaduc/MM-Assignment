from policy import Policy
import numpy as np

class FFDPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1], reverse=True)
 
        prodsize = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prodsize = prod["size"]

                for idx, stock in sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if (stock_w >= prod_w and stock_h >= prod_h) or (stock_w >= prod_h and stock_h >= prod_w):
                        pos_x, pos_y = None, None
                        for x in range(max(stock_w - prod_w + 1, stock_w - prod_h + 1)):
                            for y in range(max(stock_h - prod_h + 1, stock_h - prod_w + 1)):
                                if x < stock_w - prod_w + 1 and y < stock_h - prod_h + 1 and self._can_place_(stock, (x, y), prod_size):
                                    prodsize = prod_size
                                    pos_x, pos_y = x, y
                                    break
                                elif x < stock_w - prod_h + 1 and y < stock_h - prod_w + 1 and self._can_place_(stock, (x, y), prod_size[::-1]):
                                    prodsize = prod_size[::-1]
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = idx
                            break

                    # if stock_w >= prod_h and stock_h >= prod_w:
                    #     pos_x, pos_y = None, None
                    #     for x in range(stock_w - prod_h + 1):
                    #         for y in range(stock_h - prod_w + 1):
                    #             if self._can_place_(stock, (x, y), prod_size[::-1]):
                    #                 prod_size = prod_size[::-1]
                    #                 pos_x, pos_y = x, y
                    #                 break
                    #         if pos_x is not None and pos_y is not None:
                    #             break
                    #     if pos_x is not None and pos_y is not None:
                    #         stock_idx = idx
                    #         break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prodsize, "position": (pos_x, pos_y)}
    
    
    
class BFDPolicy(Policy):
    def __init__(self):
        if not hasattr(self, "past"):
            self.past = -1

    def get_action(self, observation, info):
        list_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        stocks = observation["stocks"]

        prodsize = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prodsize = prod["size"]
                best_fit = float('inf')
                
                if self.past != -1 : 
                    pos_x, pos_y = None, None
                    stock = stocks[self.past]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if (stock_w >= prod_w and stock_h >= prod_h) or (stock_w >= prod_h and stock_h >= prod_w):
                        
                        for x in range(max(stock_w - prod_w + 1, stock_w - prod_h + 1)):
                            for y in range(max(stock_h - prod_h + 1, stock_h - prod_w + 1)):
                                if x < stock_w - prod_w + 1 and y < stock_h - prod_h + 1 and self._can_place_(stock, (x, y), prod_size):
                                    remaining_space = np.sum(stock != -2) - prod_w * prod_h
                                    if remaining_space < best_fit:
                                        best_fit = remaining_space
                                        stock_idx = self.past
                                        pos_x, pos_y = x, y
                                        prodsize = prod_size
                                if x < stock_w - prod_h + 1 and y < stock_h - prod_w + 1 and self._can_place_(stock, (x, y), prod_size[::-1]):
                                    remaining_space = np.sum(stock != -2) - prod_w * prod_h
                                    if remaining_space < best_fit:
                                        best_fit = remaining_space
                                        stock_idx = self.past
                                        pos_x, pos_y = x, y
                                        prodsize = prod_size[::-1]
                    if pos_x is not None and pos_y is not None:
                        break
                
                for i, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if (stock_w >= prod_w and stock_h >= prod_h) or (stock_w >= prod_h and stock_h >= prod_w):
                        for x in range(max(stock_w - prod_w + 1, stock_w - prod_h + 1)):
                            for y in range(max(stock_h - prod_h + 1, stock_h - prod_w + 1)):
                                if x < stock_w - prod_w + 1 and y < stock_h - prod_h + 1 and self._can_place_(stock, (x, y), prod_size):
                                    remaining_space = np.sum(stock != -2) - prod_w * prod_h
                                    if remaining_space < best_fit:
                                        best_fit = remaining_space
                                        stock_idx = i
                                        pos_x, pos_y = x, y
                                        prodsize = prod_size
                                if x < stock_w - prod_h + 1 and y < stock_h - prod_w + 1 and self._can_place_(stock, (x, y), prod_size[::-1]):
                                    remaining_space = np.sum(stock != -2) - prod_w * prod_h
                                    if remaining_space < best_fit:
                                        best_fit = remaining_space
                                        stock_idx = i
                                        pos_x, pos_y = x, y
                                        prodsize = prod_size[::-1]

                if stock_idx != -1:
                    break
        self.past = stock_idx
        return {"stock_idx": stock_idx, "size": prodsize, "position": (pos_x, pos_y)}
    

class Policy2352113_2352103_2352740_2352570_2352442(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = FFDPolicy()
        elif policy_id == 2:
            self.policy = BFDPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed