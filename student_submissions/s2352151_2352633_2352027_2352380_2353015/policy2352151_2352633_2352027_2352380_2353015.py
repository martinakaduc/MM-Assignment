import numpy as np
from policy import Policy
import random

class Policy2352151_2352633_2352027_2352380_2353015(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = BFDPolicy()
        elif policy_id == 2:
            self.policy = FFDPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

class FFDPolicy(Policy):
    
    def __init__(self):
        pass
    def get_action(self, observation, info):
        list_prods = sorted(
            observation["products"],
            key=lambda prod: (-prod["size"][0] * prod["size"][1], prod["quantity"]),
        )

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        sorted_stocks = sorted(
            observation["stocks"],
            key=lambda stock: self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1],
            reverse=True
        )

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for i, stock in enumerate(sorted_stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:                             
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    stock_idx = next((idx for idx, s in enumerate(observation["stocks"]) if np.array_equal(s, stock)), -1)
                                    pos_x, pos_y = x, y
                                    break
                            if stock_idx != -1:
                                break
                        if stock_idx != -1:
                            break

                    if stock_w >= prod_h and stock_h >= prod_w:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    stock_idx = next((idx for idx, s in enumerate(observation["stocks"]) if np.array_equal(s, stock)), -1)
                                    pos_x, pos_y = x, y
                                    prod_size = prod_size[::-1]  
                                    break
                            if stock_idx != -1:
                                break
                        if stock_idx != -1:
                            break
                if stock_idx != -1:
                    break

        # Example usage of stock_idx
        print(f"Chỉ số kho được chọn: {stock_idx}")

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


class BFDPolicy(Policy):

    def __init__(self):
        pass
    def get_action(self, observation, info):
        list_prods = sorted(
            observation["products"],
            key=lambda prod: (-prod["size"][0] * prod["size"][1], prod["quantity"]),
        )

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                best_fit_stock_idx = -1
                best_fit_position = None
                min_waste = float('inf')

                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    waste = (stock_w * stock_h) - (prod_w * prod_h)
                                    if waste < min_waste:
                                        min_waste = waste
                                        best_fit_stock_idx = i
                                        best_fit_position = (x, y)

                    if stock_w >= prod_h and stock_h >= prod_w:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    waste = (stock_w * stock_h) - (prod_h * prod_w)
                                    if waste < min_waste:
                                        min_waste = waste
                                        best_fit_stock_idx = i
                                        best_fit_position = (x, y)
                                        prod_size = prod_size[::-1]  

                if best_fit_position is not None:
                    stock_idx = best_fit_stock_idx
                    pos_x, pos_y = best_fit_position
                    break

        print(f"Chỉ số kho được chọn: {stock_idx}")

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
    # Student code here
    # You can add more functions if needed