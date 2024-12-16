from policy import Policy
import numpy as np
import random


class Policy2314047_2311767_2310259_2313215_2313052(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.cur_stock_idx = -1
            self.solution = []
            self.loop = 250
            self.stop_condition = 0.05
            self.num_of_quantity = 0
        elif policy_id == 2:
            self.sorted_stock = None
            self.sorted_prod = None
            self.cur_stock_idx = -1
            self.cur_prod_idx = -1

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            if info["filled_ratio"] == 0.0:
                self.cur_stock_idx = -1
                self.solution = []

            sorted_stock = sorted(
                enumerate(observation["stocks"]),
                key=lambda s: self._get_stock_size_(s[1])[0] * self._get_stock_size_(s[1])[1],
                reverse=True,
            )

            if self.solution:
                return self.solution.pop(0)
            else:
                self.cur_stock_idx += 1
                idx, stock = sorted_stock[self.cur_stock_idx]
                self.solution = self.find_solution(stock.copy(), idx, observation["products"])
                return self.solution.pop(0)
                
        elif self.policy_id == 2:
            if info["filled_ratio"] == 0.0:
                self.cur_stock_idx = -1
                self.cur_prod_idx = -1
                

            sorted_prod = sorted(enumerate(observation["products"]), 
                                    key=lambda p: p[1]["size"][0] * p[1]["size"][1], 
                                    reverse=True)

            sorted_stock = sorted(enumerate(observation["stocks"]),
                                    key=lambda s: self._get_stock_size_(s[1])[0]
                                     * self._get_stock_size_(s[1])[1]
                                     ,
                                    reverse=True)
            
            if self.cur_stock_idx != -1 and self.cur_prod_idx != -1:
                stock = observation["stocks"][self.cur_stock_idx]
                prod = observation["products"][self.cur_prod_idx]
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    if (prod_w <= stock_w and prod_h <= stock_h):
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {"stock_idx": self.cur_stock_idx,
                                            "size": prod_size,
                                            "position": (x, y)} 
                    if (prod_w <= stock_h and prod_h <= stock_w):
                        prod_size = [prod_h, prod_w]
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {"stock_idx": self.cur_stock_idx,
                                            "size": prod_size,
                                            "position": (x, y)}
            
            for idx, stock in sorted_stock:
                for pidx, prod in sorted_prod:
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size
                        
                        if (prod_w <= stock_w and prod_h <= stock_h):
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        self.cur_prod_idx = pidx
                                        self.cur_stock_idx = idx
                                        return {"stock_idx": idx,
                                                "size": prod_size,
                                                "position": (x, y)}     
                        if (prod_w <= stock_h and prod_h <= stock_w):
                            prod_size = [prod_h, prod_w]
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        self.cur_prod_idx = pidx
                                        self.cur_stock_idx = idx
                                        return {"stock_idx": idx,
                                                "size": prod_size,
                                                "position": (x, y)}

            self.cur_stock_idx = -1
            self.cur_prod_idx = -1
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    def calculate_trim_loss(self, c_stock):
        total_area = np.sum(c_stock >= -1)
        trim_loss = np.sum(c_stock == -1)
        return trim_loss / total_area if total_area > 0 else 0.0

    def find_solution(self, stock, stock_idx, list_prod):
        best_solution = []
        best_trim_loss = self.calculate_trim_loss(stock)

        for _ in range(self.loop):

            c_list_prod = [prod.copy() for prod in list_prod]
            c_stock = stock.copy()
            new_solution = []

            while True:
                has_placement = False
                for __ in range(100):
                    prod_idx = random.randint(0, len(c_list_prod) - 1)
                    prod = c_list_prod[prod_idx]
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        stock_w, stock_h = self._get_stock_size_(c_stock)
                        prod_w, prod_h = prod_size

                        if stock_w >= prod_w and stock_h >= prod_h:
                            for _ in range(10):
                                pos_x = random.randint(0, stock_w - prod_w)
                                pos_y = random.randint(0, stock_h - prod_h)
                                if self._can_place_(c_stock, (pos_x, pos_y), prod_size):
                                    new_solution.append({"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)})
                                    c_list_prod[prod_idx]["quantity"] -= 1
                                    c_stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] = 0
                                    has_placement = True
                                    break
                        if stock_w >= prod_h and stock_h >= prod_w:
                            for _ in range(10):
                                pos_x = random.randint(0, stock_w - prod_h)
                                pos_y = random.randint(0, stock_h - prod_w)
                                if self._can_place_(c_stock, (pos_x, pos_y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    new_solution.append({"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)})
                                    c_list_prod[prod_idx]["quantity"] -= 1
                                    c_stock[pos_x : pos_x + prod_h, pos_y : pos_y + prod_w] = 0
                                    has_placement = True
                                    break

                        if has_placement:
                            break

                if not has_placement:  
                    break

            new_trim_loss = self.calculate_trim_loss(c_stock)
            if new_trim_loss < best_trim_loss:
                best_solution = new_solution
                best_trim_loss = new_trim_loss
            if best_trim_loss < self.stop_condition:
                break

        return best_solution
