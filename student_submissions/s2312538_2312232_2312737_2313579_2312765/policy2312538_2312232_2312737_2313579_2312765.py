from policy import Policy
import numpy as np

class Policy2312538_2312232_2312737_2313579_2312765(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = FirstFit()
        elif policy_id == 2:
            self.policy = ColumnGeneration()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

class FirstFit(Policy):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.s_stocks = []
        self.curr_prod_idx = 0  
        self.all_placed_prod = None  
    
    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        if self.all_placed_prod is None:
            flatten_products = self._generate_products_(list_prods)
            self._initialize_s_stocks_(stocks)
            self._optimize_s_stocks_(flatten_products)

            self.all_placed_prod = [
                (stock["stock_idx"] - 1, pos_x, pos_y, prod_w, prod_h)
                for stock in self.s_stocks
                for (pos_x, pos_y, prod_w, prod_h) in stock["placed_prod"]
            ]

        if self.curr_prod_idx >= len(self.all_placed_prod):
            self.reset()
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        stock_idx, pos_x, pos_y, prod_w, prod_h = self.all_placed_prod[self.curr_prod_idx]
        self.curr_prod_idx += 1  

        return {"stock_idx": stock_idx, "size": [prod_w, prod_h], "position": (pos_x, pos_y)}

    def _generate_products_(self, list_prods):
        flatten_products = []
        for prod in list_prods:
            if prod["quantity"] > 0:
                for _ in range(prod["quantity"]):
                    flatten_products.append((prod["size"][0], prod["size"][1]))     # flatten the list of products with their quantity
                    
        return flatten_products

    def _initialize_s_stocks_(self, stocks):
        self.s_stocks = []
        for idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            self.s_stocks.append({
                "stock_idx": idx + 1,
                "stock_w": stock_w,    
                "stock_h": stock_h,
                "free_spaces": [(0, 0, stock_w, stock_h)],
                "placed_prod": []
            })

    def _optimize_s_stocks_(self, product):
        for prod in sorted(product, key=lambda p: p[0] * p[1], reverse=True):
            placed = False
            for stock in self.s_stocks:
                if self._place_prod_(stock, prod):
                    placed = True
                    break
            if not placed:
                new_stock = {
                    "stock_idx": len(self.s_stocks) + 1,
                    "stock_w": self.s_stocks[0]["stock_w"], 
                    "stock_h": self.s_stocks[0]["stock_h"],
                    "free_spaces": [(0, 0, self.s_stocks[0]["stock_w"], self.s_stocks[0]["stock_h"])],
                    "placed_prod": []
                }

                self._place_prod_(new_stock, prod)
                self.s_stocks.append(new_stock)
    
    def _place_prod_(self, stock, prod):
        prod_w, prod_h = prod
        for i, (pos_x, pos_y, stock_w, stock_h) in enumerate(stock["free_spaces"]):
            if prod_h <= stock_w and prod_w <= stock_h:
                self._split_space_(stock, i, pos_x, pos_y, stock_w, stock_h, prod_h, prod_w)
                stock["placed_prod"].append((pos_x, pos_y, prod_h, prod_w))
                return True
            if prod_w <= stock_w and prod_h <= stock_h:
                self._split_space_(stock, i, pos_x, pos_y, stock_w, stock_h, prod_w, prod_h)
                stock["placed_prod"].append((pos_x, pos_y, prod_w, prod_h))
                return True
                
        return False

    def _split_space_(self, stock, idx, x, y, w, h, prod_width, prod_height):
        stock["free_spaces"].pop(idx)
        if prod_width < w:  
            stock["free_spaces"].append((x + prod_width, y, w - prod_width, prod_height))
        if prod_height < h:  
            stock["free_spaces"].append((x, y + prod_height, w, h - prod_height))

class ColumnGeneration(Policy):
    def __init__(self):
        self.solutions = []

    def _calculate_(self, stock, position, prod):
        W, H = self._get_stock_size_(stock)
        copy_of_stock = stock.copy()
        x, y = position
        w, h = prod
        for i in range(w):
            np.put(copy_of_stock[x + i], range(y, y + h), -2)
        np.place(copy_of_stock, copy_of_stock != -1, 0)
        area = np.count_nonzero(copy_of_stock[:W, y + h:H])
        area += np.count_nonzero(copy_of_stock[x+w:W,:H-y])
        return area
    
    def _columns_generation_(self, observation):
        generation = []
        sorted_cost_prods = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True)
        quantity = [prod['quantity'] for prod in sorted_cost_prods]
        
        for stock in observation["stocks"]:
            stock_w, stock_h = self._get_stock_size_(stock)
            idx = 0
            while idx < len(sorted_cost_prods):
                prod = sorted_cost_prods[idx]
                prod_w, prod_h = prod["size"]
                pos_x1, pos_y1 = None, None
                value1 =  0
                exist_x1y1 = False
                exist_x2y2 = False
                if quantity[idx] > 0:
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x1, pos_y1 = self._product_position_(stock, (prod_w, prod_h))
                        if pos_x1 is not None and pos_y1 is not None:
                            exist_x1y1 = True
                            value1 = self._calculate_(stock, (pos_x1, pos_y1), prod["size"])
                            generation.append((prod["size"], stock, pos_x1, pos_y1))
                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x2, pos_y2 = self._product_position_(stock, (prod_h, prod_w))
                        if pos_x2 is not None and pos_y2 is not None:
                            exist_x2y2 = True
                            if exist_x1y1:
                                value2 = self._calculate_(stock, (pos_x2, pos_y2), [prod_h, prod_w])
                                if(value1 < value2):
                                    prod["size"] = prod["size"][::-1]
                                    generation.pop()
                                    generation.append((prod["size"], stock, pos_x2, pos_y2))
                                
                            else:
                                prod["size"] = prod["size"][::-1]
                                generation.append((prod["size"], stock, pos_x2, pos_y2))
                        elif not exist_x1y1:
                            idx += 1
                    elif not exist_x1y1:
                        idx += 1       
                    if exist_x1y1 or exist_x2y2:
                        if quantity[idx] == 1:
                            sorted_cost_prods.pop(idx)
                            quantity.pop(idx)
                        else:
                            quantity[idx] -= 1
                else:
                    idx += 1
            if not len(sorted_cost_prods):
                break
        return generation

    def _product_position_(self, stock, prod):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod):
                    return x, y
        return None, None

    def get_action(self, observation, info):
        self.solutions = self._columns_generation_(observation)
        
        if self.solutions is None:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        prod, stock, pos_x, pos_y = self.solutions[0]

        stock_idx = -1
        stock_sizes = [self._get_stock_size_(s) for s in observation["stocks"]]
        target_stock_size = self._get_stock_size_(stock)
        
        if target_stock_size in stock_sizes:
            stock_idx = stock_sizes.index(target_stock_size)

        return {"stock_idx": stock_idx, "size": prod, "position": (pos_x, pos_y)}

