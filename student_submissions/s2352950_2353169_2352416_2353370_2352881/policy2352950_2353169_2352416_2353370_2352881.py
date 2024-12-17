from policy import Policy
import random
import numpy as np

class Policy2352950_2353169_2352416_2353370_2352881(Policy):
    def __init__(self, policy_id=1):
        super().__init__()
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.init_demand = 0
        
    def find_potential_stocks(self, stocks: list, prod_size: tuple) -> list:
        potential_stocks = []
        for i, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            if stock_w < prod_size[0] or stock_h < prod_size[1]:
                continue
            potential_stocks.append((i, stock, self._calculate_potential_waste_for_stock(stock, prod_size)))
        potential_stocks.sort(key=lambda x: x[2])
        return potential_stocks
    
    def list_used_stock(self, stocks):
        used_stock = []
        for i, stock in enumerate(stocks):
            w,h = self._get_stock_size_(stock)
            waste = self.calculate_waste(stock)
            if waste < w*h:
                used_stock.append((i, stock))
        return used_stock
    
    def _calculate_potential_waste_for_stock(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        min_waste = float('inf')
        for x in range(stock_w - prod_size[0] + 1):
            for y in range(stock_h - prod_size[1] + 1):
                position = (x, y)
                if self._can_place_(stock, position, prod_size):
                    temp_stock = stock.copy()
                    temp_stock[x:x+prod_size[0], y:y+prod_size[1]] = 1
                    waste = self.calculate_waste(temp_stock)
                    min_waste = waste
                    break
        return min_waste
    
    def count_demand(self, products):
        count = 0
        for p in products:
            count += p["quantity"]
        return count

    def calculate_waste(self, stock):
        return np.count_nonzero(stock == -1)

    def simulated_annealing(self, stock, prod_size, products):
        T = 10000
        T_min = 0.1
        alpha = 0.912
        max_iter = 1000
        stock_w, stock_h = self._get_stock_size_(stock)
        item_w, item_h = prod_size
        
        initial_position = self._find_initial_position(stock, prod_size)
        if initial_position is None:
            return None
        
        current_stock = stock.copy()
        x, y = initial_position
        if self._can_place_(current_stock, (x, y), prod_size):
            current_stock[x:x+item_w, y:y+item_h] = 1
        else:
            current_stock[x:x+item_h, y:y+item_w] = 1
            item_w, item_h = item_h, item_w
        
        current_waste = self.calculate_waste(current_stock)
        best_stock = current_stock.copy()
        best_waste = current_waste
        best_position = (x, y)

        iteration = 0
        while T > T_min and iteration < max_iter:
            new_w, new_h = self._generate_random_product(stock, prod_size, (x, y), products)
            if new_w is not None and new_h is not None:
                new_stock = stock.copy()
                new_x = x
                new_y = y
                new_stock_new = new_stock.copy()
                new_stock_new[new_x:new_x+new_w, new_y:new_y+new_h] = 1
                attempt = 0
                while not self._can_place_(new_stock, (new_x, new_y), (new_w, new_h)):
                    new_x = x
                    new_y = y
                    new_stock_new = new_stock.copy()
                    new_stock_new[new_x:new_x+new_w, new_y:new_y+new_h] = 1
                    attempt += 1
                    if attempt > 100:
                        break
                if attempt <= 100:
                    new_stock[new_x:new_x+new_w, new_y:new_y+new_h] = 1
                    new_waste = self.calculate_waste(new_stock)
                    delta = new_waste - current_waste
                    if delta < 0 or np.exp(-delta / T) > random.random():
                        item_w, item_h = new_w, new_h
                        current_stock = new_stock
                        current_waste = new_waste
                        best_position = (new_x, new_y)
                        if new_waste < best_waste:
                            best_waste = new_waste
                            best_stock = new_stock.copy()
            T *= alpha
            iteration += 1

        if best_position is not None:
            return {"position": best_position, "size": (item_w, item_h)}
        else:
            return None

    def _find_initial_position(self, stock, size):
        stock_w, stock_h = self._get_stock_size_(stock)
        item_w, item_h = size
        for x in range(stock_w - item_w + 1):
            for y in range(stock_h - item_h + 1):
                if self._can_place_(stock, (x, y), size):
                    return (x, y)
                if self._can_place_(stock, (x, y), (item_h, item_w)):
                    return (x, y)
        return None
    
    def array_product_remain(self, products):
        remain = []
        for p in products:
            if p["quantity"] > 0:
                remain.append(p)
        return remain
    
    def _generate_random_product(self, stock, size, current_position, products):
        stock_w, stock_h = self._get_stock_size_(stock)
        x, y = current_position
        r_products = self.array_product_remain(products)
        i = random.randint(0, len(r_products) - 1)
        size = r_products[i]
        if size["quantity"] == 0:
            return None, None

        if random.randint(0, 1) == 1:
            size["size"] = (size["size"][1], size["size"][0])
            if not self._can_place_(stock, (x, y), size["size"]):
                size["size"] = (size["size"][1], size["size"][0])

        if not self._can_place_(stock, (x, y), size["size"]):
            return None, None
        return size["size"]
    
    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w,stock_h
    
    def _can_place_(self, stock, position, size):
        pos_x, pos_y = position
        prod_w, prod_h = size
        stock_w, stock_h = self._get_stock_size_(stock)
        if pos_x + prod_w > stock_w or pos_y + prod_h > stock_h:
            return False
        return np.all(stock[pos_x:pos_x+prod_w, pos_y:pos_y+prod_h] == -1)

    def _find_bottom_left_position(self, stock, size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = size

        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), size):
                    return x, y

        return None
    
    def _find_best_cut(self, stock, product, products):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = product["size"]
        best_position = None
        best_orientation = None
        min_trim_loss = float('inf')

        for rotated_size in [product["size"], product["size"][::-1]]:
            r_w, r_h = rotated_size
            if stock_w < r_w or stock_h < r_h:
                continue
            position = self._find_bottom_left_position(stock, rotated_size)
            if position:
                trim_loss = self._calculate_potential_waste_for_stock(stock, rotated_size)
                if trim_loss < min_trim_loss:
                    min_trim_loss = trim_loss
                    best_position = position
                    best_orientation = rotated_size
        return best_position, best_orientation, min_trim_loss
    
    def get_action(self, observation, info):
        if (self.policy_id == 1):
            self.init_demand = max(self.init_demand, self.count_demand(observation["products"]))
            list_prods = observation["products"]
            best_stock_idx = -1
            best_position = None
            best_size = None
            best_waste = float('inf')
            
            percent_demand = self.count_demand(list_prods) / self.init_demand
            if(percent_demand > 0.2):
                for prod in list_prods:
                    if prod["quantity"] > 0:
                        prod_w, prod_h = prod["size"]
                        for i, stock in enumerate(observation["stocks"]):
                            stock_w, stock_h = self._get_stock_size_(stock)
                            if stock_w < prod_w or stock_h < prod_h:
                                if stock_w < prod_h or stock_h < prod_w:
                                    continue
                            result = self.simulated_annealing(stock, (prod_w, prod_h), list_prods)
                            if result is not None:
                                pos_x, pos_y = result["position"]
                                prod_w, prod_h = result["size"]
                                stock_idx = i
                                best_position = (pos_x, pos_y)
                                best_size = (prod_w, prod_h)
                                best_stock_idx = stock_idx
                                break
                        if stock_idx != -1:
                            break
            else:
                for prod in list_prods:
                    if prod["quantity"] > 0:
                        prod_w, prod_h = prod["size"]
                        used_stock = self.list_used_stock(observation["stocks"])
                        for stock_idx, stock in used_stock:
                            result = self.simulated_annealing(stock, (prod_w, prod_h), list_prods)
                            if result is not None:
                                pos_x, pos_y = result["position"]
                                prod_w, prod_h = result["size"]
                                best_position = (pos_x, pos_y)
                                best_size = (prod_w, prod_h)
                                best_stock_idx = stock_idx
                                break
                        if best_stock_idx != -1:
                            return {"stock_idx": best_stock_idx, "size": best_size, "position": best_position}

                for prod in list_prods:
                    if prod["quantity"] > 0:
                        prod_w, prod_h = prod["size"]
                        potential_stocks = self.find_potential_stocks(observation["stocks"], (prod_w, prod_h))
                        for stock_idx, stock, _ in potential_stocks:
                            result = self.simulated_annealing(stock, (prod_w, prod_h), list_prods)
                            if result is not None:
                                pos_x, pos_y = result["position"]
                                prod_w, prod_h = result["size"]
                                best_position = (pos_x, pos_y)
                                best_size = (prod_w, prod_h)
                                best_stock_idx = stock_idx
                                break
                        if best_stock_idx != -1:
                            break

            if best_stock_idx == -1 or best_position is None:
                return {"stock_idx": -1, "size": None, "position": None}
            return {"stock_idx": best_stock_idx, "size": best_size, "position": best_position}
        else:
            products = observation["products"]
            stocks = observation["stocks"]

            best_stock_idx = -1
            best_position = None
            best_size = None
            min_trim_loss = float('inf')

            for product in products:
                if product["quantity"] > 0:
                    for i, stock in enumerate(stocks):
                        position, size, trim_loss = self._find_best_cut(stock, product, products)
                        if position and trim_loss < min_trim_loss:
                            min_trim_loss = trim_loss
                            best_stock_idx = i
                            best_position = position
                            best_size = size

            if best_stock_idx == -1 or best_position is None:
                return {"stock_idx": -1, "size": None, "position": None}

            return {"stock_idx": best_stock_idx, "size": best_size, "position": best_position}