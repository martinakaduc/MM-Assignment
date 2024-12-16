from policy import Policy
import copy

# Greedy + Heuristic Algorithm
class Heuristic(Policy):
    def __init__(self, observation):
        self.data = []  
        self.products = list(copy.deepcopy(observation["products"]))  # Store products for reference
        self.products.sort(key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)  # Sort by product area (width * height)
        self.stocks = list(copy.deepcopy(observation["stocks"]))  # Convert to list
        sorted_stock_indices = [
            i for i, stock in sorted(
                enumerate(self.stocks), # Enumerate stocks
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],  # Sort by stock area (width * height)
            reverse=True  # Descending order
            )           
        ]
        self.stocks.sort(key=lambda stock: self._get_stock_size_(stock)[0]*self._get_stock_size_(stock)[1], reverse = True)  # Sort by stock ID
        for i, prod in enumerate(self.products):
            prod["id"] = i
        self.product_demand_fulfilled = {prod["id"]: 0 for prod in self.products}
        for i, stock in enumerate(self.stocks):
            placements = self.solve(stock, sorted_stock_indices[i])
            if all(self.product_demand_fulfilled[prod["id"]] >= prod["quantity"] for prod in self.products):
                k = len(self.stocks) - 1
                while k > i:
                    self._reset_product(placements)
                    best_placements = self.solve(self.stocks[k], sorted_stock_indices[k])
                    if all(self.product_demand_fulfilled[prod["id"]] >= prod["quantity"] for prod in self.products):
                        placements = best_placements
                        break
                    self._reset_stock(self.stocks[k])
                    k -= 1
            self.data.append(placements)
            if all(self.product_demand_fulfilled[prod["id"]] >= prod["quantity"] for prod in self.products):
                break

    def _reset_product(self, placements):
        for p in placements:
            if p is not None:
                prod_size, x, y, k, prod_id = p
                self.product_demand_fulfilled[prod_id] -= 1

    def _reset_stock(self, stock):
        for i in range(len(stock)):
            for j in range(len(stock[0])):
                if stock[i][j] != -2:
                    stock[i][j] = -1

    def solve(self, stock, stock_idx):
        STOCK_W, STOCK_H  = self._get_stock_size_(stock)
        placements = []
        for prod in self.products:
            prod_w, prod_h = prod["size"]
            prod_id = prod["id"]
            prod_size = prod_w, prod_h
            if self.product_demand_fulfilled[prod_id] >= prod["quantity"]:
                continue
            if STOCK_W >= prod_w and STOCK_H >= prod_h:
                for x in range(STOCK_W - prod_w + 1):
                    for y in range(STOCK_H - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                                placements.append((prod_size, x, y, stock_idx, prod_id))
                                for i in range(prod_w):
                                    for j in range(prod_h):
                                        stock[x + i][y + j] = prod_id                   
                                self.product_demand_fulfilled[prod_id] += 1
                        if self.product_demand_fulfilled[prod_id] >= prod["quantity"]:
                                break
                    if self.product_demand_fulfilled[prod_id] >= prod["quantity"]:
                        break
            
                if all(self.product_demand_fulfilled[prod["id"]] >= prod["quantity"] for prod in self.products):
                    break

                for x in range(STOCK_W - prod_h + 1):
                    for y in range(STOCK_H - prod_w + 1):
                        if self._can_place_(stock, (x, y), prod_size[::-1]):
                                placements.append((prod_size[::-1], x, y, stock_idx, prod_id))
                                for i in range(prod_h):
                                    for j in range(prod_w):
                                        stock[x + i][y + j] = prod_id                   
                                self.product_demand_fulfilled[prod_id] += 1

                        if self.product_demand_fulfilled[prod_id] >= prod["quantity"]:
                                break
                    if self.product_demand_fulfilled[prod_id] >= prod["quantity"]:
                        break
                
                if all(self.product_demand_fulfilled[prod["id"]] >= prod["quantity"] for prod in self.products):
                    break

        return placements

    def get_action(self, observation, info):
        """
        Get the action to be taken based on the current observation.
        """
        for placement in self.data:
            for p in placement:
                if p is not None:
                    prod_size, x, y, stock_idx, prod_id = p
                    action = {
                        "stock_idx": stock_idx,
                        "size": prod_size,
                        "position": (x, y)
                    }
                    placement.remove(p)  # Mark as used
                    if len(placement) == 0:
                        self.data.remove(placement)
                    return action
        # Default action if no placements are left
        return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}

# BestFit Algorithm
class BestFit(Policy):
    def __init__(self, observation):
        self.precomputed_positions = []
        self.products = list(copy.deepcopy(observation["products"]))
        self.products.sort(key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)

        self.stocks = list(copy.deepcopy(observation["stocks"]))
        sorted_stock_indices = [
            i for i, stock in sorted(
                enumerate(self.stocks), # Enumerate stocks
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],  # Sort by stock area (width * height)
            reverse=False  # Descending order
            )           
        ]
        self.stocks.sort(key=lambda stock: self._get_stock_size_(stock)[0]*self._get_stock_size_(stock)[1], reverse = False)  # Sort by stock ID
        for i, prod in enumerate(self.products):
            prod["id"] = i

        for i , prod in enumerate(self.products):
            for _ in range(prod["quantity"]):  
                prod_w, prod_h = prod["size"]
                prod_size = prod["size"]
                possible_positions = []
                min_waste = 1.0
                idx = None
                rotation = False
                max_stock_idx = 0
                check_fill =  False
                for j,_ in enumerate(self.stocks):
                    stock_w, stock_h = self._get_stock_size_(self.stocks[j])
                    check = False
                    not_used = 0
                    for x in range(stock_w):
                        for y in range(stock_h):
                            if self.stocks[j][x][y] == -1:
                                not_used += 1
                    if(j > max_stock_idx and check_fill):
                        break
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(self.stocks[j], (x, y), prod_size):
                                    check = True
                                    waste = self._calculate_waste(prod_size, not_used, stock_w, stock_h)
                                    if waste < min_waste:
                                        max_stock_idx = j
                                        rotation = False
                                        check_fill = True
                                        min_waste = waste
                                        possible_positions.clear()
                                        possible_positions.append(
                                            {"stock_idx": sorted_stock_indices[j], "size": prod_size, "position": (x, y)}
                                        )
                                        idx = j
                                if check:
                                    break
                            if check:
                                break
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(self.stocks[j], (x, y), prod_size[::-1]):
                                    check = True
                                    waste = self._calculate_waste(prod_size[::-1], not_used, stock_w, stock_h)
                                    if waste < min_waste:
                                        max_stock_idx = j
                                        rotation = True
                                        check_fill = True
                                        min_waste = waste
                                        possible_positions.clear()
                                        possible_positions.append(
                                            {"stock_idx": sorted_stock_indices[j], "size": prod_size[::-1], "position": (x, y)}
                                        )
                                        idx = j
                                if check:
                                    break
                            if check:
                                break

                if possible_positions:
                    self.precomputed_positions.append(possible_positions[0])
                    x, y = possible_positions[0]["position"]
                    if not rotation:
                        for k in range(prod_w):
                            for j in range(prod_h):
                                self.stocks[idx][x + k][y + j] = i
                    else:
                        for k in range(prod_h):
                            for j in range(prod_w):
                                self.stocks[idx][x + k][y + j] = i

    def _calculate_waste(self, prod_size, not_used, stock_w, stock_h):
        prod_w, prod_h = prod_size
        return (not_used - prod_w*prod_h) / (stock_w * stock_h)
    
    def get_action(self, observation, info):
        for placement in self.precomputed_positions:
            stock_idx = placement["stock_idx"]
            size = placement["size"]
            position = placement["position"]
            self.precomputed_positions.remove(placement)
            return {"stock_idx": stock_idx, "size": size, "position": position}
        
        return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}   
    
'''
Do không truyền observation vào hàm get_action, nên trước khi hoàn thành ep, ta sẽ kiểm tra số lượng sản phẩm còn lại, 
nếu bằng 1 thì gán action = False để khởi tạo lại.
'''

class Policy2313621_2310807_2211173_2311601_2312401(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_1 = False
        self.policy_2 = False
        self.action = False
        if policy_id == 1:
            self.policy_1 = True
        elif policy_id == 2:
            self.policy_2 = True

    def get_action(self, observation, info):
        if self.policy_1:
            if not self.action: # Nếu action = False thì khởi tạo lại 
                self.heuristic = Heuristic(observation)
                self.action = True
            result = self.heuristic.get_action(observation, info)
            if sum(product["quantity"] for product in observation["products"]) == 1: # Nếu số lượng sản phẩm còn lại bằng 1 thì gán action = False
                self.action = False
            return result

        if self.policy_2:
            if not self.action:
                self.bestfit = BestFit(observation)
                self.action = True
            result = self.bestfit.get_action(observation, info)
            if sum(product["quantity"] for product in observation["products"]) == 1:
                self.action = False
            return result