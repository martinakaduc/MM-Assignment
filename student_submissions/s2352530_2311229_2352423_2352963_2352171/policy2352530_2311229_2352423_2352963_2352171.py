from policy import Policy
import math
import numpy as np
class FFDPolicy(Policy):
    def __init__(self, policy_id = 1):
        self.prod_area_left = 0
        self.stock_area = 0
        self.stock_sorted = []
        self.prod_sorted = []
        self.flag = True
        self.count_stock = 0
        self.w = 0
        self.h = 0
        self.check_prod = 0
    def areastock(self, stock):
            w, h = self._get_stock_size_(stock)
            return w*h
    def check_stock(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        if pos_x + prod_w > stock.shape[0] or pos_y + prod_h > stock.shape[1]:
            return False  
        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)
    def cal_current_size(self, stock):
    # Tính toán độ rộng (w)
        row_mask = [any(cell > -1 for cell in row) for row in stock]
        w = sum(row_mask)
    
    # Tính toán chiều cao (h)
        col_mask = [any(row[i] > -1 for row in stock) for i in range(len(stock[0]))]
        h = sum(col_mask)
    
        return w, h
    def get_action(self, observation, info):
        if info.get("filled_ratio", 0) == 0:
            self.__init__()
        used_stocks = np.zeros(len(observation["stocks"]), dtype=bool)
        used_products = np.zeros(len(observation["products"]), dtype=bool)
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = None, None
        count_stock = -1
        self.prod_area_left = sum(math.prod(prod["size"]) * prod["quantity"] for prod in observation["products"])
        self.prod_sorted = sorted(enumerate(observation["products"]), key=lambda x: x[1]["size"][0] * x[1]["size"][1], reverse=True)
        stock_areas = [self.areastock(stock) for stock in observation["stocks"]]
        self.stock_sorted = sorted(enumerate(observation["stocks"]), key=lambda x: self.areastock(x[1]), reverse=True)
        for i,_ in self.stock_sorted:
            count_stock+=1
            if used_stocks[i] or count_stock < 0:
                continue
            stock = observation["stocks"][i]
            stock_w, stock_h = self._get_stock_size_(stock)
            self.stock_area = stock_w * stock_h
            if self.prod_area_left < self.stock_area * 0.7 and self.flag:
                self.count_stock +=1
                continue
            check_prod = -1
            for idx, prod in self.prod_sorted:
                if used_products[idx] and check_prod < self.check_prod or prod["quantity"] <= 0:
                    continue
                check_prod+=1
                prod_quantity = prod["quantity"]
                curr_x, curr_y = self.cal_current_size(stock)
                if prod_quantity > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        if curr_x < x: continue
                        for y in range(stock_h - prod_h + 1):
                            if curr_y < y: continue
                            if self.check_stock(stock, (x, y), prod_size):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    break
                        if pos_x is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        break
                    if stock_w <= prod_h or stock_h <= prod_w:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self.check_stock(stock, (x, y), prod_size[::-1]):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        pos_x, pos_y = x, y
                                        prod_size = prod_size[::-1]
                                        break
                            if pos_x is not None:
                                break
            if pos_x is not None and pos_y is not None:
                        self.check_prod +=1
                        stock_idx = i
                        self.prod_area_left-=np.prod(prod_size) 
                        used_products[idx] = True  
                        self.flag = False
                        break
            used_stocks[i] = True 
            self.count_stock +=1
            self.check_prod = 0
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
class BFDPolicy(Policy):
    def __init__(self, policy_id=2):
        self.w = 0
        self.h = 0
        self.prod_area_left = 0
        self.stock_area = 0
        self.stock_sorted = []
        self.prod_sorted = []
        self.flag = True
        self.count_stock = 0
        self.check_prod = 0

    def areastock(self, stock):
        w, h = self._get_stock_size_(stock)
        return w * h

    def check_stock(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        # Kiểm tra có vượt quá giới hạn ma trận stock không
        if pos_x + prod_w > stock.shape[0] or pos_y + prod_h > stock.shape[1]:
            return False  # Vị trí vượt ngoài ma trận

        # Kiểm tra vùng có giá trị khác -1 không
        return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)

    def cal_current_size(self, stock):
        row_mask = [any(cell > -1 for cell in row) for row in stock]
        w = sum(row_mask)
    
    # Tính toán chiều cao (h)
        col_mask = [any(row[i] > -1 for row in stock) for i in range(len(stock[0]))]
        h = sum(col_mask)
    
        return w, h

    def get_action(self, observation, info):
        if info.get("filled_ratio", 0) == 0:
            self.__init__()
        used_stocks = np.zeros(len(observation["stocks"]), dtype=bool)
        stock_idx = -1
        prod_size = [0, 0]
        count_stock = -1
        self.prod_area_left = sum(math.prod(prod["size"]) * prod["quantity"] for prod in observation["products"])
        self.prod_sorted = sorted(enumerate(observation["products"]), key=lambda x: x[1]["size"][0] * x[1]["size"][1], reverse=True)
        self.stock_sorted = sorted(enumerate(observation["stocks"]), key=lambda x: self.areastock(x[1]), reverse=True)

        for i, _ in self.stock_sorted:
            count_stock += 1
            if used_stocks[i] or count_stock < 0:
                continue
            stock = observation["stocks"][i]
            stock_w, stock_h = self._get_stock_size_(stock)
            self.stock_area = stock_w * stock_h
            if self.prod_area_left < self.stock_area * 0.7 and self.flag:
                self.count_stock += 1
                continue

            best_dS = float("inf")
            best_position = [0,0]
            best_prod_size = None

            for product in self.prod_sorted:
                idx, prod = product

                # Kiểm tra xem sản phẩm đã được sử dụng chưa
                if prod.get("quantity", 0) <= 0 or (hasattr(self, 'used_products') and idx in self.used_products):
                    continue

                prod_size = prod.get("size", (0, 0))
                prod_w = int(prod_size[0])
                prod_h = int(prod_size[1])
                curr_size = self.cal_current_size(stock)
                if curr_size is None:
                    continue

                curr_w, curr_h = curr_size
                if prod_w > stock_w or prod_h > stock_h:
                    continue

                for x in range(stock_w - prod_w + 1):
                    if curr_w < x:
                        continue
                    for y in range(stock_h - prod_h + 1):
                        if curr_h < y:
                            continue
                        if self.check_stock(stock, (x, y), (prod_w, prod_h)):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                new_w = max(curr_w, x + prod_w)
                                new_h = max(curr_h, y + prod_h)
                                new_S = new_w * new_h
                                new_dS = (self.stock_area - new_S) / self.stock_area
                                if new_dS < best_dS or best_dS == float("inf"):
                                    best_dS = new_dS
                                    best_position = (x, y)
                                    best_prod_size = (prod_w, prod_h)
                    if best_position and best_prod_size:
                        break
                if best_position and best_prod_size:
                    break
                if stock_w <= prod_w or stock_h <= prod_h:
                    for x in range(stock_w - prod_h + 1):
                        if curr_w < x: continue
                        for y in range(stock_h - prod_w + 1):
                            if curr_h < y: continue
                            if self.check_stock(stock, (x, y), (prod_h, prod_w)):
                                if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                                    new_w = max(curr_w, x + prod_h)
                                    new_h = max(curr_h, y + prod_w)
                                    new_S = new_w * new_h
                                    new_dS = (self.stock_area - new_S)/self.stock_area
                                    if new_dS < best_dS or best_dS == float("inf"):
                                        best_dS = new_dS
                                        best_position = (x, y)
                                        best_prod_size = (prod_h, prod_w)
                        if best_position and best_prod_size:
                            break
            self.flag = False
            if best_position and best_prod_size:
                stock_idx = i
                self.prod_area_left -= np.prod(best_prod_size)
                prod_size = best_prod_size
                position = best_position
                break

            used_stocks[i] = True
            self.count_stock += 1

        return {"stock_idx": stock_idx, "size": prod_size, "position": position}
class Policy2352530_2311229_2352423_2352963_2352171(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = FFDPolicy(policy_id)
        elif policy_id == 2:
            self.policy = BFDPolicy(policy_id)
    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
