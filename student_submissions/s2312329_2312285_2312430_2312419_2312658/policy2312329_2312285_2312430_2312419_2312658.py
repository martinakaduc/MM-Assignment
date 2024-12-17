from policy import Policy
import numpy as np



class Policy2312329_2312285_2312430_2312419_2312658(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if self.policy_id == 1:
            self.list_stocks = []
            self.list_prods = []
            pass
        elif self.policy_id == 2:
            self.square_stock = np.zeros(100)
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self.get_action1(observation, info)
        elif self.policy_id == 2:
            return self.get_action2(observation, info)
    

    def get_action1(self, observation, info):
        if(info["filled_ratio"]==0):
            self.__init__()
            
            for i,stock in enumerate( observation["stocks"]):
                stock_w,stock_h = self._get_stock_size_(stock)
                self.list_stock = {"idx":i,"stock":stock,"dientich":stock_w * stock_h}
                self.list_stocks.append(self.list_stock)
            self.list_stocks.sort(key=lambda item: item["dientich"], reverse=True) 
            
            
            self.list_prods = list(observation["products"]) #tạo list_prods
            self.list_prods.sort(key=lambda x: (x["size"][0] * x["size"][1]), reverse=True) 
        
        
        stock_idx = -1
        pos_x, pos_y = None, None

        
        for prod in self.list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                
                for item in self.list_stocks: 
                

                    i = item["idx"]
                    
                    stock = item["stock"] 
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    
         
                    if (stock_w > stock_h):
                        if prod_w < prod_h:
                            prod_w,prod_h = prod_h,prod_w
                    else:
                        if prod_w > prod_h:
                            prod_w,prod_h = prod_h,prod_w
                    
                    
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = self._find_bottom_left_position(stock, prod_size)

                    if pos_x is None and pos_y is None:
                        prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
                        pos_x, pos_y = self._find_bottom_left_position(stock, prod_size)
                    
           
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break
   
            if pos_x is not None and pos_y is not None:
                break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


    def _find_bottom_left_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock) 
        prod_w, prod_h = prod_size 

        for y in range(stock_h - prod_h, -1, -1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size): 
                    return x, y  
        return None, None
    

    def _findmaxmin_square_(self, observation):
        stocks = observation["stocks"]
        max_stock = 0
        min_stock = float("inf")
        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            area = stock_w * stock_h
            max_stock = max(max_stock, area)
            min_stock = min(min_stock, area)
        return max_stock, min_stock

    def _avg_square_(self, observation):
        max_stock, min_stock = self._findmaxmin_square_(observation)
        return (max_stock + min_stock) / 2

    def _avg_prod_(self, observation):
        list_prods = observation["products"]
        max_p = 0
        min_p = float("inf")
        for prod in list_prods:
            prod_w, prod_h = prod["size"]
            area = prod_w * prod_h
            max_p = max(max_p, area)
            min_p = min(min_p, area)
        return (max_p + min_p) / 2

    def _sort_products_by_size_(self, observation):
        """Sắp xếp sản phẩm theo thứ tự: trung bình -> lớn -> nhỏ."""
        products = observation["products"]
        areas = [prod["size"][0] * prod["size"][1] for prod in products]
        average_area = sum(areas) / len(areas)

        medium_products = [
            prod
            for prod in products
            if abs(prod["size"][0] * prod["size"][1] - average_area) <= average_area * 0.3
        ]
        large_products = [
            prod for prod in products if prod["size"][0] * prod["size"][1] > average_area
        ]
        small_products = [
            prod for prod in products if prod["size"][0] * prod["size"][1] < average_area
        ]

        medium_products = sorted(medium_products, key=lambda p: p["size"][0] * p["size"][1])
        large_products = sorted(
            large_products, key=lambda p: p["size"][0] * p["size"][1], reverse=True
        )
        small_products = sorted(small_products, key=lambda p: p["size"][0] * p["size"][1])

        return medium_products + large_products + small_products
    
    def _find_position_(self, stock, prod_size):
        """Tìm vị trí đặt sản phẩm trong stock."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h, -1, -1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y, prod_size
        prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
        for x in range(stock_w - prod_size[0] + 1):
            for y in range(stock_h - prod_size[1], -1, -1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y, prod_size
        prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
        return None, None, prod_size

    def get_action2(self, observation, info):
        list_prods = observation["products"]
        list_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        #list_prods = self._sort_products_by_size_(observation)

        stocks = observation["stocks"]
        stock_idx, pos_x, pos_y = -1, None, None
        prod_size = [0, 0]

        for prod in list_prods:
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            if prod["quantity"] > 0:
                if 0.5 < prod_w / prod_h < 1.5 and 0.8 < prod_w * prod_h / self._avg_prod_(observation) < 1.5:
                    for i, stock in enumerate(stocks):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        for k in range(4, 0, -1):
                            for h in range(4, 0, -1):
                                if 1 < stock_h / (k * prod_h) < 1.3 and 1 < stock_w / (h * prod_w) < 1.3 :
                                    if self. square_stock[i] >= 0:
                                        pos_x, pos_y, prod_size = self._find_position_(stock, prod_size)
                                        if pos_x is not None and pos_y is not None:
                                            stock_idx = i
                                            break
                                    elif self.square_stock[i] == 0:
                                        if (prod["quantity"] > k * h ):
                                        #if (self.square_stock[i] != 0 or (self.square_stock[i] == 0 and prod["quantity"] > k * h - prod["quantity"])):
                                            pos_x, pos_y, prod_size = self._find_position_(stock, prod_size)
                                            if pos_x is not None and pos_y is not None:
                                                stock_idx = i
                                                break
                                    elif (prod["quantity"] < k * h ):
                                        continue
                                        
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None:
                        break
                    if pos_x is None:
                        for i, stock in enumerate(stocks):
                            stock_w, stock_h = self._get_stock_size_(stock)
                            if (stock_w > 1.2 * stock_h and prod_size[0] > 1.2 * prod_size[1]):
                                prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
                            elif (stock_h > 1.2 * stock_w and prod_size[1] > 1.2 * prod_size[1]):
                                prod_size[0], prod_size[1] = prod_size[1], prod_size[0]

                            pos_x, pos_y, prod_size = self._find_position_(stock, prod_size)
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break

                if pos_x is None:
                    for i, stock in enumerate(stocks):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        if (stock_w > 1.2 * stock_h and prod_size[0] > 1.2 * prod_size[1]):
                            prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
                        elif (stock_h > 1.2 * stock_w and prod_size[1] > 1.2 * prod_size[0]):
                            prod_size[0], prod_size[1] = prod_size[1], prod_size[0]

                        pos_x, pos_y, prod_size = self._find_position_(stock, prod_size)
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

            if pos_x is not None and pos_y is not None:
                self.square_stock[stock_idx] += prod_size[0] * prod_size[1]
                break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    # Student code here
    # You can add more functions if needed
