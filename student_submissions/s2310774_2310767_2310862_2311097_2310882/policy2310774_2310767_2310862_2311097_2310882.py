from policy import Policy


class Policy2310774_2310767_2310862_2311097_2310882(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.prod_area=0
        if policy_id == 1: #FFDH
            self.id=1
            pass
        elif policy_id == 2: #NFDH
            self.id=2
            self.flag=True
            self.sort_prods=[] 
            self.stk_inf=[] 
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.id==1:
            # First Fit Decreasing Height:
            list_prods = observation["products"]
            used_stocks = []
            # Sắp xếp các product theo diện tích giảm dần
            sorted_prods = sorted(
                list_prods,
                key=lambda prod: prod["size"][0] * prod["size"][1],
                reverse=True
            )

            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0

            for prod in sorted_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]

                    # Lắp lại qua các stock
                    for i, stock in enumerate(observation["stocks"]):
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
                                stock_idx = i
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
                                stock_idx = i
                                break
                        if pos_x is not None and pos_y is not None:
                            if stock_idx not in used_stocks:
                                used_stocks.append(stock_idx)

                    if pos_x is not None and pos_y is not None:
                        break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        elif self.id==2:
            # Next Fit Decreasing Height:
            terminate=True
            for prod in self.sort_prods:
                if prod["quantity"] > 0: 
                    terminate=False 
                    break
            if terminate: #Nếu mọi prod_quantity = 0 thì đặt lại các attribute
                self.stk_inf=[] 
                self.flag=True
            if self.flag: #Khởi tạo các attribute cho lần chạy đầu tiên
                self.sort_prods=sorted(observation["products"], key=lambda prod: prod["size"][1], reverse=True) #Sắp xếp từ cao đến thấp
                self.stk_inf=[[] for _ in enumerate(observation["stocks"])] #List của list các level của từng stock
                self.flag=False

            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0

            for prod in self.sort_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    
                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size

                        if stock_w < prod_w or stock_h < prod_h:
                            continue

                        pos_x, pos_y = None, None

                        if not self.stk_inf[i]: #Khởi tạo level đầu tiên
                            temp=int(stock_h-1)
                            self.stk_inf[i].append(temp)
                        
                        for y in self.stk_inf[i]: #Lặp từng level
                            temp=int(y-prod_size[1]+1)
                            if temp>=0 and self._can_place_(stock, (0, temp), prod_size): #Vị trí đàu tiên của level y
                                pos_x, pos_y = 0, temp
                                if(temp>0):
                                    self.stk_inf[i].append(temp-1) #Tạo level mới nếu chèn vào vị trí đầu tiên
                                break
                            for x in range(1, stock_w - prod_w + 1): #Các vị trí tiếp theo của level y
                                if temp<0:
                                    break
                                if self._can_place_(stock, (x, temp), prod_size):
                                    pos_x, pos_y = x, temp
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                    
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            self.prod_area+=prod_w*prod_h 
                            break

                    if pos_x is not None and pos_y is not None:
                        break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        pass

    # Student code here
    # You can add more functions if needed
