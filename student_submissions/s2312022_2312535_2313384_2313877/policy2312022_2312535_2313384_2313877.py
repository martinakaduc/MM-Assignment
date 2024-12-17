from policy import Policy

class Policy2312022_2312535_2313384_2313877(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = 1
           
        elif policy_id == 2:
            self.policy = 2
            

    def get_action(self, observation, info):
        if self.policy == 1:
            return self.get_action_1(observation, info)
        elif self.policy == 2:
            return self.get_action_2(observation, info)

    def create_2SGP(self, current_product, current_stock):
        pattern = None
        L, W = current_stock["L"], current_stock["W"]
        l, w, demand = current_product["size"][0], current_product["size"][1], current_product["quantity"] 
        if l>L or w>W:
            return None
        max_vertical = L // l 

        if max_vertical >= demand:
            pattern = {"size": (demand*l, w), "quantity": demand}
            return pattern

        max_horizontal = W // w 
        for h_count in range(1, max_horizontal + 1):
            if max_vertical * h_count > demand:
                pattern = {"size": (max_vertical*l, (h_count-1) * w), "quantity": max_vertical * (h_count-1)}
                return pattern
            elif max_vertical * h_count == demand:
                pattern = {"size": (max_vertical*l, (h_count) * w), "quantity": max_vertical * (h_count)}
                return pattern
                
        pattern = {"size": (max_vertical*l, max_horizontal * w), "quantity": max_vertical * max_horizontal}
        return pattern

    def create_3SHP(self, pattern_stock, used_product):
        #used_product : index cua product da chon o 2SGP
        efficiency = float('inf')
        chosen_pattern = None
        L, W = pattern_stock["L"], pattern_stock["W"] 
        use_product = None
      
        for i, prod in enumerate(self.products):
            if i == used_product and prod["quantity"]==0:
                continue
            pattern = self.create_2SGP(prod,pattern_stock)
            if pattern["quantity"]==0:
                continue

            new_efficiency = L*W - pattern["size"][0]*pattern["size"][1]
            
            if efficiency > new_efficiency:
                efficiency = new_efficiency
                chosen_pattern = pattern
                use_product = prod
        return chosen_pattern, i, use_product

    def solve_2(self):
        for i, current_product in enumerate(self.products):
            while current_product["quantity"] > 0:
                l, w = current_product["size"][0], current_product["size"][1]
                current_stock = None
                for stock_idx, stock in enumerate(self.stocks):
                    L, W = stock["L"], stock["W"]
                    if L >= l and W >= w:
                        current_stock = stock
                        break
                if current_stock == None:
                    current_product["quantity"] = 0
                if current_stock is not None:
                    pattern = self.create_2SGP(current_product, current_stock)
                    l1, w1 = pattern["size"][0], pattern["size"][1]
                    for x in range(1, w1//w+1):
                        for y in range(1, l1//l+1):
                            self.result.append( {"stock_idx": stock_idx, 
                                                 "size": current_product["size"], 
                                                 "position": ((y-1)*l, (x-1)*w+self.stock_loss[stock_idx]["W0"])} 
                                                 )
                    self.products[i]["quantity"] -= pattern["quantity"]

                    newstock = {"L": L-l1, "W": w1}
                    chosen_pattern, ii, use_product = self.create_3SHP(newstock, i)
                    if chosen_pattern is not None:
                        l2, w2 = chosen_pattern["size"][0], pattern["size"][1]
                        for x in range(1, w2//use_product["size"][1]+1):
                            for y in range(1, l2//use_product["size"][0]+1):
                                self.result.append({"stock_idx": stock_idx, 
                                                    "size": use_product["size"], 
                                                    "position": ((y-1)*use_product["size"][0]+l1, (x-1)*use_product["size"][1]+self.stock_loss[stock_idx]["W0"])})
                        self.products[ii]["quantity"] -= chosen_pattern["quantity"]
                    
                    self.stocks[stock_idx] = {"L": L, "W": W-w1} 
                    self.stock_loss[stock_idx]["L0"] = l1
                    self.stock_loss[stock_idx]["W0"] += w1

        return 
    
    def get_action_2(self, observation, info):
        if info["trim_loss"]==1:
            self.stocks = []
            self.products = []
            self.result = [] 
            self.stock_loss = []

            for stock_idx, stock in enumerate(observation["stocks"]):
                L, W = self._get_stock_size_(stock)
                self.stocks.append({"L": L, "W": W})
                self.stock_loss.append({"stock_idx": stock_idx, "L0": 0, "W0": 0})
            self.products = [{"size": prod["size"], "quantity": prod["quantity"]} for prod in observation["products"]]
            self.products = sorted(self.products, key=lambda product: product["size"][0] * product["size"][1], reverse=True)
            self.solve_2()

        if self.result:
            return self.result.pop(0)
        
        return {"stock_idx": 0, "size": np.array([0, 0]), "position": (0, 0)}
    
    def generate_pattern(self, current_product, current_stock):
            pattern = None
            L, W = current_stock["L"], current_stock["W"]
            l, w, demand = current_product["size"][0], current_product["size"][1], current_product["quantity"] 

            if l>L or w>W or current_product["quantity"]==0:
                return None
            
            max_vertical = L // l 
            if max_vertical >= demand:
                pattern = {"size": (demand*l, w), "quantity": demand}
                return pattern

            max_horizontal = W // w 
            for h_count in range(1, max_horizontal + 1):
                if max_vertical * h_count > demand:
                    pattern = {"size": (max_vertical*l, (h_count-1) * w), "quantity": max_vertical * (h_count-1)}
                    return pattern
                elif max_vertical * h_count == demand:
                    pattern = {"size": (max_vertical*l, (h_count) * w), "quantity": max_vertical * (h_count)}
                    return pattern
                   
            pattern = {"size": (max_vertical*l, max_horizontal * w), "quantity": max_vertical * max_horizontal}
            return pattern

    def master(self, stock, stock_index, initial):
        stack = [(stock, stock_index, initial)]
        while stack:
            current_stock, current_index, current_initial = stack.pop()
            pattern, product_idx = None, None

            for idx, product in enumerate(self.products):
                if product["quantity"] > 0:
                    pattern = self.generate_pattern(product, current_stock)
                    if pattern is None:
                        continue
                    product_idx = idx
                    break

            if pattern is None:
                continue

            l1, w1 = pattern["size"]
            self.products[product_idx]["quantity"] -= pattern["quantity"]
            if self.products[product_idx]["quantity"] < 0:
                self.products[product_idx]["quantity"] = 0

            l, w = self.products[product_idx]["size"]
            for x in range(1, w1 // w + 1):
                for y in range(1, l1 // l + 1):
                    self.result.append({
                        "stock_idx": current_index,
                        "size": (l, w),
                        "position": ((y - 1) * l + current_initial["L"], (x - 1) * w + current_initial["W"])
                    })

            stack.append(({"L": current_stock["L"] - l1, "W": current_stock["W"]}, current_index, {"L": current_initial["L"] + l1, "W": current_initial["W"]}))
            stack.append(({"L": l1, "W": current_stock["W"] - w1}, current_index, {"L": current_initial["L"], "W": current_initial["W"] + w1}))

    def solve_1(self):
        full_demand = True
        for stock_index, stock in enumerate(self.stocks):
            #Kiem tra con san pham can cat khong
            for _,prod in enumerate(self.products):
                if prod["quantity"] !=0:
                    full_demand = False
                    break

            if full_demand == True:
                break
            # De quy de dien stock 1
            initial = {"L": 0, "W":0}
            self.master(stock, stock_index, initial) 
        return 

    def get_action_1(self, observation, info):
        if info["trim_loss"]==1:
            self.stocks = []
            self.products = []
            self.result = [] 

            for _, stock in enumerate(observation["stocks"]):
                L, W = self._get_stock_size_(stock)
                self.stocks.append({"L": L, "W": W})
            self.products = [{"size": prod["size"], "quantity": prod["quantity"]} for prod in observation["products"]]
            self.products = sorted(self.products, key=lambda product: product["size"][0] * product["size"][1], reverse=True)
            self.solve_1()

        if self.result:
            return self.result.pop(0)
        
        return {"stock_idx": 0, "size": np.array([0, 0]), "position": (0, 0)}
    
