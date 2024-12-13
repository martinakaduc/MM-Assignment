from policy import Policy
import numpy as np
class Policy2353017_2353215__2352030_2352588(Policy):

    def __init__(self, policy = 3):
        # Student code here
        super().__init__()
        self.list_prods = None
        self.first_place = None
        self.level_limit = None
        assert policy in [1,2,3]
        self.policy_id = policy
        pass
    def get_action(self, observation, info):
        if self.policy_id ==1:
            return self.get_action1(observation, info)      #BOTTOM LEFT DECREASING (BLD) ALGORITHM
        elif self.policy_id==2:
            return self.get_action2(observation, info)      #Next fit decreasing height algorithm (NFDH)
        elif self.policy_id==3:
            return self.get_action3(observation, info)      #First fit decreasing height algorithm (FFDH)
    
    
    
    #IMPLEMENTATION FOR BOTTOM LEFT DECREASING (BLD) ALGORITHM
    def get_action1(self, observation, info):
        # Student code here
        if self.list_prods is None:
            self.list_prods = tuple(sorted(observation["products"], key=lambda x: (x["size"][0] * x["size"][1], max(x["size"][1], x["size"][0]), min(x["size"][1], x["size"][0])), reverse=True))
                                                                        #area                           #height        #width
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        size = [0,0]
        # Pick a product that has quality > 0
        for prod in self.list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                if prod_size[0] > prod_size[1]:
                    prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = None, None
                    
                    for y in range(stock_h - min(prod_w,prod_h) + 1):
                        for x in range(stock_w - min(prod_w,prod_h) + 1):
                            size_to_check = prod_size[:]
                            if self._can_place_(stock, (x, y), size_to_check):

                                pos_x, pos_y = x, y
                                size = size_to_check
                                break
                            else:       #if cannot, try rotate the item
                                size_to_check[0], size_to_check[1] = size_to_check[1], size_to_check[0]
                                if self._can_place_(stock, (x, y), size_to_check):
                                    pos_x, pos_y = x, y
                                    size = size_to_check
                                    break
                                else:
                                    size_to_check[0], size_to_check[1] = size_to_check[1], size_to_check[0]
                        if pos_x is not None and pos_y is not None:
                            break

                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": size, "position": (pos_x, pos_y)}

    #IMPLEMENTATION FOR LEVEL ORIENTED ALGORITHMS
    #Next fit decreasing height algorithm (NFDH)
    def get_action2(self, observation, info):
        # Student code here
        if self.list_prods is None:
            self.list_prods = tuple(sorted(observation["products"], key=lambda x: (max(x["size"][1], x["size"][0]),min(x["size"][1], x["size"][0])), reverse=True))
                                                                                            #height        #width


        if self.first_place is None:
            self.first_place = np.ones(len(observation["stocks"]))

        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            
        if self.level_limit is None:
            self.level_limit = np.zeros((len(observation["stocks"]), 2))
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                self.level_limit[i][1] = stock_h-1
                
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        size = [0,0]
        # Pick a product that has quality > 0
        for prod in self.list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = None, None
                    if prod_size[0] > prod_size[1]:
                        prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
                    size_to_check = prod_size[:]
                    for y in range(int(self.level_limit[i][0]), int(self.level_limit[i][1])-prod_size[1]+2):
                        for x in range(stock_w - prod_size[0] + 1):
                            
                            if self._can_place_(stock, (x, y), size_to_check):
                                pos_x, pos_y = x, y
                                size = size_to_check[:]

                                if self.first_place[i] == 1:
                                    self.first_place[i] = 0    
                                    self.level_limit[i][1] = self.level_limit[i][0] + prod_size[1] -1
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break
                    
                    size_to_check[0], size_to_check[1] = size_to_check[1], size_to_check[0]
                    for y in range(int(self.level_limit[i][0]), int(self.level_limit[i][1])-prod_size[0]+2):
                        for x in range(stock_w - prod_size[1] + 1):
                            if self._can_place_(stock, (x, y), size_to_check):
                                pos_x, pos_y = x, y
                                size = size_to_check[:]

                                if self.first_place[i] == 1:
                                    self.first_place[i] = 0    
                                    self.level_limit[i][1] = self.level_limit[i][0] + prod_size[0] -1
                                    print(self.level_limit[i])
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                        
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break
                    elif self.level_limit[i][1]+1 < stock_h - min(prod_h, prod_w) + 1:
                        for b in range(int(self.level_limit[i][1]+1), stock_h - prod_size[1] + 2):
                            for a in range(stock_w - prod_size[0] + 1):
                                size_to_check = prod_size[:]
                                if self._can_place_(stock, (a, b), size_to_check):
                                    self.level_limit[i][0] = self.level_limit[i][1] + 1
                                    self.level_limit[i][1] = self.level_limit[i][0] + prod_size[1] -1
                                    pos_x, pos_y = a,b
                                    size = size_to_check[:]
                                    break       
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break
                        
                        for b in range(int(self.level_limit[i][1]+1), stock_h - prod_size[0] + 2):
                            for a in range(stock_w - prod_size[1] + 1):
                                size_to_check = prod_size[:]
                                size_to_check[0], size_to_check[1] = size_to_check[1], size_to_check[0]
                                if self._can_place_(stock, (a, b), size_to_check):

                                    self.level_limit[i][0] = self.level_limit[i][1] + 1
                                    self.level_limit[i][1] = self.level_limit[i][0] + prod_size[0] -1
                                    pos_x, pos_y = a,b
                                    size = size_to_check[:]

                                    break       
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break
                if pos_x is not None and pos_y is not None:
                    break
      
        return {"stock_idx": stock_idx, "size": size, "position": (pos_x, pos_y)}
    
    
    
    
    #First fit decreasing height algorithm (FFDH)
    def get_action3(self, observation, info):
        # Student code here
        if self.list_prods is None:
            self.list_prods = tuple(sorted(observation["products"], key=lambda x: ( max(x["size"][1], x["size"][0])), reverse=True))
                                                                                            #height        #width
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            
        if self.level_limit is None:
            self.level_limit = np.empty((len(observation["stocks"])),dtype=object)
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                self.level_limit[i] = []
                self.level_limit[i].append([0,stock_h])
                
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        size = [0,0]
        # Pick a product that has quality > 0
        for prod in self.list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    
                    pos_x, pos_y = None, None
                    if prod_size[0] > prod_size[1]:
                        prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
                    for level in self.level_limit[i]:
                        size_to_check = prod_size[:]
                        for y in range(int(level[0]), int(level[1]) - prod_size[1]+2):
                            for x in range (stock_w - prod_size[0] + 1):
                                if self._can_place_(stock, (x, y), size_to_check):
                                    pos_x, pos_y = x, y
                                    size = size_to_check[:]
                                    break
                                if pos_x is not None and pos_y is not None:
                                    break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break
                        if pos_x is not None and pos_y is not None:
                            if level[1]==stock_h:
                                level[1]=level[0] + size[1] -1
                                self.level_limit[i].append([level[1]+1, stock_h])     
                  
                            break 
                        size_to_check[0], size_to_check[1] = size_to_check[1], size_to_check[0]      
                        for y in range(int(level[0]), int(level[1]) - prod_size[0]+2):
                            for x in range (stock_w - prod_size[1] + 1):
                                if self._can_place_(stock, (x, y), size_to_check):

                                    pos_x, pos_y = x, y
                                    size = size_to_check[:]
                                    break
                                if pos_x is not None and pos_y is not None:
                                    break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break
                        if pos_x is not None and pos_y is not None:
                            if level[1]==stock_h:
                                level[1]=level[0] + size[1] -1
                                self.level_limit[i].append([level[1]+1, stock_h])                        
                            break 
                    if pos_x is not None and pos_y is not None:
                        break
                if pos_x is not None and pos_y is not None:
                    break
     
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    






    # Student code here
    # You can add more functions if needed
