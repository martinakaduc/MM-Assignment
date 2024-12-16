from policy import Policy
import numpy as np


class Policy2352601_2352832_2352699_2352857_2353226(Policy):
    # with can_place and return pos, we need to pass the column first, and then row => (column,row)
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id=policy_id
        # Student code here
        if self.policy_id == 1:
            self.reset = 1
            self.ref_line = None
            self.lowest = None
            self.edge_line = None
            self.area = None
            self.stock_count=None
            pass
        elif self.policy_id == 2:
            self.lower_bound = 0
            pass

 #------------------------------------------------------------BrickLaying heuristic------------------------------------------------------------------------     

    def lowest_point(self,stock,stock_idx,width,height):
        stock_w,stock_h=self._get_stock_size_(stock) 
        lowest_1 = (-1,-1)
        lowest_2 = (-1,-1)
        highest_1 = -1
        highest_2 = -1
        for x in range(height):
            for y in range(width,stock_w):
                if stock[y][x] == -1 and lowest_1==(-1,-1):
                    if self.lowest[stock_idx]==-1 or x>self.lowest[stock_idx]:
                        lowest_1=(y,x)
                        self.lowest[stock_idx]=x
                        continue
                if lowest_1!=(-1,-1) and stock[y][x]!=-1:
                    lowest_2=(y-1,x)
                    break
            if lowest_1!=(-1,-1):
                if lowest_2==(-1,-1):
                    lowest_2=(stock_w-1,x) #if move to the end
                break
        
        if lowest_1!=(-1,-1):
            if lowest_1[0]==0 or lowest_1[0]==stock_w-1: # if y in the limit bound
                highest_1=height
            else:
                for x in range(height):
                    if stock[lowest_1[0]-1][x]==-1: 
                        highest_1=x #just assign for check h1 and h2
                        break
                if highest_1==-1:
                    highest_1=height
            
        if lowest_2!=(-1,-1):
            if lowest_2[0]==0 or lowest_2[0]==stock_w-1:
                highest_2=height
            else:
                for x in range(height):
                    if stock[lowest_2[0]+1][x]==-1:
                        highest_2=x
                        break   
                if highest_2==-1:
                    highest_2=height
        return lowest_1, highest_1, lowest_2, highest_2
    
    def fitness_value(self,pos,height,prod_list,stock,line,choose2=False): #pos depend on h1 and h2
        stock_w, stock_h=self._get_stock_size_(stock)
        for prod in prod_list: # 3 satisfies
            if prod["quantity"]>0:
                w, h = prod["size"]
                for size in ([w, h], [h, w]):
                    prod_w, prod_h = size
                    if prod_h+pos[1]-1 <= line: #pos(column == width,row == height)
                        if choose2:
                            pos_check=(pos[0]-prod_w+1,pos[1]) #the pos out of this method is not change
                            if pos_check[0]>=0:
                                if self._can_place_(stock,pos_check,size) and (pos_check[0]==0 or not self._can_place_(stock,(pos_check[0]-1,pos_check[1]),size)) and (pos_check[0]==stock_w-1 or not self._can_place_(stock,(pos_check[0]+1,pos_check[1]),size)) and height-pos_check[1]==prod_h:
                                    prod["size"]= [prod_w,prod_h]
                                    return pos_check,size
                        else:    
                            if pos[0]+prod_w<=stock_w:
                                if self._can_place_(stock,pos,size) and (pos[0]==0 or not self._can_place_(stock,(pos[0]-1,pos[1]),size)) and (pos[0]==stock_w-1 or not self._can_place_(stock,(pos[0]+1,pos[1]),size)) and height-pos[1]==prod_h:
                                    prod["size"]= [prod_w,prod_h]
                                    return pos,size 

        for prod in prod_list: # 2 satisfies
            if prod["quantity"]>0:
                w, h = prod["size"]
                for size in ([w, h], [h, w]):
                    prod_w, prod_h = size
                    if prod_h+pos[1]-1 <= line: 
                        if choose2:
                            pos_check=(pos[0]-prod_w+1,pos[1]) 
                            if pos_check[0]>=0:
                                if self._can_place_(stock,pos_check,size) and (pos_check[0]==0 or not self._can_place_(stock,(pos_check[0]-1,pos_check[1]),size)) and (pos_check[0]==stock_w-1 or not self._can_place_(stock,(pos_check[0]+1,pos_check[1]),size)):                                   
                                    prod["size"]= [prod_w,prod_h]
                                    return pos_check,size
                        else:    
                            if pos[0]+prod_w<=stock_w:
                                if self._can_place_(stock,pos,size) and (pos[0]==0 or not self._can_place_(stock,(pos[0]-1,pos[1]),size)) and (pos[0]==stock_w-1 or not self._can_place_(stock,(pos[0]+1,pos[1]),size)):
                                    prod["size"]= [prod_w,prod_h]
                                    return pos,size
                    
        for prod in prod_list: # 1 satisfies
            if prod["quantity"]>0:
                w, h = prod["size"]
                for size in ([w, h], [h, w]):
                    prod_w, prod_h = size
                    if prod_h+pos[1]-1 <= line: 
                        if choose2:
                            pos_check=(pos[0]-prod_w+1,pos[1]) 
                            if pos_check[0]>=0:
                                if self._can_place_(stock,pos_check,size) and height-pos_check[1]==prod_h:
                                    prod["size"]= [prod_w,prod_h]
                                    return pos_check,size 
                        else:    
                            if pos[0]+prod_w<=stock_w:
                                if self._can_place_(stock,pos,size) and height-pos[1]==prod_h:
                                    prod["size"]= [prod_w,prod_h]
                                    return pos,size

        for prod in prod_list: # 0 satisfies
            if prod["quantity"]>0:
                w, h = prod["size"]
                for size in ([w, h], [h, w]):
                    prod_w, prod_h = size
                    if prod_h+pos[1]-1 <= line: 
                        if choose2:
                            pos_check=(pos[0]-prod_w+1,pos[1]) 
                            if pos_check[0]>=0:
                                if self._can_place_(stock,pos_check,size):
                                    prod["size"]= [prod_w,prod_h]
                                    return pos_check,size
                        else:    
                            if pos[0]+prod_w<=stock_w:
                                if self._can_place_(stock,pos,size):
                                    prod["size"]= [prod_w,prod_h]
                                    return pos,size
            
        return (-1,-1),[-1,-1]
    
    
    def heuristic(self,prod_list,stock,stock_idx):
        stock_w, stock_h = self._get_stock_size_(stock)
                
        def place_product(pos, prod_size):
            prod_w, prod_h = prod_size
            self.ref_line[stock_idx] += prod_h
            self.edge_line[stock_idx] = prod_w
            return pos, prod_size
            
        def check_place(prod_size, check_rotation=True):
            #check if can place (for both direction)
            prod_w, prod_h = prod_size
            for size in ([prod_w, prod_h], [prod_h, prod_w] if check_rotation else []):
                w, h = size
                if w <= stock_w and h <= stock_h:
                    for x in range(stock_h):
                        if self._can_place_(stock, (0, x), size) and (self.edge_line[stock_idx] == 0 or w == self.edge_line[stock_idx]):
                            prod["size"]=size
                            return (0, x), size
                        if self._can_place_(stock, (0, x), size) and (self.edge_line[stock_idx] == 0 or w < self.edge_line[stock_idx]): #check for smaller  
                            prod["size"]=size
                            return (0, x), size
            return None

        if self.ref_line[stock_idx]+1<=self.area[stock_idx]/stock_w: # fill stack until cannot
            for prod in prod_list:
                if prod["quantity"] > 0:
                    prod_w, prod_h=prod["size"]
                    if prod_w>prod_h: 
                        placement = check_place(prod["size"], check_rotation=True)
                        if placement:
                            return place_product(*placement)
                    else:
                        placement = check_place([prod_h,prod_w], check_rotation=True)
                        if placement:
                            return place_product(*placement)
                
                        
        for _ in range((stock_w-self.edge_line[stock_idx])*stock_h):
            pos_1, h1, pos_2, h2= self.lowest_point(stock,stock_idx,self.edge_line[stock_idx],self.ref_line[stock_idx])
            if pos_1!=(-1,-1) or pos_2!=(-1,-1) or h1!=-1 or h2!=-1:
                if h1>=h2:
                    pos,product=self.fitness_value(pos_1,h1,prod_list,stock,self.ref_line[stock_idx],False) #get the product size that satisfies
                    if pos!=(-1,-1):
                        self.lowest[stock_idx]=-1
                        return pos,product
                else:
                    pos,product=self.fitness_value(pos_2,h2,prod_list,stock,self.ref_line[stock_idx],True)
                    if pos!=(-1,-1):                       
                        self.lowest[stock_idx]=-1
                        return pos,product
        self.lowest[stock_idx]=-1
                               
        for prod in prod_list: #place stack 1 more time
            if prod["quantity"] > 0:
                prod_w, prod_h=prod["size"]
                if prod_w>prod_h: 
                    placement = check_place(prod["size"], check_rotation=True)
                    if placement:
                        return place_product(*placement)
                else:
                    placement = check_place([prod_h,prod_w], check_rotation=True)
                    if placement:
                        return place_product(*placement)
         
        #if cant stack anymore -> fill the rest with fitness value            
        for _ in range((stock_w-self.edge_line[stock_idx])*stock_h):
            pos_1, h1, pos_2, h2= self.lowest_point(stock,stock_idx,self.edge_line[stock_idx],stock_h)
            if pos_1!=(-1,-1) or pos_2!=(-1,-1) or h1!=-1 or h2!=-1:
                if h1>=h2:
                    pos,product=self.fitness_value(pos_1,h1,prod_list,stock,stock_h,False) #get the product size that satisfies
                    if pos!=(-1,-1):
                        self.lowest[stock_idx]=-1
                        return pos,product
                else:
                    pos,product=self.fitness_value(pos_2,h2,prod_list,stock,stock_h,True)
                    if pos!=(-1,-1):                       
                        self.lowest[stock_idx]=-1
                        return pos,product


        return (-1,-1),[-1,-1]
    
#--------------------------------------------------End of BrickLaying heuristic---------------------------------------------------------------------------

#--------------------------------------------------Touching Perimeter (TPRF) heuristic---------------------------------------------------------------------

    def _compute_lower_bound(self, products, stocks): # Compute the lower bound (L0 and L2) for the  problem
        total_area = sum(p['quantity'] * p['size'][0] * p['size'][1] for p in products)
        max_stock_area = max(stock.shape[0] * stock.shape[1] for stock in stocks)
        return int(np.ceil(total_area / max_stock_area))
    
    def _evaluate_score(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        stock_w, stock_h = self._get_stock_size_(stock)

        # Base score from perimeter adjacency
        score = 0

        # Top edge
        if pos_x == 0:  # Top edge touches the stock boundary
            score += prod_h
        else:
            score += np.sum(stock[pos_x - 1, pos_y:pos_y + prod_h] != -1)

        # Bottom edge
        if pos_x + prod_w == stock_w:  # Bottom edge touches the stock boundary
            score += prod_h
        else:
            score += np.sum(stock[pos_x + prod_w, pos_y:pos_y + prod_h] != -1)

        # Left edge
        if pos_y == 0:  # Left edge touches the stock boundary
            score += prod_w
        else:
            score += np.sum(stock[pos_x:pos_x + prod_w, pos_y - 1] != -1)

        # Right edge
        if pos_y + prod_h == stock_h:  # Right edge touches the stock boundary
            score += prod_w
        else:
            score += np.sum(stock[pos_x:pos_x + prod_w, pos_y + prod_h] != -1)

        # Normalize score by the product's perimeter
        perimeter = 2 * (prod_w + prod_h)
        return score / perimeter
    
    def _stock_filled(self, stock): # Calculate the number of empty cells (-1 values) in the stock.
        stock_w, stock_h = self._get_stock_size_(stock)
        return np.sum(stock == -1) / (stock_w * stock_h)
    
    def _find_normal_position(self, stock, position, prod_size): # Find a normal position for the product
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        if pos_x != 0:
            if not np.all(stock[pos_x - 1, pos_y:pos_y + prod_h] == -1):
                if pos_y != 0:
                    return not np.all(stock[pos_x: pos_x + prod_w, pos_y - 1] == -1)
            else: return False
        if pos_y != 0:
            if not np.all(stock[pos_x: pos_x + prod_w, pos_y - 1] == -1):
                if pos_x != 0:
                    return not np.all(stock[pos_x - 1, pos_y:pos_y + prod_h] == -1)
            else: return False
        return True      
    
    def _first_orientation(self, stock, prod_size): # Orientating the product so that it's longest edge is parrallel to the longest edge of the stock
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        
        if stock_w >= stock_h:
            if prod_w >= prod_h:
                return prod_size
            else: return prod_size[::-1]
        else:
            if prod_h >= prod_w:
                return prod_size
            else: return prod_size[::-1]    
    
#----------------------------------------------------------End of Touching Perimeter (TPRF) heuristic------------------------------------------------------------------------------------
    
    def get_action(self, observation, info):
        #-------------------------BrickLaying action--------------------------------------------------

        if self.policy_id==1:
            if self.reset==1:
                self.ref_line= [0] * len(observation["stocks"])
                self.lowest= [-1] * len(observation["stocks"]) # use to track another lowest point if the space is too small
                self.edge_line= [0] * len(observation["stocks"])
                self.area=[0] * len(observation["stocks"])
                self.stock_count=0
                self.reset=0
            
            if sum(prod["quantity"] for prod in observation["products"]) == 1:
                self.reset = 1
                
            #sorted product depend on perimeter
            sorted_prods = sorted(observation["products"], key=lambda prod: 2 * (prod["size"][0] + prod["size"][1]), reverse=True)
            
            indexed_stocks = list(enumerate(observation["stocks"])) # stick each stock with each og index

            #sort stocks based on area
            sorted_stocks = sorted(
                indexed_stocks,
                key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1]
            )
            
            while self.stock_count < len(sorted_stocks):
                original_index, stock = sorted_stocks[self.stock_count]
                pos, prod_size = self.heuristic(sorted_prods, stock, original_index)
                if pos != (-1, -1):  # Valid position found
                    self.area[original_index] += prod_size[0] * prod_size[1]
                    return {"stock_idx": original_index, "size": prod_size, "position": (pos[0], pos[1])}
                else:
                   self.stock_count += 1
                    
        
        #-----------------------End of bricklaying action------------------------------------------------

        #-----------------------Touching Perimeter (TPRF) action-----------------------------------------

        elif self.policy_id==2:
            products = sorted(
                [p for p in observation['products'] if p['quantity'] > 0],
                key=lambda p: (p['size'][0] * p['size'][1], min(p['size'])),
                reverse=True
            )
            stocks = observation['stocks']
            prod_size = [0, 0]
            stock_idx = -1

            if self.lower_bound == 0:  # Compute the initial lower_bound once
                self.lower_bound = self._compute_lower_bound(products, stocks)

            curr_lower_bound = self.lower_bound
            max_lower_bound = min(100, len(stocks))  # Limit lower_bound to a maximum of 100 or total stocks

            for product in products: # Loop through all the products
                prod_size = product['size']
                best_score = -1.0
                best_stock = 1.0
                placed = False
                best_pos_x, best_pos_y = -1, -1  # Track the best position

                while not placed:
                    if curr_lower_bound < self.lower_bound: # If the all the stocks in lowerbound can fit the product then access to a new one and put the product at 0, 0 
                        stock = stocks[curr_lower_bound]
                        stock_w, stock_h = self._get_stock_size_(stock)
                        curr_lower_bound = self.lower_bound
                        score = 0.5
                        prod_size = self._first_orientation(stock, prod_size)
                        for orientation in [prod_size, prod_size[::-1]]:
                            prod_w, prod_h = orientation
                            if stock_w >= prod_w and stock_h >= prod_h:
                                if self._can_place_(stock, (0, 0), orientation):
                                    if score > best_score:
                                        best_score = score
                                        best_pos_x, best_pos_y = 0, 0
                                        prod_size = orientation

                        if best_pos_x != -1 and best_pos_y != -1:
                            stock_idx = curr_lower_bound - 1
                            placed = True
                            break

                    else:
                        for i, stock in enumerate(stocks[:self.lower_bound]): # Loop through all the stocks in the lowerbound range
                            stock_w, stock_h = self._get_stock_size_(stock)
                            temp_best_stock = self._stock_filled(stock)
                            prod_size = self._first_orientation(stock, prod_size)
                            for orientation in [prod_size, prod_size[::-1]]:
                                prod_w, prod_h = orientation
                                if stock_w >= prod_w and stock_h >= prod_h:
                                    for x in range(stock_w - prod_w + 1):
                                        for y in range(stock_h - prod_h + 1):
                                            if self._can_place_(stock, (x, y), orientation):
                                                if self._find_normal_position(stock, (x, y), orientation):# Find the normal position for the product
                                                    score = self._evaluate_score(stock, (x, y), orientation) # Evaluate the edge fill for the orientation
                                                    # Compare edge fill score to prioritize rotations
                                                    if score > best_score:
                                                        best_score = score
                                                        best_stock = temp_best_stock
                                                        best_pos_x, best_pos_y = x, y
                                                        prod_size = orientation

                                                    elif score == best_score and temp_best_stock < best_stock:
                                                        best_stock = temp_best_stock
                                                        best_pos_x, best_pos_y = x, y
                                                        prod_size = orientation

                            if best_pos_x != -1 and best_pos_y != -1:
                                stock_idx = i
                                placed = True
                                break

                    if not placed:
                        curr_lower_bound = self.lower_bound
                        self.lower_bound += 1
                        if self.lower_bound > max_lower_bound:
                            placed = True  # No placement possible, stop searching
                            break

                if placed:
                    break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (best_pos_x, best_pos_y)}
        
        #------------------------------End of Touching Perimeter (TPRF) action---------------------------------------
                    


                    