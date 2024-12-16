from policy import Policy
import math

class Policy2310211_2310068_2310077_2310148_2310201(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
          self.policy_name = "FFD_modified"
          self.stock_idx = []
          self.stock_sheets = []
          self.sorted_list_prods = []
          self.first_action = False
          self.residual_vector = []

        elif policy_id == 2:
            self.policy_name = "CCF_heuristic"
            self.curr_idx = 0
            self.sequencePosReturn = []
            self.products = []
            self.stock_sheets = [] # Priority cutting large stocks
            self.stock_idx = [] # Keep track of stock index after sort stocksheets
            self.sorted_list_prods = [] # Priority process large products
            self.component_stack_for_curr_stock = [] # Store a stack of components each time we cut
            self.undominated_products = [] # Store list of undominate product 
            self.first_action = False
            self.residual_vector = []
            self.used_stock_idx = []
            self.prev_idx = 0

    def get_action(self, observation, info):

        if self.policy_id == 1:
          return self.get_action_FFD_modified(observation, info)
        elif self.policy_id == 2:
          return self.get_action_CCF(observation, info)

    ###################################################### Define methods for both policy ######################################################
    # Function to sort products
    def SortProducts(self, products):
      return sorted(products, key=lambda x:-x["size"][0]*x["size"][1])
    
    # Function to sort stocksheets
    def SortStockSheet(self, stocks):
      self.stock_info_list = []
      self.stock_idx = []
      self.stock_sheets = []
      for i, stock in enumerate(stocks):
        w, h = self._get_stock_size_(stock)
        self.stock_info_list.append((w*h, i, stock))
      self.stock_info_list = sorted(self.stock_info_list, key=lambda x: x[0], reverse=True)
      self.stock_idx = [item[1] for item in self.stock_info_list]
      self.stock_sheets = [item[2] for item in self.stock_info_list]
    
    #Function to determine if we could cut the product on a position on a stocksheet
    def CanCut(self, stockSheet, pos, prod_size):
      return self._can_place_(stockSheet, pos, prod_size)
    
    ###################################################### End define methods for both policy ######################################################   



    ###################################################### Define methods for FFD modified ######################################################    
    #Function to cut, rotate
    def CutProduct(self, product_size, stockSheet):
      stock_width, stock_height = self._get_stock_size_(stockSheet)
      product_width, product_height = product_size
      # if product is not cuttable in this current stock, move to the next stock
      if (product_width < 0 or product_width > stock_width or product_height < 0 or product_height > stock_height):
        pass
      else:
        product_size = product_size
        #Initialize the first cutting posititon, (-1, -1) means the product hasn't been cut
        cut_position = (-1, -1)
        # Iterate through every possible place that can cut the product
        for x in range(0, stock_width - product_width + 1):
          for y in range(0, stock_height - product_height + 1):
            # If can cut the product, save the cut posittion
            if (stockSheet[x][y] != -1):
              continue
            else:
              if (self.CanCut(stockSheet, (x, y), product_size)):
                cut_position = (x, y)
                product_size = product_size
                break
          # If finded cuttable posittion, out the loop
          if (cut_position[0] != -1 and cut_position[1] != -1):
            break
      return cut_position, product_size

    # Action for FFD
    def get_action_FFD_modified(self, observation, info):
      # Student code here
      if (self.first_action == False or info["filled_ratio"] == 0.00):
        self.first_action = True
        # Sort stock sheets based on area in decending order 
        self.SortStockSheet(observation["stocks"])
        # Sort products based on area in decending order    
        self.sorted_list_prods = self.SortProducts(observation["products"])

      for i in range (0, len(self.stock_sheets)):
        for product in self.sorted_list_prods:
          if product["quantity"] > 0:
            cut_position, product_size = self.CutProduct(product["size"], self.stock_sheets[i])
            # Rotate to find if rotated product could fit the remain area
            if (cut_position[0] == -1 or cut_position[1] == -1):
              cut_position, product_size = self.CutProduct(product["size"][::-1], self.stock_sheets[i])
            
            if cut_position[0] != -1 and cut_position[1] != -1:
              stock_idx = self.stock_idx[i]
              break
            else:
              continue
        if cut_position[0] != -1 and cut_position[1] != -1:
          break
        else:
          continue
      return {"stock_idx": stock_idx, "size": product_size, "position": cut_position}
    ###################################################### End define methods for FFD modified ######################################################



    ###################################################### Define methods for CCF algorithm ######################################################
    def update_products(self, observation):
        self.products.clear()
        self.residual_vector.clear()
        NewDictProducts = observation["products"]
        SortedListProds = self.SortProducts(NewDictProducts)
        # products list after sorted and eliminate product with quantity == 0
        for product in SortedListProds:
            if (product["quantity"] == 0):
                continue
            else:
              self.products.append(product)
              self.residual_vector.append(product["quantity"])

    def GetUndominateProducts(self):
        numProd = len(self.products)
        self.undominateProds = []
        for i in range (0, numProd):
            dominatedFlag = False
            current_prod_width, current_prod_height = self.products[i]["size"]
            for j in range(0, numProd):
                  other_width, other_height = self.products[j]["size"]
                  if (current_prod_height < other_height and current_prod_width < other_width):
                      dominatedFlag = True
            if (dominatedFlag == True):
                continue
            else:
                self.undominateProds.append(self.products[i])  

    def calc_differ_neigbors(self, component_pos, currStockSheet, blockSize, original_width, original_height):
        try:
            neigborLeft = 0
            x = component_pos[0]
            y = component_pos[1]
            while (currStockSheet[x][y] > 0 and y >= 0 and y <= original_height):
              neigborLeft += 1
              y += 1
            y = component_pos[1]
            while (currStockSheet[x][y] > 0 and y >= 0 and y <= original_height):
              neigborLeft+=1
              y -= 1
        except:
          neigborLeft = 0
        # try:
        #     neigborRight = 0
        #     x = component_pos[0] + blockSize[0]
        #     y = component_pos[1]
        #     while(currStockSheet[x][y] > 0 and y >= 0 and y <= original_height):
        #       neigborRight += 1
        #       y += 1
        #     y = component_pos[1]
        #     while(currStockSheet[x][y] > 0 and y >= 0 and y <= original_height):
        #       neigborRight += 1
        #       y -= 1
        # except:
        #     neigborRight = 0
        try:
            neigborUp = 0
            x = component_pos[0]
            y = component_pos[1]
            while currStockSheet[x][y] > 0 and x >= 0 and x <= original_width:
              neigborUp += 1
              x += 1
            x = component_pos[0]
            while currStockSheet[x][y] > 0 and x >= 0 and x <= original_width:
              neigborUp += 1
              x -= 1
        except:
            neigborUp = 0
        # try:
        #     neigborDown = 0
        #     x = component_pos[0]
        #     y = component_pos[1] + blockSize[1]
        #     while currStockSheet[x][y] != -1 and x >= 0 and x <= original_width:
        #       neigborDown += 1
        #       x += 1
        #     x = component_pos[0]
        #     while currStockSheet[x][y] != -1 and x >= 0 and x <= original_width:
        #       neigborDown += 1
        #       x -= 1
        # except:
        #     neigborDown =  0
        # n = abs(neigborLeft - blockSize[1]) + abs(neigborRight - blockSize[1]) + abs(neigborUp - blockSize[0]) + abs(neigborDown - blockSize[0])
        n = abs(neigborLeft - blockSize[1]) + abs(neigborUp - blockSize[0])
        return n

    def get_action_CCF(self, observation, info):
        if self.first_action == False or info["filled_ratio"] == 0:
          self.first_action = True
          self.products = self.SortProducts(observation["products"])
          self.SortStockSheet(observation["stocks"])
          self.currStock = self.stock_sheets[self.curr_idx]
          self.component_stack_for_curr_stock.append(((0, 0), self._get_stock_size_(self.currStock)))
          self.used_stock_idx.append(self.curr_idx)
          

        if (len(self.sequencePosReturn) == 0) and len(self.component_stack_for_curr_stock) == 0:
           self.update_products(observation)
           self.curr_idx += 1
           self.curr_idx = self.curr_idx%100
           self.used_stock_idx.append(self.curr_idx)
           self.currStock = self.stock_sheets[self.curr_idx]
           self.component_stack_for_curr_stock.clear()
           self.component_stack_for_curr_stock.append(((0, 0), self._get_stock_size_(self.currStock)))
           # Try to use leftover space
           try:
            self.prev_idx = self.used_stock_idx[0]
            previous_stock = self.stock_sheets[self.prev_idx]
            prev_stock_w, prev_stock_h = self._get_stock_size_(previous_stock)
            for prod in self.products:
              for x in range (0, prev_stock_w):
                  for y in range (0, prev_stock_h):
                    if (previous_stock[x][y] != -1):
                       pass
                    else:
                      if self._can_place_(previous_stock, (x, y), prod["size"]):
                        return {"stock_idx": self.stock_idx[self.prev_idx], "size": prod["size"], "position": (x, y)}
                      elif self._can_place_(previous_stock, (x, y), prod["size"][::-1]):
                        return {"stock_idx": self.stock_idx[self.prev_idx], "size": prod["size"][::-1], "position": (x, y)}
            self.used_stock_idx.pop(0)
           except:
              pass
           self.used_stock_idx.pop(0)

        
        while(len(self.sequencePosReturn) == 0 and len(self.component_stack_for_curr_stock) != 0):
           component = self.component_stack_for_curr_stock.pop(0)
           c_w = component[1][0]
           c_h = component[1][1]
           c_x = component[0][0]
           c_y = component[0][1]
           maxFitness = -100
           bestProdSize = (0, 0)
           bestBlockSize = (0, 0)
           best_n_w = 0
           best_n_h = 0
           for prod in self.products:
              p_w, p_h = prod["size"]
              d = prod["quantity"]
              # FOR NO ROTATION BLOCK
              if p_w > c_w or p_h > c_h:
                 pass
              else:
                 grid_w = math.floor(c_w/p_w)
                 grid_h = math.floor(c_h/p_h)
                 num = min(grid_h*grid_w, d)
                 n_w = math.floor(math.sqrt(num))
                 n_h = math.ceil(math.sqrt(num))
                 blockSize = (n_w*p_w, n_h*p_h)
                 if (blockSize[0] > c_w or blockSize[1] > c_h or not self._can_place_(self.currStock, (c_x, c_y), blockSize)):
                    pass
                 else:
                    d_w = c_w - blockSize[0]
                    d_h = c_h - blockSize[1]
                    n = self.calc_differ_neigbors((c_x, c_y), self.currStock, blockSize, c_w, c_h)
                    fitness = 1/((d_w+1)*(d_h+1)*(n+1))
                    if (fitness > maxFitness):
                       maxFitness = fitness
                       bestProdSize = (p_w, p_h)
                       bestBlockSize = blockSize
                       best_n_w = n_w
                       best_n_h = n_h
                    else:
                       pass
              
              # FOR ROTATION BLOCK
              if p_h > c_w or p_w > c_h:
                  pass
              else:
                  grid_w = math.floor(c_w / p_h)
                  grid_h = math.floor(c_h / p_w)
                  num = min(grid_h*grid_w, d)
                  n_w = math.floor(math.sqrt(num))
                  n_h = math.ceil(math.sqrt(num))
                  blockSize = (n_w*p_h, n_h *p_w)
                  if (blockSize[0] > c_w or blockSize[1] > c_h or not self._can_place_(self.currStock, (c_x, c_y), blockSize)):
                     pass
                  else:
                    d_w = c_w - blockSize[0]
                    d_h = c_h - blockSize[1]
                    n = self.calc_differ_neigbors((c_x, c_y), self.currStock, blockSize, c_w, c_h)
                    fitness = 1/((d_w+1)*(d_h+1)*(n+1))
                    if (fitness > maxFitness):
                       maxFitness = fitness
                       bestProdSize = (p_h, p_w)
                       bestBlockSize = blockSize
                       best_n_w = n_w
                       best_n_h = n_h
                    else:
                       pass
      
           if (maxFitness < 0 or bestBlockSize == (0, 0)):
              pass
           else:
              for i in range (0, best_n_w):
                 for j in range(0, best_n_h):
                    self.sequencePosReturn.append((self.stock_idx[self.curr_idx], (c_x + i*bestProdSize[0], c_y + j * bestProdSize[1]), bestProdSize))

              component1 = ((c_x + bestBlockSize[0], c_y), (c_w - bestBlockSize[0], c_h))
              self.component_stack_for_curr_stock.insert(0,component1)
              component2 = ((c_x, c_y + bestBlockSize[1]), (bestBlockSize[0], c_h - bestBlockSize[1]))
              self.component_stack_for_curr_stock.insert(0,component2)

        while(len(self.sequencePosReturn) != 0):
           step = self.sequencePosReturn.pop(0)
           index = step[0]
           pos = step[1]
           prodSize = step[2]
           return {"stock_idx": index, "size": prodSize, "position": pos}
        
        return {"stock_idx": 0, "size": (0, 0), "position": (0, 0)}
        # CONTINUE

    ###################################################### End define mothods for CCF algorithm ######################################################