from policy import Policy
import numpy as np

class Policy2252612_2252645_2152822(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            self.flag = 0
            self.waiting_stocks = set()
            self.arrange_area = set()
            self.being_used_stocks = set()
            self.sorted_products=set()
            self.num_prod=0
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self.get_action_skyline(observation, info)
        elif self.policy_id == 2:
            return self.get_action_bfff(observation, info)

    def get_action_skyline(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]

        for prod_index, prod in enumerate(products):
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # try to put product
                best_stock_index, prod_size, best_position = self._find_and_place_(
                    stocks, prod_size, -1, None
                )
                if best_stock_index != -1 and best_position is not None:
                    products[prod_index]["quantity"] -= 1
                    return {"stock_idx": best_stock_index, "size": prod_size, "position": best_position}

                # try to rotate product
                best_stock_index, prod_size, best_position = self._find_and_place_(
                    stocks, prod_size[::-1], -1, None
                )
                if best_stock_index != -1 and best_position is not None:
                    products[prod_index]["quantity"] -= 1
                    return {"stock_idx": best_stock_index, "size": prod_size[::-1], "position": best_position}

        print("No valid position found for any product")
        return {"stock_idx": -1, "size": [0, 0], "position": None}

    def _get_skyline_(self, stock,prod_size):
        skyline = []
        stock_w, stock_h = self._get_stock_size_(stock)

        # Create a list to store the height of the skyline for each column
        heights = [-1] * stock_h

        # Iterate through the stock and find the highest position for each column
        for i in range(stock_w):
            for j in range(stock_h):
                if (self._can_place_(stock,(i,j),prod_size)):
                        heights[j] = i
        # Create skyline from the heights
        for j in range(stock_h):
            if heights[j] != -1:
                    skyline.append((heights[j],j))
            else:
                    skyline.append((stock_w,j))
            
        return skyline #this policy is trying to implement a skyline algorithm
    def _get_used_stock(self,list_used_stock):
        used_stocks = 0
        for i in range(len(list_used_stock)):
            if (list_used_stock[i] == 1):
                used_stocks += 1
        # calculate number of used stocks
        return used_stocks
    def _find_and_place_(self, stocks, prod_size, best_stock_index, best_position):
        min_skyline_height = float('inf')

        for stock_index, stock in enumerate(stocks):
            skyline = self._get_skyline_(stock, prod_size)
            stock_w, stock_h = self._get_stock_size_(stock)

            for pos_x, pos_y in skyline:
                if self._can_place_(stock, (pos_x, pos_y), prod_size):
                    current_skyline_height = pos_x
                    if current_skyline_height < min_skyline_height:
                        min_skyline_height = current_skyline_height
                        best_stock_index = stock_index
                        best_position = (pos_x, pos_y)

            # if find the best position, mark used
            if best_position is not None and best_stock_index != -1:
                return best_stock_index, prod_size, best_position

        # didn't find any suitable stocks
        return -1, prod_size, None
    
    def arrange_index_stock(self, stocks):
        count_of_minus_ones = []
        for stock in stocks:
            count = np.count_nonzero(stock == -1)
            count_of_minus_ones.append(count)
        return sorted(range(len(count_of_minus_ones)), key=lambda i: count_of_minus_ones[i], reverse=True), sorted(count_of_minus_ones, reverse = True)

    def prod_area(self, produc):
        total_prod = 0
        for i in produc:
            total_prod+=i["quantity"]
        return sorted(produc, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True), total_prod

    def get_action_bfff(self, observation, info):
        if(self.flag==0):
            self.waiting_stocks = set()
            self.arrange_area = set()
            self.being_used_stocks = set()
            self.sorted_products=set()

            self.waiting_stocks, self.arrange_area = self.arrange_index_stock(observation['stocks'])
            self.sorted_products, self.num_prod = self.prod_area(observation["products"])
            self.flag = 1

        set_stocks = observation["stocks"]

        for prod in self.sorted_products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Try to fit the piece in stocks in the 'being used' group using the best-fit algorithm
                best_stock_idx = -1
                best_position = None
                best_wasted_space = float('inf')

                for stock_idx in self.being_used_stocks:
                    stock = set_stocks[stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                # Calculate wasted space if this piece is placed
                                remaining_w = stock_w - prod_w
                                remaining_h = stock_h - prod_h
                                wasted_space = (remaining_w * stock_h + stock_w * remaining_h - remaining_w * remaining_h)

                                # Update best-fit stock if wasted space is minimal
                                if wasted_space < best_wasted_space:
                                    best_wasted_space = wasted_space
                                    best_stock_idx = stock_idx
                                    best_position = (x, y)

                    if best_position is not None:
                        self.num_prod-=1
                        if(self.num_prod==0):
                            self.flag=0
                        return {"stock_idx": best_stock_idx, "size": prod_size, "position": best_position} 
                    else:
                        prod_size = [int(prod_h), int(prod_w)]
                        prod_w, prod_h = prod_size

                        for stock_idx in self.being_used_stocks:
                            stock = set_stocks[stock_idx]
                            stock_w, stock_h = self._get_stock_size_(stock)

                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        # Calculate wasted space if this piece is placed
                                        remaining_w = stock_w - prod_w
                                        remaining_h = stock_h - prod_h
                                        wasted_space = (remaining_w * stock_h + stock_w * remaining_h - remaining_w * remaining_h)

                                        # Update best-fit stock if wasted space is minimal
                                        if wasted_space < best_wasted_space:
                                            best_wasted_space = wasted_space
                                            best_stock_idx = stock_idx
                                            best_position = (x, y)

                            if best_position is not None:
                                self.num_prod-=1
                                if(self.num_prod==0):
                                    self.flag=0
                                return {"stock_idx": best_stock_idx, "size": prod_size, "position": best_position}   

                for stock_idx in self.waiting_stocks:
                    stock = set_stocks[stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                self.being_used_stocks.add(stock_idx)
                                self.waiting_stocks.remove(stock_idx)
                                self.num_prod-=1
                                if(self.num_prod==0):
                                    self.flag=0
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (x,y)}                         
                
        # If no action could be taken (this case shouldn't normally occur)
        return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}
