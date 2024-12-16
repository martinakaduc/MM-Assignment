from policy import Policy
import numpy as np

class Policy2310210_2310370_2310730_2310624_2312955(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = 1
        elif policy_id == 2:
            self.policy = 2
            
    policy = 1
    last_prod = None
    last_stock_idx = -1


    def get_action(self, observation, info):
        # Student code here
        if self.policy == 1:
            return self.First_Fit_Decreasing(observation, info)
        if self.policy == 2:
            return self.Best_Fit_Decreasing(observation, info)

    # Student code here
    def First_Fit_Decreasing(self, observation, info):
        list_prods = observation["products"]
        list_prods = sorted(list_prods, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)

        list_stocks = list(enumerate(observation["stocks"]))
        
        # This line is stock sort: largest to smallest ( comment # it if you want to used normal stock)
        list_stocks.sort(key=lambda list: np.sum(list[1] != -2), reverse=True)

        return self.Cutting(list_prods, list_stocks, info)




    def Best_Fit_Decreasing(self, observation, info):
        list_prods = observation["products"]
        list_prods = sorted(list_prods, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)

        list_stocks = list(enumerate(observation["stocks"]))

        # This line is normal "best fit" stock sort (currently unused)
        # list_stocks.sort(key=lambda list: np.sum(list[1] == -1))

        # 5 lines below are stock sorted: largest to smallest + "best fit" sort
        filter_list = [stock for stock in list_stocks if np.sum(stock[1] >= 0)]
        filter_list.sort(key=lambda list: np.sum(list[1] >= 0), reverse=True)

        remain_list = [stock for stock in list_stocks if np.all(stock[1] < 0)]
        remain_list.sort(key=lambda list: np.sum(list[1] != -2), reverse=True)

        list_stocks = filter_list + remain_list


        return self.Cutting(list_prods, list_stocks, info)        

    # You can add more functions if needed
    def Cutting(self, list_prods, list_stocks, info):
        stock_idx = -1
        prod_size = [0, 0]
        last_prod_size = [0, 0]
        pos_x, pos_y = 0, 0
        cutted = 0
        jump_check = 1

        if self.last_prod is not None:
            last_prod_size = self.last_prod["size"]


        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for i, stock in list_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size


                    #This here is fast check
                    #need to make it dont recheck if a prod already check all previous stock and fail, and only current stock available
                    #since to reach last stock point, you need to pass all previous point in the first prod, so next prod just need to skip (continue) without fearing skipping unchecked stock
                    if (self.last_stock_idx == i and  jump_check == 1) or self.last_stock_idx == -1:
                        jump_check = 0
                        
                    if  last_prod_size[0] == prod_w and last_prod_size[1] == prod_h and jump_check == 1:
                        #print("jumped stock" , i)
                        continue

                    current_area_left =  np.sum(stock == -1)
                    if current_area_left < prod_w * prod_h:
                        #print("skip current stock", i)
                        continue

                    if stock_w >= prod_w and stock_h >= prod_h and cutted == 0:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    cutted = 1
                                    break
                            if cutted == 1:
                                break


                    if stock_w >= prod_h and stock_h >= prod_w and cutted == 0:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    pos_x, pos_y = x, y
                                    cutted = 1
                                    break
                            if cutted == 1:
                                break


                    if cutted == 1:
                        stock_idx = i
                        self.last_prod = prod
                        self.last_stock_idx = stock_idx
                        break


                if cutted == 1:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}