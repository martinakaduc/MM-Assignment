'''
Description: Simplex application for 2D cutting-stock
author: Phạm Lê Tiến Đạt - 2310687
Time: Dec, 2024
'''

import gym_cutting_stock
import gymnasium as gym
from CuttingStockEnv import CuttingStockEnv
import numpy as np
from scipy.optimize import linprog
from scipy.ndimage import label
from time import time, sleep
from policy import Policy

# First Fit Policy:
class FFPolicy(Policy):
    def __init__(self):
        self.c_prod=0
        self.c_stock=0
        self.idx_stock=[]
        self.list_prod=[]
        self.prod_area_left=0
        self.m=True
    def ___stock_area___(self,stock):
        x,y=self._get_stock_size_(stock)
        return x*y
    def ___csize___(self,stock):
        return  np.max(np.sum(np.any(stock>-1, axis=1))),np.max(np.sum(np.any(stock>-1, axis=0)))
    def get_action(self, observation, info):
        if(info["filled_ratio"]==0):
            self.__init__()
            for prod in observation["products"]:
                self.prod_area_left+=np.prod(prod["size"])*prod["quantity"]
            self.prod_area=self.prod_area_left
            self.idx_stock=sorted(enumerate(observation["stocks"]),key=lambda x:self.___stock_area___(x[1]),reverse=True)
            self.list_prod =sorted(observation["products"],key=lambda x: x["size"][0]*x["size"][1],reverse=True)
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = None, None
        c_stock=-1
        for i,_ in self.idx_stock:
            c_stock+=1
            if c_stock<self.c_stock: continue
            stock=observation["stocks"][i]
            stock_w, stock_h = self._get_stock_size_(stock)
            if self.prod_area<stock_h*stock_w*0.75 and c_stock<len(self.idx_stock)-1 and self.m: 
                self.c_stock+=1
                continue
            c_prod=-1
            cx,cy=self.___csize___(stock)
            for prod in self.list_prod:
                c_prod+=1
                if c_prod<self.c_prod: continue
                if prod["quantity"] > 0:
                    prod_size=prod["size"]
                    prod_w, prod_h =  prod_size
                    if stock_w < prod_w or stock_h < prod_h: continue
                    for x in range(stock_w - prod_w + 1):
                        if x>cx: continue
                        for y in range(stock_h - prod_h + 1):
                            if y>cy: continue
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x=x
                                pos_y=y
                                break
                        if pos_x is not None:
                            break   
                    if pos_x is not None:
                        break
                self.c_prod+=1
            if pos_x is not None:
                stock_idx = i
                self.prod_area_left-=np.prod(prod_size)
                break
            if self.c_stock==len(self.idx_stock)-1:
                self.c_stock-=2
                self.m =False
            self.c_stock+=1
            self.c_prod=0
            self.prod_area=self.prod_area_left
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

# Simplex solve function
def simplex_method(stock_sizes, demand, item_sizes, stock):
    num_stocks = len(stock_sizes)		#so luong stock
    num_items = len(demand)		# so luong item yeu cau
    A = np.zeros((num_items + num_stocks, num_stocks))
    array_zero = np.zeros(num_stocks)
    b = np.array(demand)
    b = np.concatenate((b, array_zero))
    # Định nghĩa hàm dếm sô lượng từng item
    def num_item_per_stock(i, stock):
        item_mask = (stock == i)
        _, num_features = label(item_mask)
        return num_features
        
    for j in range(num_stocks):
        for i in range(num_items): 
            A[i, j] = num_item_per_stock(i + 1, stock[j])
 
    for i in range(num_stocks):
        A[i + num_items, i] = 1
    c = np.ones(num_stocks)  # Hàm mục tiêu
    # Giai
    result = linprog(c, A_ub=-A, b_ub=-b)

    # In ra kết quả
    if result.success:
        return np.round(result.x).astype(np.int32)
    else:
        return None
    
env =CuttingStockEnv(
    #  render_mode="human"
                     )


NUM_EPISODES = 1

if __name__ == "__main__":
    # Reset the environment
    observation, info = env.reset(seed = 42)
    # Khoi tao cac tham so cho simplex 
    demand = []
    item_sizes = []
    for item in observation["products"]:
        demand.append(item["quantity"])
        item_sizes.append(item["size"])

    sp_policy = FFPolicy()
    ep = 0
    action_chain = dict()
    start_time = time()
    while ep < NUM_EPISODES:
        action = sp_policy.get_action(observation, info)
        # luu action vao
        if action['stock_idx'] not in action_chain:
            action_chain[action['stock_idx']] = [action]
        else:
            action_chain[action['stock_idx']].append(action)

        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            sleep(2)
            num_stocks = len(observation["stocks"])
            num_items = len(observation["products"])


            stock_sizes = []

            for stock in observation["stocks"]:
                stock_width = np.sum(np.any(stock != -2, axis=1))
                stock_height = np.sum(np.any(stock != -2, axis=0))
                stock_sizes.append((stock_width, stock_height))

            solution = simplex_method(stock_sizes, demand, item_sizes, observation["stocks"])

            # Do simplex
            if solution is not None:
                print("------------- ACTION CHAIN ------------")
                for index, x in enumerate(solution):
                    print(x)
                    if(x >= 0.5):
                        for j in range(round(x)):
                            for k in range(len(action_chain[index])):
                                # action = action_chain[index][k]
                                print(f"size {action_chain[index][k]['size']}, position: {(action_chain[index][k]['position'])}")
                                # observation, reward, terminated, truncated, info = env.step(action)
            else:
                print("No solution")
            end_time = time()
            print("Time: ", round(end_time-start_time,2), "second")
            print("Num of stocks: ", num_stocks, "- Number of items:", num_items)
            print("-------------------------")
            print(f"Demand: {demand}")
            print("-------------------------")
            print("Simplex: ")
            if solution is not None:
                print(f"Solution: {solution}")
                print("Total stock in simplex:", np.sum(solution[solution > 0]))
            else:
                print("No solution")
            print("-------------------------")
            print("Stock cut Before Simplex: ", env.cutted_stocks)
            print("Total stock cut: ", np.sum(env.cutted_stocks[env.cutted_stocks > 0]))
            observation, info = env.reset(seed=ep)
            ep += 1
    
env.close()
