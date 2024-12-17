'''
Description: Best fit for 2D cutting-stock
author: Lê Trọng Thiện - 2313233
Time: Nov, 2024
'''

from policy import Policy
import numpy as np
from copy import deepcopy

class BFPolicy(Policy):
    def __init__(self):
        self.c_stock=0
        self.idx_stock=[]
        self.list_prod=[]
        self.prod_area_left=0
        self.prod_area=0
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
            if self.prod_area<stock_h*stock_w*0.75 and c_stock<len(self.idx_stock)-1 and self.m : 
                self.c_stock+=1
                continue
            cx,cy=self.___csize___(stock)
            # c_prod=-1
            
            min_dS=-1
            min_size = [0, 0]
            for prod in self.list_prod:
                # c_prod+=1
                # if c_prod<self.c_prod: continue
                if prod["quantity"] > 0:
                    min_size=prod["size"]
                    prod_w, prod_h =  min_size
                    if stock_w < prod_w or stock_h < prod_h: continue
                    for x in range(min(stock_w - prod_w + 1,cx+1)):
                        for y in range(min(stock_h - prod_h + 1,cy+1)):
                            if self._can_place_(stock, (x, y), min_size):
                                new_S=max(cy,y+prod_h)*max(cx,x+prod_w)
                                dS=(new_S-cx*cy)/(prod_h*prod_w)
                                if dS<min_dS  or min_dS==-1:
                                    prod_size=min_size
                                    pos_x=x
                                    pos_y=y
                                    min_dS=dS
                    min_size=prod["size"][::-1]
                    prod_w, prod_h =  min_size
                    if stock_w < prod_w or stock_h < prod_h: continue
                    for x in range(min(stock_w - prod_w + 1,cx+1)):
                        for y in range(min(stock_h - prod_h + 1,cy+1)):
                            if self._can_place_(stock, (x, y), min_size):
                                new_S=max(cy,y+prod_h)*max(cx,x+prod_w)
                                dS=(new_S-cx*cy)/(prod_h*prod_w)
                                if dS<min_dS  or min_dS==-1:
                                    prod_size=min_size
                                    pos_x=x
                                    pos_y=y
                                    min_dS=dS
            if pos_x is not None:
                stock_idx = i
                self.prod_area_left-=np.prod(prod_size)
                break
            if self.c_stock==len(self.idx_stock)-1:
                self.c_stock-=2
                self.m =False
            self.c_stock+=1
            self.prod_area=self.prod_area_left
            # self.c_prod=0
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}