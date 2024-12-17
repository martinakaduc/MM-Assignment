'''
Description: First fit for 2D cutting-stock
author: Phạm Lê Tiến Đạt - 2310687
Time: Nov, 2024
'''

from policy import Policy
import numpy as np


class Policy_first_fit(Policy):
    def __init__(self):
        pass
    def get_action(self, observation, info):
    	# Lay danh sach stock va item
        list_prods = observation["products"]
        list_stock = enumerate(observation["stocks"])
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = None, None
        # xet theo thu tu stock dau tien tro di (first fit)
        for i, stock in list_stock:
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod in list_prods:
            	# Lay item co quantity > 0
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                        	# Xet vi tri tu trai sang phai tu tren xuong duoi va kiem tra vi tri do trong khong
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                if pos_x is not None and pos_y is not None:
                    break
            # luu lai vi tri cua stock ma chen item vao
            if pos_x is not None and pos_y is not None:
                stock_idx = i
                break
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}