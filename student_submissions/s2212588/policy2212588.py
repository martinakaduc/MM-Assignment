import random
from abc import abstractmethod
from policy import Policy
import numpy as np 

class Policy2212588(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # Khởi tạo policy con tùy thuộc vào policy_id
        if policy_id == 1:
            self.policy = FirstFitDecreasingPolicy()
        elif policy_id == 2:
            self.policy = BestFitDecreasingPolicy()

    def get_action(self, observation, info):
        # Gọi hàm `get_action` từ policy con
        return self.policy.get_action(observation, info)
    
class FirstFitDecreasingPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        prod_size = [0, 0]

        # Sort products by area in decreasing order
        list_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {
                                    "stock_idx": i,
                                    "size": prod_size,
                                    "position": (x, y),
                                }
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                prod_size = prod_size[::-1]
                                return {
                                    "stock_idx": i,
                                    "size": prod_size,
                                    "position": (x, y),
                                }

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


class BestFitDecreasingPolicy(Policy):
    def __init__(self):
        pass
    
    def get_action(self, observation, info):
        list_prods = observation["products"]
        
        # Sắp xếp sản phẩm theo diện tích giảm dần
        list_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        stock_idx = -1
        pos_x, pos_y = 0, 0
        prod_size = [0, 0]

        # Duyệt qua từng sản phẩm
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                best_fit_stock_idx = -1
                best_fit_position = None
                min_waste_area = float('inf')

                # Duyệt qua tất cả các tấm stock
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for x in range(stock_w - prod_size[0] + 1):  # Duyệt qua các vị trí có thể
                        for y in range(stock_h - prod_size[1] + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                # Tính toán không gian thừa nếu đặt sản phẩm vào đây
                                remaining_area = (stock_w - x) * (stock_h - y) - (prod_size[0] * prod_size[1])
                                
                                if remaining_area < min_waste_area:
                                    min_waste_area = remaining_area
                                    best_fit_stock_idx = i
                                    best_fit_position = (x, y)

                if best_fit_stock_idx != -1 and best_fit_position is not None:
                    stock_idx = best_fit_stock_idx
                    pos_x, pos_y = best_fit_position
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
