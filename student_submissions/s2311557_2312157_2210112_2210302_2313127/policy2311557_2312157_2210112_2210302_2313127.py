from policy import Policy
import random
import numpy as np
import math
class Policy2311557_2312157_2210112_2210302_2313127(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Xác định chính sách
        self.policy_id = policy_id
        if policy_id == 1:
            self.policy = FirstFitDecreasingPolicy()
        elif policy_id == 2:
            self.policy = BestFitPolicy()

    def get_action(self, observation, info):
        # Gọi get_action từ chính sách đã chọn
        return self.policy.get_action(observation, info)
class FirstFitDecreasingPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Sắp xếp các sản phẩm theo kích thước giảm dần
        sorted_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Duyệt qua tất cả các stock sheets
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
class BestFitPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Danh sách các sản phẩm đã được xử lý
        handled_products = []

        # Tiến hành tìm sản phẩm có quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0 and prod not in handled_products:
                handled_products.append(prod)  # Đánh dấu sản phẩm đã xử lý

                prod_size = prod["size"]
                best_fit = None
                best_waste = float('inf')  # Mặc định là vô cùng lớn

                # Tìm stock phù hợp nhất cho sản phẩm
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Kiểm tra nếu stock có thể chứa sản phẩm
                    if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                        for x in range(stock_w - prod_size[0] + 1):
                            for y in range(stock_h - prod_size[1] + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    # Tính toán waste nếu cắt sản phẩm ở vị trí này
                                    waste = (stock_w * stock_h) - (prod_size[0] * prod_size[1])

                                    if waste < best_waste:
                                        best_waste = waste
                                        best_fit = (i, (x, y), prod_size)

                    # Kiểm tra khi xoay sản phẩm
                    if stock_w >= prod_size[1] and stock_h >= prod_size[0]:
                        for x in range(stock_w - prod_size[1] + 1):
                            for y in range(stock_h - prod_size[0] + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    # Tính toán waste nếu cắt sản phẩm xoay
                                    waste = (stock_w * stock_h) - (prod_size[1] * prod_size[0])

                                    if waste < best_waste:
                                        best_waste = waste
                                        best_fit = (i, (x, y), prod_size[::-1])

                if best_fit:
                    stock_idx, pos, prod_size = best_fit
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": pos}
