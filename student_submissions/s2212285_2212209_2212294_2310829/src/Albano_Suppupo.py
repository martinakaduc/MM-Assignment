from policy import Policy
import numpy as np

class Albano_Suppuno(Policy):
    def __init__(self):
        self.used_stock = []
        self.corner_point=[]
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
    # Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_products = sorted(
            list_prods,
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )
        index = 0
        #print(len(sorted_products))
        for prod in sorted_products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                # Duyệt qua các góc tiềm năng
                for i, stock, point in self.corner_point:
                    for size in [prod_size, prod_size[::-1]]:  # Thử cả hướng xoay
                        stock_size = self._get_stock_size_(stock)
                        stock_w,stock_h = stock_size
                        if stock_w >= point[0]+size[0] and stock_h >= point[1]+size[1]:
                            if self._can_place_(stock, point, size):
                                # Cập nhật corner_point
                                self.corner_point.remove((i, stock, point))
                                self.corner_point.extend([
                                    (i, stock, (point[0] + size[0], point[1])),
                                    (i, stock, (point[0], point[1] + size[1]))
                                ])
                                if index == len(sorted_products) - 1 and prod["quantity"] == 1:
                                    self.used_stock = []
                                    self.corner_point = []
                                return {"stock_idx": i, "size": size, "position": point}

                # Nếu không có vị trí phù hợp, thêm kho mới
                if not self.used_stock:
                    new_stock_idx = 0
                else:
                    if index == len(sorted_products) - 1 and  prod["quantity"] == 1:
                        self.used_stock = []
                        self.corner_point = []
                        return {"stock_idx": self.used_stock[-1] + 1, "size": prod_size, "position": (0, 0)}
                    else:
                        new_stock_idx = self.used_stock[-1] + 1

                self.used_stock.append(new_stock_idx)
                new_stock = observation["stocks"][new_stock_idx]

                # Thêm góc cho kho mới
                self.corner_point.extend([
                    (new_stock_idx, new_stock, (prod_size[0], 0)),
                    (new_stock_idx, new_stock, (0, prod_size[1]))
                ])
                return {"stock_idx": new_stock_idx, "size": prod_size, "position": (0, 0)}