from policy import Policy
import numpy as np


class Policy2313221_2311269_2313765_2313446_2213989(Policy):  # Ngẫu nhiên từ trên xuống dưới, trái qua phải, lấy theo thứ tự, nhưng chỉ lấy stock mới khi stock đầu đã đầy
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = bestfit()
        elif policy_id == 2:
            self.policy = left_bottom()
    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

class bestfit(Policy):
    def __init__(self):
        pass
    def get_action(self, observation, info):
        list_prods = observation.get("products", [])
        stocks = observation.get("stocks", [])

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        #Duyệt qua mẫu
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Tìm stock vừa
                for i, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Skip stock nếu k vừa
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # tìm vị trí phù hợp
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                stock_idx = i
                                pos_x, pos_y = x, y
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        # nếu không trả về mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

class left_bottom(Policy):  # Lấy product ngẫu nhiên từ trên xuống dưới, trái qua phải, lấy stock theo thứ tự
    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stock_idx = -1
        pos_x, pos_y = None, None

        # Duyệt qua từng kho theo thứ tự
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Duyệt qua từng sản phẩm
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    # Kiểm tra nếu sản phẩm có thể đặt vào kho
                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    # Áp dụng thuật toán Bottom-Left để tìm vị trí phù hợp
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                # Nếu đặt được sản phẩm, trả về hành động
                                return {
                                    "stock_idx": i,
                                    "size": prod_size,
                                    "position": (x, y),
                                }
                            

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
