
import numpy as np
from policy import Policy


class Policy2310250_2310365_2310381_2310251(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            super().__init__()
            pass
        self.policy_id = policy_id
    def get_action(self, observation, info):
        # Student code here
        if self.policy_id ==1:
            return self.get_action_BFD(observation,info)
        else:
            return self.get_action_BLD(observation,info)

#-------------------------------BFD------------------------------#
    def get_action_BFD(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]

        # Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_products = sorted(
            products, key=lambda p: p["size"][0] * p["size"][1], reverse=True
        )

        for product in sorted_products:
            if product["quantity"] > 0:
                size = product["size"]

                best_fit_idx = -1
                best_fit_score = float('inf')

                # Duyệt qua các thanh để tìm thanh phù hợp nhất
                for stock_idx, stock in enumerate(stocks):
                    stock_width, stock_height = self._get_stock_size_(stock)
                    prod_w, prod_h = size
                    if stock_width < prod_w or stock_height < prod_h:
                        continue
                    # Tính toán không gian trống còn lại trong thanh
                    for x in range(stock_width - size[0] + 1):
                        for y in range(stock_height - size[1] + 1):
                            # if np.all(stock[x:x + size[0], y:y + size[1]] == -1):
                            if self._can_place_(stock,(x, y),size):
                                empty_space = self.calculate_space_left(stock,size)
                                # Cập nhật thanh tốt nhất nếu có không gian trống nhỏ nhất
                                if empty_space < best_fit_score:
                                    best_fit_score = empty_space
                                    best_fit_idx = stock_idx
                                    best_position = (x, y)
                # Nếu tìm thấy thanh tốt nhất, cắt sản phẩm vào đó
                if best_fit_idx != -1:
                    return {
                        "stock_idx": best_fit_idx,
                        "size": size,
                        "position": best_position,
                    }

        # Nếu không có thanh nào phù hợp, trả về hành động mặc định
        return {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (0, 0),
        }
    def calculate_space_left(self,stock,prod_size):
        prod_w, prod_h = prod_size
        prod_area = prod_w*prod_h
        remain = np.sum(stock == -1)
        return remain
# --------------------------BLD----------------------------------------#           
    def get_action_BLD(self, observation, info):
        """
        Lấy hành động cắt dựa trên thuật toán Bottom Left Decreasing (BLD).
        """
        products = observation["products"]
        stocks = observation["stocks"]

        # Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_products = sorted(
            enumerate(products),
            key=lambda p: p[1]["size"][0] * p[1]["size"][1],
            reverse=True
        )

        # Thử đặt từng sản phẩm vào tấm vật liệu
        for prod_idx, product in sorted_products:
            if product["quantity"] > 0:
                size = product["size"]

                # Duyệt qua từng tấm vật liệu
                for stock_idx, stock in enumerate(stocks):
                    position = self.blhd_place(stock, size)
                    if position is not None:
                        # Nếu tìm được vị trí, giảm số lượng sản phẩm và trả về hành động
                        # product["quantity"] -= 1
                        return {
                            "stock_idx": stock_idx,
                            "size": size,
                            "position": position,
                        }

        # Nếu không có vị trí nào phù hợp, trả về hành động mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def blhd_place(self, stock, prod_size):
        """
        Tìm vị trí đặt sản phẩm theo chiến lược Bottom Left Decreasing (BLD).
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Nếu sản phẩm lớn hơn tấm vật liệu, trả về None
        if prod_w > stock_w or prod_h > stock_h:
            return None

        positions = []

        # Duyệt qua các vị trí khả thi trên tấm vật liệu (ưu tiên từ dưới lên trên)
        for y in reversed(range(stock_h - prod_h + 1)):  # Duyệt từ hàng thấp nhất
            for x in range(stock_w - prod_w + 1):  # Duyệt từ trái sang phải
                if self._can_place_(stock, (x, y), prod_size):
                    positions.append((x, y))

        # Nếu không có vị trí khả thi, trả về None
        if not positions:
            return None

        # Không cần sắp xếp lại vì vòng lặp đã duyệt theo thứ tự Bottom-Left
        return positions[0]

    


    

    


    



