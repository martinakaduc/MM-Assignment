from policy import Policy
import numpy as np
class Policy2213332_2213996_2213851_2134006_2010545(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2 ,3], "Policy ID must be 1 or 2 or 3"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.near_optimal_cutting_action(observation)
        elif self.policy_id == 2:
            return self.two_dimensional_cutting_action(observation)
        elif self.policy_id == 3:
            return self.first_fit_decreasing_height_action(observation)

    def near_optimal_cutting_action(self, observation):
        list_prods = observation["products"]  
        stock_idx = -1  
        pos_x, pos_y = None, None  

        # Lọc các sản phẩm có số lượng > 0 và sắp xếp theo diện tích (từ lớn đến nhỏ)
        prod_sizes = sorted(
            [prod["size"] for prod in list_prods if prod["quantity"] > 0],
            key=lambda x: x[0] * x[1],  
            reverse=True,  
        )
        
        if not prod_sizes:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        def fit_item_in_stock(stock, item):
            stock_w, stock_h = self._get_stock_size_(stock)  
            positions = []  

            # Kiểm tra các vị trí không xoay sản phẩm
            for x in range(stock_w - item[0] + 1):
                for y in range(stock_h - item[1] + 1):
                    if self._can_place_(stock, (x, y), item):  
                        positions.append((x, y, item))  

            # Kiểm tra các vị trí với sản phẩm xoay 90 độ
            rotated_item = (item[1], item[0])  
            for x in range(stock_w - rotated_item[0] + 1):
                for y in range(stock_h - rotated_item[1] + 1):
                    if self._can_place_(stock, (x, y), rotated_item):  
                        positions.append((x, y, rotated_item))  

            # Chọn vị trí có diện tích dư thừa ít nhất
            best_position = None
            min_residual = float("inf")  
            for pos in positions:
                residual_space = (
                    (stock_w - (pos[0] + pos[2][0])) * stock_h +
                    (stock_h - (pos[1] + pos[2][1])) * stock_w
                )  # Tính diện tích dư thừa
                if residual_space < min_residual:  
                    min_residual = residual_space
                    best_position = pos

            return best_position  # Trả về vị trí tối ưu cho sản phẩm

        # Duyệt qua tất cả các cuộn và các sản phẩm để tìm vị trí cắt
        for i, stock in enumerate(observation["stocks"]):
            for item in prod_sizes:
                placement = fit_item_in_stock(stock, item)  
                if placement:
                    pos_x, pos_y, final_size = placement  
                    return {"stock_idx": i, "size": final_size, "position": (pos_x, pos_y)}  

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}  
        
    def two_dimensional_cutting_action(self, observation):
        list_prods = observation["products"]
        stock_idx, pos_x, pos_y = -1, None, None

        # Sắp xếp sản phẩm theo chiều cao giảm dần
        sorted_prods = sorted(list_prods, key=lambda x: x["size"][1], reverse=True)

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Tìm vị trí cắt cho sản phẩm không xoay
                    for x in range(stock_w - prod_size[0] + 1):
                        for y in range(stock_h - prod_size[1] + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {"stock_idx": i, "size": prod_size, "position": (x, y)}

                    # Kiểm tra sản phẩm xoay
                    if stock_w >= prod_size[1] and stock_h >= prod_size[0]:
                        for x in range(stock_w - prod_size[1] + 1):
                            for y in range(stock_h - prod_size[0] + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    return {"stock_idx": i, "size": prod_size[::-1], "position": (x, y)}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}  # Không tìm thấy vị trí hợp lệ
        
    def first_fit_decreasing_height_action(self, observation):
        """ Thuat toan First fit decreasing height (FFDH) """
        list_prods = list(observation["products"])
        list_prods.sort(key=lambda p: max(p["size"]), reverse=True)

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Tìm vị trí thấp nhất mà có thể đặt sản phẩm
                    for y in range(stock_h - prod_h + 1):
                        for x in range(stock_w - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}

                            # Thử xoay ngược sản phẩm
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                prod_size = prod_size[::-1]
                                return {"stock_idx": stock_idx, "size": prod_size, "position": (x, y)}

        # Nếu không thể đặt sản phẩm ở bất kỳ đâu, tạo vị trí ở level mới
        return {"stock_idx": None, "size": prod_size, "position": (0, 0)}
