from policy import Policy
import numpy as np
class Policy2311518_2312603_2310198_2312186_2312550(Policy):
    def __init__(self, policy_id=1):
        """
        Constructor for Policy2311518_2312603_2310198_2312186_2312550
        :param policy_id: ID of the policy (1: First Fit Decreasing, 2: Best Fit Decreasing)
        """
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id  # Lưu lại policy_id để xác định thuật toán

        # Khởi tạo các thuật toán tương ứng
        if policy_id == 1:
            self.policy = FFDPolicy()  # Thuật toán First Fit Decreasing
        elif policy_id == 2:
            self.policy = BFDPolicy()  # Thuật toán Best Fit Decreasing
        

    def get_action(self, observation, info):

        # Gọi phương thức `get_action` của thuật toán được chọn
        return self.policy.get_action(observation, info)
    # Student code here
    # You can add more functions if needed


# Create a policy using the Best Fit Decreasing (BFD) method
class BFDPolicy(Policy):
    def __init__(self): 
        pass

    def _get_empty_area_(self, stock):
        # Calculate the remaining empty area
        return np.sum(stock == -1)

    def _find_best_position_(self, stock, prod_size):
        # Find the best position (bottom-left corner)
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        best_pos = None
        min_y = float("inf")

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    if y < min_y:
                        min_y = y
                        best_pos = (x, y)
        return best_pos

    def get_action(self, observation, info):
        stocks = observation["stocks"]
        products = list(observation["products"])

        # Sort products in decreasing order of area
        products.sort(
            key=lambda x: x["size"][0] * x["size"][1] * (x["quantity"] > 0),
            reverse=True,
        )

        stock_idx = -1
        prod_size = [0, 0]
        pos_x, pos_y = 0, 0

        for prod in products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                best_empty_area = float("inf")
                best_position = None
                best_stock_idx = -1
                rotated = False  # Đánh dấu xem sản phẩm có xoay không

                # Duyệt qua từng kho để tìm kho phù hợp nhất
                for i, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Kiểm tra sản phẩm theo kích thước gốc
                    if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                        position = self._find_best_position_(stock, prod_size)
                        if position:
                            empty_area = self._get_empty_area_(stock)
                            if empty_area < best_empty_area:
                                best_empty_area = empty_area
                                best_position = position
                                best_stock_idx = i
                                rotated = False  # Không xoay

                    # Kiểm tra sản phẩm sau khi xoay
                    rotated_size = prod_size[::-1]  # Đổi chiều sản phẩm
                    if stock_w >= rotated_size[0] and stock_h >= rotated_size[1]:
                        position = self._find_best_position_(stock, rotated_size)
                        if position:
                            empty_area = self._get_empty_area_(stock)
                            if empty_area < best_empty_area:
                                best_empty_area = empty_area
                                best_position = position
                                best_stock_idx = i
                                rotated = True  # Đánh dấu xoay

                # Nếu tìm thấy vị trí phù hợp, cập nhật thông tin
                if best_position:
                    pos_x, pos_y = best_position
                    stock_idx = best_stock_idx
                    if rotated:
                        prod_size = prod_size[::-1]  # Áp dụng kích thước xoay
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


# Create a policy using the First Fit Decreasing (FFD) method
class FFDPolicy(Policy):
    def __init__(self):
        pass

    def sort_stocks_by_area(self, stocks):
        # Create a list of tuples (index, area)
        stock_areas = []
        for i, stock in enumerate(stocks):
            # Calculate the actual area of the stock (excluding cells with -2)
            stock_w = np.sum(np.any(stock != -2, axis=1))
            stock_h = np.sum(np.any(stock != -2, axis=0))
            area = stock_w * stock_h
            stock_areas.append((i, area))

        # Sort by area in decreasing order and get indices
        sorted_indices = [
            idx for idx, _ in sorted(stock_areas, key=lambda x: x[1], reverse=True)
        ]
        return sorted_indices

    def get_action(self, observation, info):
        stocks = observation["stocks"]
        products = list(observation["products"])

        # Sort products in decreasing order of area
        products.sort(
            key=lambda x: x["size"][0] * x["size"][1] * (x["quantity"] > 0),
            reverse=True,
        )
        # Sort stocks in decreasing order of area
        sorted_stock_indices = self.sort_stocks_by_area(stocks)

        stock_idx = -1
        prod_size = [0, 0]
        pos_x, pos_y = 0, 0

        for prod in products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                # Find the first suitable stock
                for i in sorted_stock_indices:
                    stock_w, stock_h = self._get_stock_size_(stocks[i])
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stocks[i], (x, y), prod_size):
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
                                if self._can_place_(stocks[i], (x, y), prod_size[::-1]):
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


