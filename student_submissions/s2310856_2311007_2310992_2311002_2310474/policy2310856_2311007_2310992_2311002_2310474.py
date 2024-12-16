import numpy as np
from policy import Policy


class Policy2310856_2311007_2310992_2311002_2310474(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        if policy_id == 1:
            self.policy = Policy1()
        elif policy_id == 2:
            self.policy = Policy2()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)


class Policy1:
    def __init__(self):
        super().__init__()

    def get_action(self, observation, info):
        # Lấy danh sách sản phẩm và kho từ observation
        products = observation.get("products", [])
        stocks = observation.get("stocks", [])

        # Nếu không có sản phẩm hoặc kho, trả về hành động mặc định
        if not products or not stocks:
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

        # Sắp xếp sản phẩm theo diện tích giảm dần (greedy)
        products = sorted(products, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        # Duyệt qua tất cả các kho
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Duyệt qua tất cả các sản phẩm
            for product in products:
                if product["quantity"] > 0:
                    prod_w, prod_h = product["size"]

                    # Kiểm tra nếu sản phẩm có thể đặt vào kho mà không quay
                    if stock_w >= prod_w and stock_h >= prod_h:
                        positions = self._generate_positions(stock_w, stock_h, prod_w, prod_h)

                        # Kiểm tra các vị trí hợp lệ để đặt sản phẩm
                        for x, y in positions:
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                # Trả về hành động khi tìm thấy vị trí phù hợp
                                return {
                                    "stock_idx": stock_idx,
                                    "size": (prod_w, prod_h),
                                    "position": (x, y),
                                }

                    # Kiểm tra nếu sản phẩm có thể đặt vào kho sau khi quay
                    elif stock_w >= prod_h and stock_h >= prod_w:
                        # Quay sản phẩm
                        rotated_prod_w, rotated_prod_h = prod_h, prod_w
                        positions = self._generate_positions(stock_w, stock_h, rotated_prod_w, rotated_prod_h)

                        # Kiểm tra các vị trí hợp lệ để đặt sản phẩm quay
                        for x, y in positions:
                            if self._can_place_(stock, (x, y), (rotated_prod_w, rotated_prod_h)):
                                # Trả về hành động khi tìm thấy vị trí phù hợp sau khi quay
                                return {
                                    "stock_idx": stock_idx,
                                    "size": (rotated_prod_w, rotated_prod_h),
                                    "position": (x, y),
                                }

            # Nếu không tìm thấy vị trí hợp lệ cho sản phẩm trong kho này, tiếp tục với kho tiếp theo
            print(f"No valid placement in stock {stock_idx}, moving to next stock.")

        # Nếu không tìm được vị trí hợp lệ cho bất kỳ sản phẩm nào trong tất cả các kho
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    def _generate_positions(self, stock_w, stock_h, prod_w, prod_h):
        """
        Tạo tất cả các vị trí hợp lệ có thể đặt sản phẩm.
        """
        positions = [(x, y) for x in range(stock_w - prod_w + 1) for y in range(stock_h - prod_h + 1)]
        return positions

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)

class Policy2:  # Lấy product từ lớn đến nhỏ, trái qua phải, lấy theo thứ tự

    def __init__(self):
        pass
    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h
    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)
    def get_action(self, observation, info):
        list_prods = observation["products"]

        # Sắp xếp danh sách sản phẩm theo kích thước giảm dần (diện tích)
        sorted_prods = sorted(
            list_prods,
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )

        stock_idx = -1
        pos_x, pos_y = None, None

        # Duyệt qua từng kho theo thứ tự
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Duyệt qua từng sản phẩm (đã được sắp xếp từ lớn đến nhỏ)
            for prod in sorted_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    # Kiểm tra cả hai chiều của sản phẩm (cho phép xoay)
                    orientations = [(prod_w, prod_h), (prod_h, prod_w)]

                    for prod_w, prod_h in orientations:
                        # Kiểm tra nếu sản phẩm có thể đặt vào kho
                        if stock_w < prod_w or stock_h < prod_h:
                            continue

                        # Áp dụng thuật toán Bottom-Left để tìm vị trí phù hợp
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                    # Nếu đặt được sản phẩm, trả về hành động
                                    return {
                                        "stock_idx": i,
                                        "size": (prod_w, prod_h),
                                        "position": (x, y),
                                    }

        # Nếu không thể đặt được sản phẩm nào
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}