import numpy as np
from policy import Policy


class GreedyPolicy(Policy):
    def __init__(self):
        self.sorted_products = []  # Danh sách sản phẩm được sắp xếp giảm dần theo diện tích

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]
        # Sắp xếp sản phẩm nếu cần
        if not self.sorted_products:
            self.sorted_products = self._sort_products_(products)

        for prod_idx, _, prod_size in self.sorted_products:
            if products[prod_idx]["quantity"] <= 0:
                continue

            for stock_idx, stock in enumerate(stocks):
                position, rotated = self._find_first_fit_(stock, prod_size)
                if position is not None:
                    # Cập nhật sản phẩm đã chọn và trả về hành động
                    products[prod_idx]["quantity"] -= 1
                    width, height = prod_size[::-1] if rotated else prod_size

                    return {
                        "stock_idx": stock_idx,
                        "size": (width, height),
                        "position": position,
                    }

        # Không tìm thấy vị trí phù hợp
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _sort_products_(self, products):
        """Sắp xếp sản phẩm giảm dần theo diện tích."""
        product_list = []
        for idx, prod in enumerate(products):
            if prod["quantity"] > 0:
                area = prod["size"][0] * prod["size"][1]
                product_list.append((idx, area, prod["size"]))

        return sorted(product_list, key=lambda x: x[1], reverse=True)

    def _find_first_fit_(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Thử cả hai hướng đặt (ngang và xoay 90 độ)
        for width, height in [(prod_w, prod_h), (prod_h, prod_w)]:
            if width > stock_w or height > stock_h:
                continue

            for x in range(stock_w - width + 1):
                for y in range(stock_h - height + 1):
                    if self._can_place_(stock, (x, y), (width, height)):
                        return (x, y), (width != prod_w)  # Trả về vị trí và trạng thái xoay

        return None, False

    def _can_place_(self, stock, position, prod_size):
        x, y = position
        width, height = prod_size
        
        return np.all(stock[x:x + width, y:y + height] == -1)