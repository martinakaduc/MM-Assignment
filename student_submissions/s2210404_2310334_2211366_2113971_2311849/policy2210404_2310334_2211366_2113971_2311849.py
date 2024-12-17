from policy import Policy
import random
import numpy as np


class Policy2210404_2310334_2211366_2113971_2311849(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            self.used_areas = {}


    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            list_prods = observation["products"]

        # Sắp xếp danh sách sản phẩm theo kích thước giảm dần (diện tích)
        #BLD algorithm
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

                    # Kiểm tra nếu sản phẩm có thể đặt vào kho
                        if stock_w < prod_w or stock_h < prod_h:
                            continue

                    # Áp dụng thuật toán Bottom-Left để tìm vị trí phù hợp
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                # Nếu đặt được sản phẩm, trả về hành động
                                # print({
                                #     "stock_idx": i,
                                #     "size": prod_size,
                                #     "position": (x, y),
                                # })
                                    return {
                                    "stock_idx": i,
                                    "size": prod_size,
                                    "position": (x, y),
                                    }         
            #print({"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)})
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        elif self.policy_id == 2:
            #NFDH algorithm
            list_prods = observation["products"]
            sorted_prods = sorted(list_prods, key=lambda prod: prod["size"][1], reverse=True)  # Sắp xếp theo chiều cao giảm dần

            stocks = observation["stocks"]
            action = {"size": None, "position": None, "stock_idx": None}

            for prod in sorted_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    placed = False

                    for stock_idx, stock in enumerate(stocks):
                        stock_w, stock_h = self._get_stock_size_(stock)

                        # 1. Lấy danh sách khoảng trống còn lại trong kho
                        free_spaces = self._find_free_spaces(stock)

                        # 2. Thử đặt sản phẩm vào các khoảng trống
                        for space in free_spaces:
                            space_x, space_y, space_w, space_h = space

                            if space_w >= prod_w and space_h >= prod_h:
                                if self._can_place_(stock, (space_x, space_y), prod_size):
                                    action["size"] = prod_size
                                    action["position"] = (space_x, space_y)
                                    action["stock_idx"] = stock_idx
                                    placed = True
                                    break
                        if placed:
                            break

                    # 3. Nếu không đặt được vào khoảng trống, tìm vị trí mới từ góc trái dưới cùng
                        row_tracker = np.zeros(stock_w, dtype=int)
                        for x in range(stock_w - prod_w + 1):  # Duyệt từ trái sang phải
                            y = row_tracker[x]  # Lấy dòng hiện tại theo tracker
                            if y + prod_h <= stock_h and self._can_place_(stock, (x, y), prod_size):
                                action["size"] = prod_size
                                action["position"] = (x, y)
                                action["stock_idx"] = stock_idx
                                placed = True

                                # Cập nhật row tracker cho vùng được đặt
                                for i in range(x, x + prod_w):
                                    row_tracker[i] = y + prod_h
                                break
                        if placed:
                            break

                    if placed:
                        break

            if not placed:
                action["size"] = None
                action["position"] = None
                action["stock_idx"] = None

            return action
    def _find_free_spaces(self, stock):
        """Tìm các khoảng trống trong kho"""
        free_spaces = []
        stock_w, stock_h = stock.shape

        visited = np.zeros_like(stock, dtype=bool)

        for x in range(stock_w):
            for y in range(stock_h):
                if stock[x, y] == -1 and not visited[x, y]:  # Nếu ô trống và chưa được duyệt
                    # Bắt đầu quét một vùng trống
                    width = 0
                    while x + width < stock_w and stock[x + width, y] == -1 and not visited[x + width, y]:
                        width += 1

                    height = 0
                    while y + height < stock_h and all(stock[x:x + width, y + height] == -1) and \
                            not any(visited[x:x + width, y + height]):
                        height += 1

                    # Đánh dấu vùng trống đã được duyệt
                    visited[x:x + width, y:y + height] = True

                    # Lưu vùng trống
                    free_spaces.append((x, y, width, height))

        return free_spaces
