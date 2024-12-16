from queue import PriorityQueue
from policy import Policy
import numpy as np


class Policy2433042_2312763_2211199_2211299_2213444(Policy):
    def __init__(self, policy_id=1):
        """
        Khởi tạo policy.
        :param policy_id: 1 - BFD, 2 - B&B
        """
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        """
        Chọn hành động dựa trên policy_id.
        """
        if self.policy_id == 1:
            # Giai thuat BFD
            return self._get_action_bfd(observation, info)
        elif self.policy_id == 2:
            #Giai thuat B&B
            return self._get_action_bb(observation, info)

    # ==================== BFD Implementation ====================
    def _get_action_bfd(self, observation, info):
        """
        Triển khai giải thuật BFD.
        """
        # Lấy danh sách sản phẩm và sắp xếp theo diện tích giảm dần
        products = sorted(
            [p for p in observation["products"] if p["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],
            reverse=True,
        )

        for product in products:
            prod_size = product["size"]
            best_stock_idx = -1
            best_position = None
            min_waste = float("inf")

            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                for x in range(stock_w - prod_size[0] + 1):
                    for y in range(stock_h - prod_size[1] + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            waste = self._calculate_waste(stock, (x, y), prod_size)
                            if waste < min_waste:
                                best_stock_idx = stock_idx
                                best_position = (x, y)
                                min_waste = waste

            if best_stock_idx != -1 and best_position is not None:
                return {"stock_idx": best_stock_idx, "size": prod_size, "position": best_position}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    # ==================== B&B Implementation ====================
    def _get_action_bb(self, observation, info):
        """
        Triển khai giải thuật B&B với pruning mạnh.
        """
        products = [
            {"size": p["size"], "quantity": p["quantity"]}
            for p in observation["products"]
            if p["quantity"] > 0
        ]
        initial_stocks = observation["stocks"]

        queue = PriorityQueue()
        state_id = 0
        queue.put((0, 0, state_id, {"delta_stock": [], "products": products, "actions": []}))

        best_solution = None
        min_trim_loss = float("inf")
        MAX_QUEUE_SIZE = 1000
        MAX_DEPTH = 10

        while not queue.empty():
            if queue.qsize() > MAX_QUEUE_SIZE:
                queue.get()  # Loại bỏ trạng thái xấu nhất

            bound, depth, _, state = queue.get()

            if depth > MAX_DEPTH:
                continue

            stocks = self._reconstruct_stocks(initial_stocks, state["delta_stock"])

            # Kiểm tra điều kiện kết thúc
            if all(p["quantity"] == 0 for p in state["products"]):
                trim_loss = info["trim_loss"]
                if trim_loss < min_trim_loss:
                    min_trim_loss = trim_loss
                    best_solution = state["actions"]
                continue

            for product_idx, product in enumerate(state["products"]):
                if product["quantity"] == 0:
                    continue

                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    for x in range(stock_w - product["size"][0] + 1):
                        for y in range(stock_h - product["size"][1] + 1):
                            if self._can_place_(stock, (x, y), product["size"]):
                                new_state = self._branch_state(
                                    state, stock_idx, product_idx, (x, y), product["size"]
                                )
                                bound = self._calculate_bound(new_state, info)
                                state_id += 1
                                queue.put((bound, depth + 1, state_id, new_state))

        if best_solution:
            return best_solution[0]

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    # ==================== Helper Methods ====================
    def _reconstruct_stocks(self, initial_stocks, delta_stock):
        """
        Tái tạo trạng thái stocks từ delta_stock.
        """
        stocks = [np.copy(stock) for stock in initial_stocks]
        for delta in delta_stock:
            stock_idx, position, size, product_idx = delta
            x, y = position
            w, h = size
            stocks[stock_idx][x : x + w, y : y + h] = product_idx
        return stocks

    def _branch_state(self, state, stock_idx, product_idx, position, size):
        """
        Tạo nhánh mới từ trạng thái hiện tại, chỉ lưu delta.
        """
        new_delta_stock = state["delta_stock"] + [(stock_idx, position, size, product_idx)]
        new_products = [p.copy() for p in state["products"]]
        new_products[product_idx]["quantity"] -= 1

        new_actions = state["actions"] + [
            {"stock_idx": stock_idx, "size": size, "position": position}
        ]

        return {"delta_stock": new_delta_stock, "products": new_products, "actions": new_actions}

    def _calculate_waste(self, stock, position, prod_size):
        """
        Tính toán lãng phí khi đặt sản phẩm vào stock.
        """
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        used_cells = np.count_nonzero(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] != -2)
        return used_cells - (prod_w * prod_h)

    def _get_stock_size_(self, stock):
        """
        Lấy kích thước thực tế của stock.
        """
        width = np.sum(np.any(stock != -2, axis=1))
        height = np.sum(np.any(stock != -2, axis=0))
        return width, height

    def _can_place_(self, stock, position, size):
        """
        Kiểm tra xem sản phẩm có thể đặt vào vị trí không.
        """
        pos_x, pos_y = position
        w, h = size
        return np.all(stock[pos_x : pos_x + w, pos_y : pos_y + h] == -1)

    def _calculate_bound(self, state, info):
        """
        Tính toán giá trị bound.
        """
        remaining_products = sum(p["quantity"] for p in state["products"])
        remaining_space = info["trim_loss"]
        return remaining_products / max(remaining_space, 1)
