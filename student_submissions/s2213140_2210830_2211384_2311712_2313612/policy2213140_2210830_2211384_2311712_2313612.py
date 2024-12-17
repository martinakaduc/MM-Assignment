from policy import Policy
import numpy as np

class Policy2213140_2210830_2211384_2311712_2313612(Policy):
    def __init__(self, policy_id=1):
        super().__init__()
        assert policy_id in [1, 2, 3], "Policy ID must be 1 or 2 or 3"

        # Student code here
        if policy_id == 1:
            # First-Fit Decreasing Heuristic
            pass
        elif policy_id == 2:
            # Best Fit Decrease Heuristic
            pass
        elif policy_id == 3:
            # Kanapsack algorithm
            self.required_quantities = None
            self.dimensions = None
            self.stock_idx = 0
            self.flag = 1

        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            # First-Fit Decreasing Heuristic
            return self.get_action_policy1(observation, info)
        elif self.policy_id == 2:
            # Best Fit Decrease Heuristic
            return self.get_action_policy2(observation, info)
        elif self.policy_id == 3:
            # Kanapsack algorithm
            return self.get_action_policy3(observation, info)

    # Student code here
    # You can add more functions if needed

    # First-Fit Decreasing Heuristic
    def get_action_policy1(self, observation, info):
        # Lấy danh sách sản phẩm
        list_prods = observation["products"]
    
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Sắp xếp sản phẩm theo diện tích (diện tích = width x height)
        sorted_prods = sorted(
            list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True
        )

        # Duyệt từng sản phẩm có quantity > 0
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
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
    
    # Best Fit Decrease Heuristic
    def get_action_policy2(self, observation, info):
        list_prods = observation["products"]

        # Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_prods = sorted(
            list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True
        )

        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                best_fit = None

                # Lặp qua các kho, tìm kho phù hợp nhất
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                wasted_space = (stock_w * stock_h) - (prod_w * prod_h)
                                if best_fit is None or wasted_space < best_fit[0]:
                                    best_fit = (wasted_space, i, x, y, prod_size)
                    
                    if stock_w >= prod_h and stock_h >= prod_w:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    wasted_space = (stock_w * stock_h) - (prod_w * prod_h)
                                    if best_fit is None or wasted_space < best_fit[0]:
                                        best_fit = (wasted_space, i, x, y, prod_size[::-1])

                if best_fit:
                    _, stock_idx, pos_x, pos_y, prod_size = best_fit
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
    # Kanapsack algorithm
    def get_action_policy3(self, observation, info):
        if self.flag == 1:
            #lay so lieu
            stocks = observation["stocks"]
            items = sorted(observation["products"], key=lambda prod: prod["size"][0])
            observation["products"] = items
            #xu ly so lieu
            self._pre_data_(items)
            #xu ly stock index
            self.stock = stocks[self.stock_idx]
            stock_length, stock_height = self._get_stock_size_(self.stock)
            #tao ma tran
            self.dimensions = [dim for qty, dim in zip(self.quantities, self.dimensions) if qty != 0]

            initial_patterns, lengths, widths = self._initial_2d_patterns(stock_length, stock_height)
            num_patterns = len(self.dimensions)
            cB = [1] * num_patterns
            B_inv = np.linalg.inv(initial_patterns)
            cBB_inv = np.dot(cB, B_inv)
            #tim cot toi ưu
            matrix_width_e, matrix_width_y = self._knapsack(stock_height, widths, cBB_inv)
            matrix_length_e, matrix_length_y = self._knapsack(stock_length, lengths, matrix_width_e)

            diagonal_width_y = np.diagonal(matrix_width_y)
            last_matrix = matrix_length_y[-1]
            new_column = np.array(diagonal_width_y * last_matrix)

            rows = max(new_column)  # Số hàng = phần tử lớn nhất trong mảng
            cols = len(new_column)  # Số cột = chiều dài mảng
            # Khởi tạo ma trận 2D toàn 0
            self.result_matrix = np.zeros((rows, cols), dtype=int)

            # Điền giá trị vào ma trận
            for col, val in enumerate(new_column):
                self.result_matrix[:val, col] = 1  # Gán 1 vào 'val' hàng đầu tiên của cột 'col'
            self.flag = 0

        result = self._cut_action()
        if self.stock_idx >= 100:
            self.stock_idx = 0
        return result

    def _pre_data_(self, items):
        self.quantities = np.array([prod["quantity"] for prod in items])  # Số lượng sản phẩm
        self.dimensions = np.array([prod["size"] for prod in items])  # Kích thước sản phẩm

    def _initial_2d_patterns(self, roll_length, roll_width):
        num_orders = len(self.dimensions)
        patterns = np.zeros((num_orders, num_orders), dtype=float)
        for i, (order_length, order_width) in enumerate(self.dimensions):
            max_length = roll_length // order_length
            max_width = roll_width // order_width
            patterns[i, i] = max_length * max_width

        order_lengths = [order[0] for order in self.dimensions]
        order_widths = [order[1] for order in self.dimensions]

        return patterns, order_lengths, order_widths

    def _knapsack(self, S, weights, values):
        n = len(weights)
        dp = [0.0] * (S + 1)
        matrix_e = [0.0] * n
        matrix_y = [[0] * n for _ in range(n)]
        y = [[0] * n for _ in range(S + 1)]

        for i in range(1, n + 1):
            for j in range(weights[i - 1], S + 1):
                if dp[j] < dp[j - weights[i - 1]] + values[i - 1]:
                    dp[j] = dp[j - weights[i - 1]] + values[i - 1]
                    y[j] = y[j - weights[i - 1]][:]
                    y[j][i - 1] += 1
            matrix_e[i - 1] = dp[S]
            matrix_y[i - 1] = y[S][:]

        return matrix_e, matrix_y

    def _cut_action(self):
        print("Current column to cut:", self.result_matrix)
        pos_x, pos_y = None, None
        stock_length, stock_height = self._get_stock_size_(self.stock)
        for row in range(self.result_matrix.shape[0]): # Duyệt qua số hàng
            for col in range(self.result_matrix.shape[1]):  # Duyệt qua số cột
                if self.result_matrix[row, col] > 0:
                    proc_length, proc_height = self.dimensions[col]

                    if stock_length >= proc_length and stock_height >= proc_height:
                        for y in range(stock_height - proc_height + 1):
                            for x in range(stock_length - proc_length + 1):
                                if self._can_place_(self.stock, (x, y), self.dimensions[col]):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                    if pos_x is None and pos_y is None:
                        if stock_length >= proc_height and stock_height >= proc_length:
                            for y in range(stock_height - proc_length + 1):
                                for x in range(stock_length - proc_height + 1):
                                    if self._can_place_(self.stock, (x, y), self.dimensions[col][::-1]):
                                        self.dimensions[col] = self.dimensions[col][::-1]
                                        pos_x, pos_y = x, y
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break

                    if pos_x is not None and pos_y is not None:
                        self.result_matrix[row, col] -= 1  # Trừ 1
                       # print(f"Cutting item {index} at position {(pos_x, pos_y)}")
                        return {
                            "stock_idx": self.stock_idx,
                            "size": self.dimensions[col],
                            "position": (pos_x, pos_y),
                        }
        self.stock_idx += 1
        self.flag = 1
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
