import numpy as np
from policy import Policy
from scipy.optimize import linprog  

class ColumnGenerationPolicy(Policy):
    def __init__(self):
        super().__init__()
        self.patterns = []
        self.init = False
        self.num_prods = None

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        demand = np.array([prod["quantity"] for prod in products])
        sizes = [prod["size"] for prod in products]
        sorted_indices = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
        sizes = [sizes[i] for i in sorted_indices]
        demand = demand[sorted_indices]

        num_prods = len(products)

        if not self.init or self.num_prods != num_prods:
            self.init_patterns(num_prods, sizes, stocks)
            self.init = True
            self.num_prods = num_prods

        while True:
            c = np.ones(len(self.patterns))
            A = np.array(self.patterns).T
            b = demand
            result = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, None), method='highs')
            if result.status != 0:
                break
            dual_prices = result.ineqlin.marginals if hasattr(result.ineqlin, 'marginals') else None
            if dual_prices is None:
                break
            new_pattern = self.solve_pricing_problem(dual_prices, sizes, stocks)
            if new_pattern is None or any(np.array_equal(new_pattern, p) for p in self.patterns):
                break
            self.patterns.append(new_pattern)
        best_pattern = self.select_best_pattern(self.patterns, demand)
        action = self.pattern_to_action(best_pattern, sizes, stocks)
        return action

    def init_patterns(self, num_prods, sizes, stocks):
        self.patterns = []
        for j in range(len(stocks)):
            stock_size = self._get_stock_size_(stocks[j])
            for i in range(num_prods):
                if stock_size[0] >= sizes[i][0] and stock_size[1] >= sizes[i][1]:
                    pattern = np.zeros(num_prods, dtype=int)
                    pattern[i] = 1
                    self.patterns.append(pattern)
        self.patterns = list({tuple(p): p for p in self.patterns}.values())

    # Hàm giải bài toán phụ (subproblem) bằng tối ưu tuyến tính
    def solve_pricing_problem(self, dual_prices, sizes, stocks):
        best_pattern = None
        best_reduced_cost = -1
        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)

            if stock_w <= 0 or stock_h <= 0:
                continue
            n = len(sizes)
            # Cấu hình bài toán tối ưu tuyến tính
            c = -np.array(dual_prices)  # Mục tiêu: tối đa hóa giá trị giảm
            A_ub = []  # Ràng buộc về kích thước vật liệu
            b_ub = []

            # Ràng buộc về số lượng vật liệu (không vượt quá kích thước stock)
            for i in range(n):
                w, h = sizes[i]
                # Ràng buộc về không gian vật liệu cho mỗi sản phẩm
                A_ub.append([1 if j == i else 0 for j in range(n)])
                b_ub.append(stock_w // w * stock_h // h)  # Đảm bảo không vượt quá diện tích của stock

            # Ràng buộc: Sản phẩm phải nằm trong kích thước của vật liệu
            bounds = [(0, None)] * n

            # Giải bài toán tối ưu tuyến tính
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if result.success:
                # Tính toán chi phí giảm
                reduced_cost = np.dot(result.x, dual_prices) - 1
                if reduced_cost > best_reduced_cost:
                    best_reduced_cost = reduced_cost
                    best_pattern = result.x

        return best_pattern if best_reduced_cost > 1e-6 else None

    def select_best_pattern(self, patterns, demand):
        best_pattern = None
        best_coverage = -1
        best_cost = float('inf')

        for pattern in patterns:
            coverage = np.sum(np.minimum(pattern, demand))  # Tổng số sản phẩm mẫu có thể cắt
            cost = np.sum(pattern)  # Chi phí (số lượng mẫu)
            if coverage > best_coverage or (coverage == best_coverage and cost < best_cost):
                best_coverage = coverage
                best_cost = cost
                best_pattern = pattern

        return best_pattern

    def pattern_to_action(self, pattern, sizes, stocks):
        for i, count in enumerate(pattern):
            if count > 0:
                prod_size = sizes[i]
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        position = self.bottom_left_place(stock, prod_size)
                        if position is not None:
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": position
                            }
        return {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (0, 0)
        }

    def bottom_left_place(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        if stock_w < prod_w or stock_h < prod_h:
            return None
        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None