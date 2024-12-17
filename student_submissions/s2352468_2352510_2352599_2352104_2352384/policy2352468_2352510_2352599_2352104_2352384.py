from policy import Policy
from scipy.optimize import linprog
import numpy as np

class Policy2352468_2352510_2352599_2352104_2352384(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        if policy_id == 1:
            self.policy = ColumnGeneration() # Column Generation
        elif policy_id == 2:
            self.policy = BranchAndBound() # Branch & Bound

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.policy.get_action(observation)
        elif self.policy_id == 2:
            return self.policy.get_action(observation)

######################## COLUMN GENERATION ########################
class ColumnGeneration:
    def __init__(self):
        self.patterns = []  # Lưu các mẫu cắt
        self.dual_prices = None  # Lưu dual prices
        self.remaining_products = []  # Danh sách sản phẩm cần cắt

    def _get_stock_size_(self, stock):   
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        if pos_x + prod_w <= stock.shape[0] and pos_y + prod_h <= stock.shape[1]:
            return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)
        return False

    def generateInitialPatterns(self, observation):
        num_products = len(observation["products"])
        self.patterns = []
        self.remaining_products = observation["products"]
        for i, prod in enumerate(observation["products"]):
            if prod["quantity"] > 0:
                pattern = np.zeros(num_products)
                pattern[i] = 1
                self.patterns.append(pattern)

    def solveRMP(self):
        num_products = len(self.remaining_products)
        num_patterns = len(self.patterns)

        # Matrix A [row = product][col = cut]
        A = np.zeros((num_products, num_patterns))

        # pattern into A
        for j, pattern in enumerate(self.patterns):
            A[:, j] = pattern[:num_products]

        b = np.array([prod["quantity"] for prod in self.remaining_products])
        c = np.ones(num_patterns)  # Minimize

        # Giải ILP
        res = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if res.success:
            self.dual_prices = res.x
        else:
            self.dual_prices = np.zeros(num_patterns)

    def generateNewPattern(self, observation):
        num_products = len(observation["products"])
        new_pattern = np.zeros(num_products)
        stock = observation["stocks"][0]  # Use first material
        stock_w, stock_h = self._get_stock_size_(stock)

        for i, prod in enumerate(observation["products"]):
            prod_w, prod_h = prod["size"]
            if prod["quantity"] > 0:
                for x in range(stock_w):
                    for y in range(stock_h):
                        if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                            new_pattern[i] += 1
                            break
                        if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                            new_pattern[i] += 1
                            break
        return new_pattern

    def get_action(self, observation):
        if not self.patterns:
            self.generateInitialPatterns(observation)

        self.solveRMP()
        new_pattern = self.generateNewPattern(observation)

        if new_pattern.sum() > 0:
            self.patterns.append(new_pattern)

        for i, prod in enumerate(observation["products"]):
            if prod["quantity"] > 0:
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod["size"]

                    for x in range(stock_w):
                        for y in range(stock_h):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": (prod_w, prod_h),
                                    "position": (x, y),
                                }
                            if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": (prod_h, prod_w),
                                    "position": (x, y),
                                }
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

######################## BRANCH AND BOUND ########################
class BranchAndBound:
    def __init__(self):
        self.best_solution = None  # Lưu nghiệm tốt nhất
        self.best_value = float('inf')  # Giá trị mục tiêu nhỏ nhất (minimize)
        self.patterns = []  # Danh sách mẫu cắt hiện tại
        self.remaining_products = []

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        if pos_x + prod_w <= stock.shape[0] and pos_y + prod_h <= stock.shape[1]:
            return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)
        return False

    def generateInitialPatterns(self, observation):
        num_products = len(observation["products"])
        self.patterns = []
        self.remaining_products = observation["products"]

        for i, prod in enumerate(observation["products"]):
            if prod["quantity"] > 0:
                pattern = np.zeros(num_products)
                pattern[i] = 1
                self.patterns.append(pattern)

    def solveRelaxedLP(self, A, b, c):
        res = linprog(c, A_eq = A, b_eq = b, bounds = (0, None), method = "highs")
        
        if res.success: # Nếu tìm ra nghiệm cho bài toán ILP
            return res.fun, res.x  # Giá trị mục tiêu và nghiệm thư giãn
        return float('inf'), None

    def branchAndBound(self, A, b, c):
        queue = [(0, [], c)]  # List

        while queue:
            queue.sort(key=lambda x: x[0]) # Sắp xếp hàng đợi theo giá trị mục tiêu (độ ưu tiên cao nhất)
            _, fixed_indices, c_mod = queue.pop(0)  # Lấy phần tử có độ ưu tiên cao nhất

            value, solution = self.solveRelaxedLP(A, b, c_mod) # Giải LP

            if value >= self.best_value: # Cắt nhánh
                continue

            # Kiểm tra nghiệm nguyên
            if all(int(x) == x for x in solution) == True:
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = solution
                continue

            # Chia nhánh (Branching)
            for i in range(len(solution)):
                if i not in fixed_indices and solution[i] != int(solution[i]):
                    # Nhánh trái: xi <= floor(solution[i])
                    new_fixed_indices_left = fixed_indices[:] + [i]
                    c_left = c_mod[:]  # Sao chép danh sách
                    c_left[i] = float('inf')
                    queue.append((value, new_fixed_indices_left, c_left))

                    # Nhánh phải: xi >= ceil(solution[i])
                    new_fixed_indices_right = fixed_indices[:] + [i]
                    c_right = c_mod[:]  # Sao chép danh sách
                    c_right[i] = float('-inf')
                    queue.append((value, new_fixed_indices_right, c_right))
                    break

    def get_action(self, observation):
        self.generateInitialPatterns(observation)
        num_products = len(self.remaining_products)
        num_patterns = len(self.patterns)

        A = np.zeros((num_products, num_patterns))
        for j, pattern in enumerate(self.patterns):
            A[:, j] = pattern[:num_products]

        b = np.array([prod["quantity"] for prod in self.remaining_products])
        c = np.ones(num_patterns)  # Minimize mục tiêu

        # Gọi Branch and Bound
        self.branchAndBound(A, b, c)

        if self.best_solution is not None:
            for _, prod in enumerate(observation["products"]):
                if prod["quantity"] > 0:
                    for stock_idx, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod["size"]

                        for x in range(stock_w):
                            for y in range(stock_h):
                                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": (prod_w, prod_h),
                                        "position": (x, y),
                                    }
                                if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": (prod_h, prod_w),
                                        "position": (x, y),
                                    }
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}