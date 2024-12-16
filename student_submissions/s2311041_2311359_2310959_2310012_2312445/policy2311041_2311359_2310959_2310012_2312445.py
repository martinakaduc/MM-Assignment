from policy import Policy
import numpy as np
from scipy.optimize import linprog
import json
#####################################################################################################
##################### DEFINE FUNCTION IN POLICY_ID=1 ################################################
#####################################################################################################


def PG(stock_length, product_widths, required_widths):
    # Lưu bản sao của product_widths ban đầu để sau khi xử lý sẽ đảo lại theo thứ tự ban đầu
    original_product_widths = product_widths.copy()

    # Đảo ngược danh sách product_widths trước khi xử lý
    product_widths = product_widths[::-1]
    num_products = len(product_widths)

    patterns = []
    seen = set()

    def backtrack(level, current_pattern, leftover):
        if level == num_products:
            # Tìm các chỉ số của các chiều rộng trong required_widths có trong product_widths
            required_indices = [product_widths.index(w) for w in required_widths if w in product_widths]

            # Kiểm tra điều kiện:
            if required_indices:
                condition = any(current_pattern[i] > 0 for i in required_indices)
            else:
                condition = True

            if (leftover == 0 or all(leftover < w for w in product_widths)) and condition:
                key = tuple(current_pattern)
                if key not in seen:
                    seen.add(key)
                    patterns.append(current_pattern[:])
            return

        width = product_widths[level]
        max_count = leftover // width

        for count in range(max_count + 1):
            current_pattern[level] = count
            new_leftover = leftover - count * width
            backtrack(level + 1, current_pattern, new_leftover)

        current_pattern[level] = 0

    current_pattern = [0] * num_products
    backtrack(0, current_pattern, stock_length)

    # Chuyển danh sách mẫu thành ma trận
    pattern_matrix = np.array(patterns).T

    # Đảo lại hàng trong ma trận để phản ánh thứ tự ban đầu của product_widths
    # product_widths hiện tại là đã đảo, vì vậy sau khi tính toán xong, ta sẽ đảo lại thứ tự cột
    # tương ứng với thứ tự ban đầu của original_product_widths.
    # Sử dụng np.argsort để tìm lại chỉ số theo thứ tự ban đầu
    sort_indices = np.argsort(original_product_widths)[::-1]
    pattern_matrix = pattern_matrix[sort_indices, :]

    return pattern_matrix


##
def Matrix(stock_size, product_size):
    # Sắp xếp tập hợp theo kích thước số 1
    product_sorted = sorted(product_size, key=lambda pair: (pair[0], pair[1]))
    first_dimension = [pair[0] for pair in product_sorted]
    unique_first_dimension = list(set(first_dimension))
    unique_first_dimension.sort()
    ##Tạo block 1
    block_1 = PG(stock_size[0], unique_first_dimension, [-1])
    new_matrix = np.zeros((len(product_sorted), block_1.shape[1]), dtype=int)
    ##Tạo block sau
    second_dimension = [pair[1] for pair in product_sorted]
    current_length = None
    all_widths_so_far = []
    widths_for_this_length = []
    row_index = 0
    for length, width in product_sorted:
        if length != current_length:
            if current_length is not None:
                pattern_matrix = PG(stock_size[1], all_widths_so_far, widths_for_this_length)
                if new_matrix.shape[0] != pattern_matrix.shape[0]:
                    max_rows = max(new_matrix.shape[0], pattern_matrix.shape[0])
                    # Thêm các hàng 0 để khớp kích thước
                    new_matrix = np.pad(new_matrix, ((0, max_rows - new_matrix.shape[0]), (0, 0)), mode='constant')
                    pattern_matrix = np.pad(pattern_matrix, ((0, max_rows - pattern_matrix.shape[0]), (0, 0)),
                                            mode='constant')
                new_matrix = np.hstack((new_matrix, pattern_matrix))
                new_block = np.zeros((block_1.shape[0], pattern_matrix.shape[1]))
                new_block[row_index, :] = -1
                block_1 = np.hstack((block_1, new_block))
                row_index += 1
            current_length = length
            widths_for_this_length = [width]
            all_widths_so_far += widths_for_this_length
        else:
            widths_for_this_length.append(width)
            all_widths_so_far += [width]
    if current_length is not None:
        pattern_matrix = PG(stock_size[1], all_widths_so_far, widths_for_this_length)
        new_matrix = np.hstack((new_matrix, pattern_matrix))

        # Tạo new_block và gộp vào block_1
        new_block = np.zeros((block_1.shape[0], pattern_matrix.shape[1]))
        new_block[row_index, :] = -1
        block_1 = np.hstack((block_1, new_block))
    result = np.vstack((block_1, new_matrix))
    # In ra danh sách đã sắp xếp (tuỳ chọn)
    ##print( result)

    return result  # Trả về danh sách đã sắp xếp

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.policy = abc();
        elif policy_id == 2:
            self.policy = xyz();
    def get_action(self, observation, info):
        # Student code here
        # Gọi phương thức `get_action` của thuật toán được chọn
        return self.policy.get_action(observation, info)


def load_data_from_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)
class abc(Policy):
    def __init__(self, policy_id=1):
        self.current_patterns = None
        self.current_solution = None
        self.pattern_index = 0
    def get_action(self, observation, info):        
        # Reset pattern index if we're starting with new products
        if self.current_patterns is None or self.pattern_index >= len(self.current_solution):
            # Extract product sizes and quantities from observation
            products = [(prod["size"][0], prod["size"][1]) for prod in observation["products"]
                        if prod["quantity"] > 0]
            quantities = np.array([prod["quantity"] for prod in observation["products"]
                                   if prod["quantity"] > 0])

            if not products:
                return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}

            # Get stock dimensions from first stock (assuming all stocks are same size)
            stock = observation["stocks"][0]
            stock_size = self._get_stock_size_(stock)

            # Generate cutting patterns using Matrix function
            self.current_patterns = Matrix(stock_size, products)

            # Set up and solve the cutting stock optimization problem
            num_patterns = self.current_patterns.shape[1]

            # Objective function: minimize number of stock sheets used
            c = np.ones(num_patterns)

            # Ensure quantities is a 1D array
            b_eq = quantities.flatten()

            # Bounds: non-negative integers
            bounds = [(0, None)] * num_patterns

            try:
                # Simple linear programming solution as initial solution
                result = linprog(c, A_eq=self.current_patterns, b_eq=b_eq,
                                 bounds=bounds, method='highs')

                if result.success:
                    # Round up the solution to integers
                    self.current_solution = np.ceil(result.x)
                else:
                    # Fallback to basic solution
                    self.current_solution = np.ones(num_patterns)

            except ValueError:
                # Fallback if optimization fails
                self.current_solution = np.ones(num_patterns)

            self.pattern_index = 0

        # Find next non-zero pattern
        while self.pattern_index < len(self.current_solution):
            if self.current_solution[self.pattern_index] > 0:
                pattern = self.current_patterns[:, self.pattern_index]

                # Find first non-zero element in pattern
                prod_idx = 0
                for i, count in enumerate(pattern):
                    if count > 0:
                        prod_idx = i
                        break

                # Get corresponding product
                active_prods = [p for p in observation["products"] if p["quantity"] > 0]
                if prod_idx < len(active_prods):
                    prod = active_prods[prod_idx]
                    prod_size = prod["size"]

                    # Try each stock
                    for stock_idx, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)

                        # Try normal orientation
                        if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                            for x in range(stock_w - prod_size[0] + 1):
                                for y in range(stock_h - prod_size[1] + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        self.pattern_index += 1
                                        return {
                                            "stock_idx": stock_idx,
                                            "size": prod_size,
                                            "position": (x, y)
                                        }

                        # Try rotated orientation
                        if stock_w >= prod_size[1] and stock_h >= prod_size[0]:
                            for x in range(stock_w - prod_size[1] + 1):
                                for y in range(stock_h - prod_size[0] + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        self.pattern_index += 1
                                        return {
                                            "stock_idx": stock_idx,
                                            "size": prod_size[::-1],
                                            "position": (x, y)
                                        }

            self.pattern_index += 1

        # Reset if no valid placement found
        self.current_patterns = None
        self.current_solution = None
        self.pattern_index = 0
        return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}
    

class xyz(Policy):
    def __init__(self,policy=2):
        self.current_stock_idx = 0
        self.current_level = 0
        self.max_height = 0
    def get_action(self, observation, info):
        stocks = observation["stocks"]
        sorted_products = self._sort_products(observation["products"])

        for stock_idx in range(self.current_stock_idx, len(stocks)):
            stock = stocks[stock_idx]
            stock_h, stock_w = self._get_stock_size_(stock)

            for product in sorted_products:
                if product["quantity"] <= 0:
                    continue

                prod_size = product["size"]
                # Tìm góc tốt nhất để đặt sản phẩm
                best_pos = self._find_best_corner(stock, prod_size, stock_h, stock_w)
                if best_pos:
                    return {
                        "stock_idx": stock_idx,
                        "size": prod_size,
                        "position": best_pos
                    }

                # Thử xoay sản phẩm nếu không tìm được vị trí
                rotated_size = prod_size[::-1]
                best_pos = self._find_best_corner(stock, rotated_size, stock_h, stock_w)
                if best_pos:
                    return {
                        "stock_idx": stock_idx,
                        "size": rotated_size,
                        "position": best_pos
                    }

        return {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (0, 0)
        }

    def _sort_products(self, products):
        """Sắp xếp sản phẩm theo diện tích giảm dần"""
        return sorted(
            products,
            key=lambda p: (
                p["size"][0] * p["size"][1] if p["quantity"] > 0 else -1,
                -(p["size"][0] + p["size"][1])  # Ưu tiên hình dạng vuông hơn
            ),
            reverse=True
        )

    def _find_best_corner(self, stock, prod_size, stock_h, stock_w):
        """Tìm góc tốt nhất để đặt sản phẩm dựa trên các heuristic"""
        best_score = float('-inf')
        best_pos = None
        prod_h, prod_w = prod_size

        for x in range(stock_h - prod_h + 1):
            for y in range(stock_w - prod_w + 1):
                if not self._can_place_(stock, (x, y), prod_size):
                    continue

                # Tính điểm cho vị trí này
                score = self._calculate_position_score(
                    stock, (x, y), prod_size, stock_h, stock_w
                )

                if score > best_score:
                    best_score = score
                    best_pos = (x, y)

        return best_pos

    def _calculate_position_score(self, stock, position, prod_size, stock_h, stock_w):
        """Tính điểm cho một vị trí dựa trên các tiêu chí:
        1. Khoảng cách tới các cạnh
        2. Tiếp xúc với các sản phẩm khác hoặc biên
        3. Không gian thừa xung quanh
        """
        x, y = position
        h, w = prod_size
        score = 0

        # 1. Ưu tiên vị trí gần các cạnh
        edge_distance = min(x, y, stock_h - (x + h), stock_w - (y + w))
        score -= edge_distance * 2

        # 2. Tính điểm tiếp xúc
        # Kiểm tra tiếp xúc bên trái
        if y == 0 or np.any(stock[x:x + h, y - 1] != -1):
            score += h

        # Kiểm tra tiếp xúc bên phải
        if y + w == stock_w or np.any(stock[x:x + h, y + w] != -1):
            score += h

        # Kiểm tra tiếp xúc phía trên
        if x == 0 or np.any(stock[x - 1, y:y + w] != -1):
            score += w

        # Kiểm tra tiếp xúc phía dưới
        if x + h == stock_h or np.any(stock[x + h, y:y + w] != -1):
            score += w

        # 3. Trừ điểm cho không gian thừa
        if y + w < stock_w:
            right_space = np.sum(stock[x:x + h, y + w] == -1)
            score -= right_space

        if x + h < stock_h:
            bottom_space = np.sum(stock[x + h, y:y + w] == -1)
            score -= bottom_space

        return score