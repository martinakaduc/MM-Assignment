from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2320001_2312119_2312065_2312696_2312861(Policy):
    #####################khai bao gen_col################################
    columns = []
    size = []
    W = 0
    H = 0
    last_stock_idx = 0
    pattern = []
    rest = []
    orders = []

    #################khai bao first_fit##############################
    policy = 1
    last_stock_index = -1
    last_product = None


    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = 1
        elif policy_id == 2:
            self.policy = 2


    def get_action(self, observation, info):
        # Student code here
        if self.policy == 1:
            return self.first_fit(observation)
        if self.policy == 2:
            return self.gen_col(observation)

    # Student code here
    def first_fit(self, observation):
        stock_idx = -1
        product_x, product_y = 0, 0
        flag_cut = 0
        flag_check = [1]
        product_size = [0, 0]
        last_product_size = [0, 0]

        ###############################sort################################################################
        list_of_stocks = list(enumerate(observation["stocks"]))
        list_of_stocks.sort(key=lambda my_list: np.sum(my_list[1] != -2), reverse=True)

        list_of_prods = observation["products"]
        list_of_prods = sorted(list_of_prods, key=lambda my_prod: my_prod["size"][0] * my_prod["size"][1], reverse=True)
        ###################################################################################################


        if self.last_product is not None:
            last_product_size = self.last_product["size"]

        for prod in list_of_prods:
            if prod["quantity"] > 0:
                product_size = prod["size"]

                for i, stock in list_of_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = product_size

                    ##################################check stock#######################################
                    if self.check_conditions(flag_check, last_product_size, stock, prod_w, prod_h, i):
                        continue
                    #######################################################################################


                    ###################################chieu ban dau##########################################
                    if stock_w >= prod_w and stock_h >= prod_h and flag_cut == 0:
                        product_x, product_y = self.original_orientation(stock, product_size, prod_w, prod_h, stock_w, stock_h)
                        if product_x is not None and product_y is not None:
                            flag_cut = 1
                    ####################################################################################


                    #######################################xoay 90#######################################################
                    if stock_w >= prod_h and stock_h >= prod_w and flag_cut == 0:
                        product_x, product_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), product_size[::-1]):
                                    product_size = product_size[::-1]
                                    product_x, product_y = x, y
                                    flag_cut = 1
                                    break
                            if flag_cut == 1:
                                break
                    ##########################################################################################
                    # if stock_w >= prod_h and stock_h >= prod_w and flag_cut == 0:
                    #     product_x, product_y = self.rotated_orientation(stock, prod_size, prod_h, prod_w, stock_w, stock_h)
                    #     if product_x is not None and product_y is not None:
                    #         prod_size = prod_size[::-1]
                    #         flag_cut = 1

                    if flag_cut == 1:
                        stock_idx = i
                        self.last_product = prod
                        self.last_stock_index = stock_idx
                        break

                if flag_cut == 1:
                    break

        return {"stock_idx": stock_idx, "size": product_size, "position": (product_x, product_y)}

    def check_conditions(self, flag_check_ref, last_product_size, stock, prod_w, prod_h, i):
        if (self.last_stock_index == i and flag_check_ref[0] == 1) or self.last_stock_index == -1:
            flag_check_ref[0] = 0

        if last_product_size[0] == prod_w and last_product_size[1] == prod_h and flag_check_ref[0] == 1:
            return True

        current_area_left = np.sum(stock == -1)
        if current_area_left < prod_w * prod_h:
            return True
        return False

    def original_orientation(self, stock, product_size, prod_w, prod_h, stock_w, stock_h):
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), product_size):
                    return x, y
        return None, None

    # def rotated_orientation(self, stock, prod_size, prod_w, prod_h, stock_w, stock_h):
    #     for x in range(stock_w - prod_h + 1):
    #         for y in range(stock_h - prod_w + 1):
    #             if self._can_place_(stock, (x, y), prod_size[::-1]):
    #                 return x, y
    #     return None, None



    def gen_col(self, observation):

        if self.pattern != []:
            action = self.pattern.pop(0)
            # print("Num elements left:", len(self.pattern))
            print("Số lượng sản phẩm còn lại:", sum(prod["quantity"] for prod in observation["products"]))
            if sum(prod["quantity"] for prod in observation["products"]) <= 1:
                self.last_stock_idx = 0
            return action
        self.orders = [prod["quantity"] for prod in observation["products"]]
        self.size = [prod["size"] for prod in observation["products"]]
        stock_idx = self.last_stock_idx
        stock = observation["stocks"][stock_idx]
        self.W, self.H = self._get_stock_size_(stock)

        self.lengths = [size[0] for size in self.size]
        self.widths = [size[1] for size in self.size]
        self.rest = np.array(self.orders, dtype=float).flatten()
        self.initialize_master_problem()

        solution, dual_values = self.solve_master_problem()

        new_pattern, used_area = self.solve_knapsack(solution, observation)

        self.pattern = new_pattern
        self.generate_action(observation, used_area)
        self.last_stock_idx += 1
        action = self.pattern.pop(0)
        if sum(prod["quantity"] for prod in observation["products"]) <= 1:
            self.last_stock_idx = 0
        # print("Số lượng sản phẩm còn lại:", sum(prod["quantity"] for prod in observation["products"]))

        return action

    def initialize_master_problem(self):
        """
        - Với mỗi loại vật liệu sẽ là 1 biến trong bài toán chính.
        - Các mẫu này sẽ được sử dụng trong hệ phương trình ở bài toán chính
        """
        self.columns = []
        for i in range(len(self.orders)):
            pattern = [0] * len(self.orders)
            pattern[i] = 1
            self.columns.append(pattern)

    def solve_master_problem(self):
        """
        Giải bài toán chính bằng cách lập hệ phương trình tuyến tính
        - Xây dựng hàm mục tiêu:
        + Sử dụng diện tích của từng sản phẩm để thiết lập hệ số tối ưu.
        - Ràng buộc:
        + Đảm bảo số lượng các sản phẩm sẽ không quá số lượng diện tích còn lại, và tổng diện tích của sản phẩm không vượt quá
            diện tích kho hiện tại.
        - Sử dụng phương pháp simplex để giải quyết bài toán tuyến tính.
        - Kết quả:
        + Trả về lời giải tối ưu, số lượng sản phẩm có thể trong stock.
        """
        max_length = max(len(col) for col in self.columns)
        for col in self.columns:
            while len(col) < max_length:
                col.append(0)

        areas = [self.widths[i] * self.lengths[i] for i in range(len(self.size))]

        c = - np.array(areas)
        A_eq = np.array(self.columns, dtype=float).T
        b_eq = np.array(self.orders, dtype=float).flatten()
        bounds = [(0, None)] * len(self.columns)
        if b_eq.ndim != 1:
            b_eq = b_eq.flatten()
        total_area_constraint = np.array([self.widths[i] * self.lengths[i] for i in range(len(self.size))])
        A_ub = np.vstack([A_eq, total_area_constraint])
        b_ub = np.append(b_eq, self.W * self.H)

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')

        solution = np.floor(result.x)
        dual_values = result.slack
        print("Solution: ", solution)
        return solution, dual_values

    def solve_knapsack(self, solution, observation):
        """
        Giải bài toán knapsack để sắp xếp sản phẩm vào kho:
        - Mục tiêu:
        + Dựa trên số lượng sản phẩm từ kết quả của bài toán chính (`solution`), tìm cách sắp xếp sản phẩm vào kho sao cho không gian được sử dụng hiệu quả nhất.
        - Quy trình:
        1. Lấy thông tin sản phẩm (kích thước, số lượng).
        2. Khởi tạo mảng đánh dấu (`used_area`) để theo dõi vùng đã được sử dụng trong stock.
        3. Lặp qua từng sản phẩm:
            + Kiểm tra xem sản phẩm có thể đặt ở vị trí nào trong stock.
            + Nếu đặt được, cập nhật vị trí sản phẩm vào mẫu (`patterns`) và đánh dấu vùng đã sử dụng.
        4. Xử lý các trường hợp cần xoay sản phẩm (nếu phù hợp).
        - Kết quả:
        + Trả về danh sách mẫu sắp xếp (`patterns`) và trạng thái sử dụng vùng của tấm vật liệu(`used_area`).
        """
        items = observation["products"]
        weights = [item["size"][0] for item in items]
        heights = [item["size"][1] for item in items]
        quantities = [int(solution[i]) for i in range(len(items))]
        n = len(items)
        rest = self.rest
        stock = observation["stocks"][self.last_stock_idx]
        stock_w, stock_h = self.W, self.H
        patterns = []
        used_area = np.zeros((stock_w, stock_h), dtype=bool)

        for idx in range(n):
            quantity = quantities[idx]
            for _ in range(quantity):
                placed = False
                for x in range(stock_w + 1):
                    for y in range(stock_h + 1):
                        if rest[idx] >= 1 and self._can_place_new_(used_area, (x, y), (weights[idx], heights[idx])):
                            patterns.append({
                                "stock_idx": self.last_stock_idx,
                                "size": items[idx]["size"],
                                "position": (x, y)
                            })
                            rest[idx] -= 1
                            self._update_used_area_(used_area, (x, y), (weights[idx], heights[idx]))
                            placed = True
                            break
                        elif rest[idx] != 0 and self._can_place_new_(used_area, (x, y), (heights[idx], weights[idx])):
                            patterns.append({
                                "stock_idx": self.last_stock_idx,
                                "size": (heights[idx], weights[idx]),
                                "position": (x, y)
                            })
                            self._update_used_area_(used_area, (x, y), (heights[idx], weights[idx]))
                            rest[idx] -= 1
                            placed = True
                            break
                    if placed:
                        break
        self.rest = rest
        return patterns, used_area

    def _can_place_new_(self, used_area, position, prod_size):
        """
        Kiểm tra xem có thể đặt vật liệu không
        """
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        rows, cols = used_area.shape
        if (pos_x + prod_w >= rows or pos_y + prod_h >= cols):
            return False
        return np.all(used_area[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == 0)

    def _update_used_area_(self, used_area, position, prod_size):
        """
        Cập nhật diện tích sử dụng của mảnh stoke
        """
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        used_area[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] = 1

    def is_improving_pattern(self, pattern, dual_values):
        reduced_cost = sum(dual_values[i] * pattern[i] for i in range(len(pattern))) - 1
        return reduced_cost > 0

    def generate_action(self, observation, used_area):
        """
        Duyệt lại tấm stoke 1 lần nữa để đặt thêm các tấm vật liệu nếu có thể.
        1. Sắp xếp các sản phẩm còn lại theo diện tích từ lớn đến nhỏ.
        2. Lặp qua từng sản phẩm trong danh sách. Nếu đặt được sản phẩm, ta sẽ thêm hành động vào danh sách pattern. Sau đó cập nhật vùng diện tích sử dụng lại của stock.
        """
        rest = self.rest
        combined = list(zip(observation["products"], rest))
        sorted_combined = sorted(combined, key=lambda x: x[0]["size"][0] * x[0]["size"][1], reverse=True)
        sorted_products, rest = zip(*sorted_combined)
        rest = list(map(int, rest))
        sorted_products = list(sorted_products)
        # print("Prev_rest: ", rest)
        temp = np.array(self.orders, dtype=float).flatten()
        # print("Temp: ", temp)
        stock = observation["stocks"][self.last_stock_idx]
        stock_w, stock_h = self.W, self.H
        for idx, prod in enumerate(sorted_products, start=0):
            if rest[idx] <= 0:
                continue
            prod_size = prod["size"]
            prod_w, prod_h = prod_size
            prod_size_rotate = (prod_h, prod_w)
            # Thử đặt sản phẩm mà không xoay
            for x in range(stock_w):
                for y in range(stock_h):
                    if rest[idx] >= 1 and self._can_place_new_(used_area, (x, y), prod_size):
                        self.pattern.append({
                            "stock_idx": self.last_stock_idx,
                            "size": prod_size,
                            "position": (x, y)
                        })
                        self._update_used_area_(used_area, (x, y), prod_size)
                        rest[idx] -= 1
                        y += prod_h

                    elif rest[idx] != 0 and self._can_place_new_(used_area, (x, y), (prod_h, prod_w)):
                        self.pattern.append({
                            "stock_idx": self.last_stock_idx,
                            "size": (prod_h, prod_w),
                            "position": (x, y)
                        })
                        self._update_used_area_(used_area, (x, y), (prod_h, prod_w))
                        rest[idx] -= 1

    def _get_stock_size_(self, stock):
        """
        Trả về kích thước của mảnh vật liệu (stoke)
        """
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h
