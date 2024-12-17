from policy import Policy
import math
import random
class Policy2312664_2313303_2313508_2213696_1852316(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id
        self.remaining_areas = []
        self.initial_temperature = 100
        self.cooling_rate = 0.99

    def get_action(self, observation, info):
        # Student code here
        return self.BF_1_Policy(observation) if self.policy_id == 1 else self.FFD_Policy(observation)
        
#BF
    def BF_1_Policy(self, observation):
        # Sắp xếp danh sách sản phẩm theo kích thước (diện tích) giảm dần
        sorted_products = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )
        # Sắp xếp kho nguyên liệu theo kích thước (diện tích) giảm dần
        sorted_stocks = sorted(
            enumerate(observation["stocks"]),
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
            reverse=True
        )
         # Khởi tạo mảng diện tích còn lại cho từng kho: mỗi kho có diện tích ban đầu là diện tích kho
        self.remaining_areas = [self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for _, stock in sorted_stocks]
        # Duyệt qua từng kho nguyên liệu
        for i, stock in sorted_stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod in sorted_products:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    # Kiểm tra nếu kích thước kho có đủ để chứa sản phẩm
                    smallest_product = sorted_products[-1]
                    smallest_w, smallest_h = smallest_product["size"]
                    smallest_area = smallest_w * smallest_h
                    if smallest_area > self.remaining_areas[i]: break
                    if self.remaining_areas[i]<prod_w*prod_h: continue
                    # Khám phá các vị trí khả thi trong kho
                    best_solution = None
                    best_fit = float('inf')
                    for m in [prod_size, prod_size[::-1]]:
                        prod_w, prod_h = m
                        if stock_w < prod_w or stock_h < prod_h: continue
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), m):
                                    # Tính toán độ vừa vặn
                                    fit = (stock_w - x) * (stock_h - y)
                                    # Kiểm tra và cập nhật nếu tìm được vị trí phù hợp hơn
                                    if fit < best_fit:
                                        best_fit = fit
                                        best_solution = {
                                            "stock_idx": i,
                                            "size": m,
                                            "position": (x, y)
                                        }
                    # Nếu tìm được vị trí phù hợp cho sản phẩm, giảm số lượng và trả về
                    if best_solution:
                        # Cập nhật kích thước còn lại
                        self.remaining_areas[i] -= prod_w * prod_h
                        # Trả về kết quả
                        return best_solution

        # Trả về giá trị mặc định nếu không tìm được vị trí cho bất kỳ sản phẩm nào
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
#SA
    def SA_Policy(self, observation):
        current_temperature = self.initial_temperature

        # Khởi tạo giải pháp ban đầu
        current_solution = {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (0, 0)
        }
        best_solution = current_solution
        best_fit = float('inf')

        sorted_products = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )
        sorted_stocks = sorted(
            enumerate(observation["stocks"]),
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
            reverse=True
        )
        best_solution = None
        while current_temperature > 1e-8:
            for i, stock in sorted_stocks:
                stock_w, stock_h = self._get_stock_size_(stock)
                for prod in sorted_products:
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        for m in [prod_size, prod_size[::-1]]:
                            prod_w, prod_h = m
                            if stock_w < prod_w or stock_h < prod_h:
                                continue

                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):

                                    if self._can_place_(stock, (x, y), m):
                                        solution = {
                                                "stock_idx": i,
                                                "size": m,
                                                "position": (x, y)
                                            }
                                        fit = self.calculate_used_empty_area_with_best_solution(observation,solution)
                                        if fit < best_fit or random.uniform(0, 1) < math.exp(-(fit - best_fit) / current_temperature):
                                            best_fit = fit
                                            best_solution = solution

                            # Giảm nhiệt độ
                        current_temperature *= self.cooling_rate
                if best_solution:
                   return best_solution
    def calculate_used_empty_area_with_best_solution(self, observation, solution):
        total_empty_area = 0

        # Duyệt qua từng stock
        for stock in observation["stocks"]:
            stock_dict = stock if isinstance(stock, dict) else {}
            stock_width, stock_height = stock_dict.get("width", 0), stock_dict.get("height", 0)
            stock_area = stock_width * stock_height

            # Tính diện tích đã sử dụng trong stock này
            used_area = 0
            for product in observation["products"]:
                if product.get("stock_idx") == stock_dict.get("id"):  # Kiểm tra sản phẩm đã đặt vào stock này
                    product_width, product_height = product.get("size", (0, 0))
                    used_area += product_width * product_height * product.get("used_quantity", 0)

            # Nếu stock là stock được giả định thêm sản phẩm từ best_solution
            if solution and stock_dict.get("id") == solution["stock_idx"]:
                product_width, product_height = solution.get("size", (0, 0))
                used_area += product_width * product_height

            # Chỉ tính phần trống nếu stock này đã được sử dụng
            if used_area > 0:
                empty_area = stock_area - used_area
                total_empty_area += empty_area

        return total_empty_area



#FFD
    def FFD_Policy(self, observation):
        # Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_products = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )
        # Sắp xếp kho nguyên liệu theo diện tích giảm dần
        sorted_stocks = sorted(
            enumerate(observation["stocks"]),
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
            reverse=True
        )
        # Duyệt qua các sản phẩm
        for prod in sorted_products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                # Khởi tạo biến best_solution
                best_solution = None
                # Duyệt qua các kho nguyên liệu
                for i, stock in sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    # Kiểm tra nếu kho có đủ diện tích cho sản phẩm
                    if (stock_w >= prod_w and stock_h >= prod_h) or (stock_w >= prod_h and stock_h >= prod_w):
                        placed = False  # Đặt biến placed là False trước khi tìm vị trí cắt
                        # Duyệt qua các vị trí khả thi trong kho
                        if (stock_w >= prod_w and stock_h >= prod_h):
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        # Cắt sản phẩm vào kho tại vị trí (x, y)
                                        placed = True  # Đánh dấu rằng sản phẩm đã được cắt thành công
                                        best_solution = {
                                            "stock_idx": i,
                                            "size": prod_size,
                                            "position": (x, y)
                                        }
                                        if placed: break  # Ngừng duyệt kho nguyên liệu khác vì đã tìm được vị trí hợp lệ
                                if placed: break  # Ngừng duyệt kho nguyên liệu khác vì đã tìm được vị trí hợp lệ        
                            if placed: break  # Ngừng duyệt kho nguyên liệu khác vì đã tìm được vị trí hợp lệ
                        if not placed:
                            if stock_w >= prod_h and stock_h >= prod_w:
                                for x in range(stock_w - prod_h + 1):
                                    for y in range(stock_h - prod_w + 1):
                                        if self._can_place_(stock, (x, y), prod_size[::-1]):
                                            placed = True
                                            best_solution = {
                                                "stock_idx": i,
                                                "size": prod_size[::-1],
                                                "position": (x, y)
                                            }
                                            if placed: break
                                    if placed: break
                                if placed: break
                if best_solution: return best_solution
        # Nếu không tìm được vị trí cắt cho bất kỳ sản phẩm nào, trả về giá trị mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    


    def BF_2_Policy(self, observation):
        # Sắp xếp danh sách sản phẩm theo kích thước (diện tích) giảm dần
        sorted_products = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )
        
        # Sắp xếp kho nguyên liệu theo kích thước (diện tích) giảm dần
        sorted_stocks = sorted(
            enumerate(observation["stocks"]),
            key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
            reverse=True
        )

        # Khởi tạo mảng diện tích còn lại cho từng kho
        self.remaining_areas = [
            self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]
            for _, stock in sorted_stocks
        ]
        
        # Duyệt qua từng kho nguyên liệu
        for i, stock in sorted_stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod in sorted_products:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    # Kiểm tra nếu kích thước kho có đủ để chứa sản phẩm
                    smallest_product = sorted_products[-1]
                    smallest_w, smallest_h = smallest_product["size"]
                    smallest_area = smallest_w * smallest_h
                    if smallest_area > self.remaining_areas[i]:
                        break
                    if self.remaining_areas[i] < prod_w * prod_h:
                        continue
                    
                    # Khám phá các vị trí khả thi trong kho
                    best_solution = None
                    best_cut_count = float('inf')  # Khởi tạo số lần cắt tối thiểu

                    for m in [prod_size, prod_size[::-1]]:
                        prod_w, prod_h = m
                        if stock_w < prod_w or stock_h < prod_h:
                            continue
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), m):
                                    # Tính toán số lần cắt (cut count)
                                    cut_count = self._calculate_cut_count(stock, (x, y), m)
                                    
                                    # Cập nhật nếu số lần cắt ít hơn
                                    if cut_count < best_cut_count:
                                        best_cut_count = cut_count
                                        best_solution = {
                                            "stock_idx": i,
                                            "size": m,
                                            "position": (x, y),
                                            "cut_count": cut_count  # Lưu thông tin số lần cắt
                                        }

                    # Nếu tìm được vị trí phù hợp cho sản phẩm, giảm số lượng và trả về
                    if best_solution:
                        # Cập nhật kích thước còn lại
                        self.remaining_areas[i] -= prod_w * prod_h
                        return best_solution

        # Trả về giá trị mặc định nếu không tìm được vị trí cho bất kỳ sản phẩm nào
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _calculate_cut_count(self, stock, position, size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = size
        remaining_areas = []
        remaining_width = stock_w - position[0] - prod_w
        remaining_height = stock_h - position[1] - prod_h

        if position[1] > 0:
            remaining_areas.append(position[1] * stock_w)  

        if remaining_height > 0:
            remaining_areas.append(remaining_height * stock_w)  

        if position[0] > 0:
            remaining_areas.append(position[0] * prod_h)  

        if remaining_width > 0:
            remaining_areas.append(remaining_width * prod_h)  

        cut_count = len(remaining_areas)
        return cut_count
