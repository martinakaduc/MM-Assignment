from policy import Policy
import numpy as np
import random
import math

class SA(Policy):
    def __init__(self, initial_temperature=1000, cooling_rate=0.99, max_iterations=1000, no_improve_limit=100):
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.no_improve_limit = no_improve_limit

    def get_action(self, observation, info):
        # Tạo nghiệm khởi đầu bằng chiến lược FFD đầy đủ
        current_solution = self._generate_ffd_solution(observation)
        if not current_solution:
            # Không tạo được nghiệm khởi đầu khả thi
            return None

        current_cost = self._calculate_cost(current_solution, observation)
        best_solution = current_solution.copy()
        best_cost = current_cost

        no_improve_count = 0
        iteration = 0

        while iteration < self.max_iterations and no_improve_count < self.no_improve_limit:
            neighbor_solution = self._generate_neighbor_solution(best_solution, observation)
            neighbor_cost = self._calculate_cost(neighbor_solution, observation)

            if self._accept_solution(current_cost, neighbor_cost, self.temperature):
                current_solution = neighbor_solution
                current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = current_solution.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1

            self.temperature *= self.cooling_rate
            if self.temperature < 1e-3:
                break
            iteration += 1

        return best_solution[0] if best_solution else None
    
    def _mark_placed_(self, stock, position, size, value):
        px, py = position
        w, h = size
        # Giả sử stock là np.array 2D, truy cập bằng stock[y, x]
        # Cần chú ý: x là chiều ngang, y là chiều dọc
        stock[py:py+h, px:px+w] = value

    def _generate_ffd_solution(self, observation):
        products = observation["products"]
        # Sắp xếp sản phẩm giảm dần theo diện tích
        sorted_products = sorted(products, key=lambda x: x["size"][0]*x["size"][1], reverse=True)

        # Sao chép observation để ta có thể đánh dấu stock
        # tránh làm thay đổi observation gốc
        stocks = [s.copy() for s in observation["stocks"]]

        solution = []

        for prod_idx, prod in enumerate(sorted_products):
            prod_w, prod_h = prod["size"]
            quantity = prod.get("quantity", 0)
            for _ in range(quantity):
                placed = False
                # Thử đặt sản phẩm trên các stock
                for s_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Thử không xoay
                    if stock_w >= prod_w and stock_h >= prod_h:
                        # Tìm vị trí trống để đặt
                        pos = self._find_position_for_product(stock, (prod_w, prod_h))
                        if pos is not None:
                            px, py = pos
                            self._mark_placed_(stock, (px, py), (prod_w, prod_h), prod_idx)
                            solution.append({"stock_idx": s_idx, "size": (prod_w, prod_h), "position": (px, py)})
                            placed = True
                            break

                    # Thử xoay (nếu chưa đặt được)
                    if not placed and stock_w >= prod_h and stock_h >= prod_w:
                        pos = self._find_position_for_product(stock, (prod_h, prod_w))
                        if pos is not None:
                            px, py = pos
                            self._mark_placed_(stock, (px, py), (prod_h, prod_w), prod_idx)
                            solution.append({"stock_idx": s_idx, "size": (prod_h, prod_w), "position": (px, py)})
                            placed = True
                            break

                if not placed:
                    # Không thể đặt được tất cả sản phẩm => nghiệm không khả thi
                    return []

        return solution

    def _find_position_for_product(self, stock, prod_size):
        """
        Tìm vị trí để đặt sản phẩm trên stock bằng cách quét từ trên xuống dưới, trái sang phải.
        prod_size: (w, h)
        Giả sử _can_place_ kiểm tra được chỗ trống.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    return x, y
        return None

    def _generate_neighbor_solution(self, solution, observation):
        """
        Sinh nghiệm láng giềng:
        - 50%: di chuyển vị trí 1 sản phẩm
        - 30%: xoay 1 sản phẩm (nếu có thể)
        - 20%: hoán đổi vị trí 2 sản phẩm
        """
        if not solution:
            return solution

        new_solution = [s.copy() for s in solution]
        move_type = random.random()

        if move_type < 0.5:
            # Di chuyển vị trí 1 sản phẩm
            idx = random.randint(0, len(new_solution)-1)
            product = new_solution[idx]
            stock_idx = random.randint(0, len(observation["stocks"])-1)
            pos_x, pos_y = self._find_random_position_in_stock(observation["stocks"][stock_idx], product["size"])
            if pos_x is not None:
                new_solution[idx] = {"stock_idx": stock_idx, "size": product["size"], "position": (pos_x, pos_y)}

        elif move_type < 0.8:
            # Xoay 1 sản phẩm 90 độ (nếu có thể)
            idx = random.randint(0, len(new_solution)-1)
            product = new_solution[idx]
            rotated_size = (product["size"][1], product["size"][0])
            stock = observation["stocks"][product["stock_idx"]]
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = rotated_size
            px, py = product["position"]
            # Kiểm tra có thể xoay tại chỗ được không
            if prod_w <= stock_w and prod_h <= stock_h and self._can_place_(stock, (px, py), rotated_size):
                new_solution[idx]["size"] = rotated_size

        else:
            # Hoán đổi vị trí 2 sản phẩm
            if len(new_solution) > 1:
                idx1, idx2 = random.sample(range(len(new_solution)), 2)
                new_solution[idx1], new_solution[idx2] = new_solution[idx2], new_solution[idx1]

        return new_solution

    def _calculate_cost(self, solution, observation):
        """
        Chi phí là tổng diện tích dư trên tất cả các stock.
        Với nghiệm đầy đủ, ta có thể tính diện tích sử dụng và trừ đi diện tích tổng.
        """
        # Tạo bản sao stocks để đánh dấu chỗ đặt
        stocks = [s.copy() for s in observation["stocks"]]
        # Đánh dấu tất cả sản phẩm trong solution
        for action in solution:
            stock_idx = action["stock_idx"]
            prod_w, prod_h = action["size"]
            px, py = action["position"]
            self._mark_placed_(stocks[stock_idx], (px, py), (prod_w, prod_h), 1) # 1: đặt chỗ

        total_waste = 0
        for s_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            total_area = stock_w * stock_h
            used_area = np.sum(stock != -1)  # giả sử -1 là trống
            waste = total_area - used_area
            total_waste += waste

        return total_waste

    def _accept_solution(self, current_cost, neighbor_cost, temperature):
        if neighbor_cost < current_cost:
            return True
        return random.random() < math.exp((current_cost - neighbor_cost) / temperature)

    def _find_random_position_in_stock(self, stock, prod_size):
        """
        Tìm vị trí ngẫu nhiên có thể đặt sản phẩm.
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        if stock_w < prod_w or stock_h < prod_h:
            return None, None
        for _ in range(50):
            pos_x = random.randint(0, stock_w - prod_w)
            pos_y = random.randint(0, stock_h - prod_h)
            if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                return pos_x, pos_y
        return None, None