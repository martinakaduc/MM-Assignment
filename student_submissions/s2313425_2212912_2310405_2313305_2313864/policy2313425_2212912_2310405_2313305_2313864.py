import random
import numpy as np
from policy import Policy, GreedyPolicy, RandomPolicy

class ColumnPolicy(Policy):
    def __init__(self):
        self.current_stock_idx = 0

    def _find_best_position(self, stock, prod_size):
        """Tìm vị trí tốt nhất"""
        L, W = self._get_stock_size_(stock)
        l, w = prod_size
        
        # Kiểm tra nhanh kích thước
        if l > L or w > W:
            return None, None

        # Thử các góc trước
        corners = [(0, 0), (0, W-w), (L-l, 0), (L-l, W-w)]
        for x, y in corners:
            if x >= 0 and y >= 0 and self._can_place_(stock, (x, y), [l, w]):
                return (x, y), [l, w]
            if l != w and x >= 0 and y >= 0 and self._can_place_(stock, (x, y), [w, l]):
                return (x, y), [w, l]

        # Thử các vị trí khác với bước nhảy
        step = max(1, min(l, w) // 4)
        for x in range(0, L - l + 1, step):
            for y in range(0, W - w + 1, step):
                if self._can_place_(stock, (x, y), [l, w]):
                    return (x, y), [l, w]
                if l != w and self._can_place_(stock, (x, y), [w, l]):
                    return (x, y), [w, l]

        return None, None

    def get_action(self, observation, info):
        list_prods = observation["products"]
        active_prods = [p for p in list_prods if p.get("quantity", 0) > 0]
        
        if not active_prods:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Sắp xếp sản phẩm theo diện tích giảm dần
        active_prods.sort(key=lambda p: -(p["size"][0] * p["size"][1]))

        # Thử từng sản phẩm với tấm vật liệu hiện tại
        if self.current_stock_idx < len(observation["stocks"]):
            for prod in active_prods:
                pos, size = self._find_best_position(
                    observation["stocks"][self.current_stock_idx], 
                    prod["size"]
                )
                if pos:
                    return {
                        "stock_idx": self.current_stock_idx,
                        "size": size,
                        "position": pos
                    }
            # Nếu không đặt được trên tấm hiện tại, chuyển sang tấm tiếp theo
            self.current_stock_idx += 1

        # Nếu đã hết tấm hoặc không tìm được vị trí, thử lại từ đầu
        self.current_stock_idx = 0
        for stock_idx, stock in enumerate(observation["stocks"]):
            for prod in active_prods:
                pos, size = self._find_best_position(stock, prod["size"])
                if pos:
                    self.current_stock_idx = stock_idx
                    return {
                        "stock_idx": stock_idx,
                        "size": size,
                        "position": pos
                    }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

class BnBPolicy(Policy):
    def __init__(self, branching_factor=3, use_greedy=True):
        super().__init__()
        self.branching_factor = branching_factor
        self.use_greedy = use_greedy
        
        # Chọn chính sách greedy hoặc random
        if self.use_greedy:
            self.policy = GreedyPolicy()
        else:
            self.policy = RandomPolicy()
    
    def get_action(self, observation, info):
        best_solution = None
        best_cost = float('inf')

        # Đệ quy branch and bound
        def branch_and_bound(curr_solution, curr_cost, remaining_products, heuristic_cost):
            nonlocal best_solution, best_cost

            # Nếu không còn sản phẩm, ta đã có một giải pháp hoàn chỉnh
            if not remaining_products:
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_solution = curr_solution
                return

            # Nếu giá trị hiện tại không khả thi, quay lại
            if curr_cost + heuristic_cost >= best_cost:
                return

            # Sắp xếp lại các sản phẩm theo một số tiêu chí như diện tích hoặc nhu cầu
            remaining_products_sorted = sorted(remaining_products, key=lambda p: p['size'][0] * p['size'][1])

            # Thử các nhánh tiếp theo (sử dụng chính sách greedy hoặc random)
            product = remaining_products_sorted[0]
            for _ in range(self.branching_factor):
                action = self.policy.get_action(observation, info)
                stock_idx = action['stock_idx']
                size = action['size']
                position = action['position']
                
                # Kiểm tra tính khả thi trước khi thực hiện hành động
                if self.is_feasible(observation, stock_idx, size, position):
                    new_solution = curr_solution + [action]
                    new_cost = self.calculate_cost(new_solution)
                    new_heuristic_cost = self.calculate_heuristic_cost(remaining_products_sorted[1:])
                    branch_and_bound(new_solution, new_cost, remaining_products_sorted[1:], new_heuristic_cost)
                
                # Kiểm tra các trường hợp xoay sản phẩm
                rotated = (np.transpose(size))
                if self.is_feasible(observation, stock_idx, rotated, position):
                    action['size'] = rotated
                    new_solution = curr_solution + [action]
                    new_cost = self.calculate_cost(new_solution)
                    new_heuristic_cost = self.calculate_heuristic_cost(remaining_products_sorted[1:])
                    branch_and_bound(new_solution, new_cost, remaining_products_sorted[1:], new_heuristic_cost)

        # Khởi tạo danh sách các sản phẩm chưa được chọn
        remaining_products = [prod for prod in observation['products'] if prod['quantity'] > 0]

        # Tính toán chi phí heuristic cho các sản phẩm còn lại
        initial_heuristic_cost = self.calculate_heuristic_cost(remaining_products)

        # Khởi tạo giải pháp ban đầu
        branch_and_bound([], 0, remaining_products, initial_heuristic_cost)

        # Trả về giải pháp tốt nhất
        return best_solution[-1] if best_solution else self.policy.get_action(observation, info)

    def calculate_cost(self, solution):
        # Hàm tính toán chi phí của giải pháp. Ví dụ có thể tính theo số lượng sản phẩm đã cắt.
        return len(solution)

    def calculate_heuristic_cost(self, remaining_products):
        # Heuristic cost: Tính toán chi phí ước tính dựa trên sản phẩm còn lại
        # Có thể tính theo số lượng sản phẩm chưa được xử lý, diện tích còn lại trong kho, v.v.
        total_area_left = 0
        for prod in remaining_products:
            prod_w, prod_h = prod['size']
            total_area_left += prod_w * prod_h
        return total_area_left

    def is_feasible(self, observation, stock_idx, size, position):
        # Kiểm tra xem hành động có khả thi không, ví dụ, kiểm tra xem sản phẩm có thể được đặt vào kho không.
        stock = observation['stocks'][stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = size
        x, y = position
        spined_size = (prod_h, prod_w)
        return stock_w >= prod_w and stock_h >= prod_h and self._can_place_(stock, position, size)

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)


class Policy2313425_2212912_2310405_2313305_2313864(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        self.policy_id = policy_id
        # Nếu policy_id là 1, sử dụng BnBPolicy
        if self.policy_id == 1:
            self.bnb_policy = BnBPolicy()

        # Nếu policy_id là 2, sử dụng ColumnPolicy
        if self.policy_id == 2:
            self.column_policy = ColumnPolicy()

    def get_action(self, observation, info):
        if self.policy_id == 1:
            # Thực hiện giải thuật Branch and Bound khi policy_id = 1
            return self.bnb_policy.get_action(observation, info)
        elif self.policy_id == 2:
            # Thực hiện ColumnPolicy khi policy_id = 2
            return self.column_policy.get_action(observation, info)
    