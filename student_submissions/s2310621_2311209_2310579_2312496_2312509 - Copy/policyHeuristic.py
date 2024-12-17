from policy import Policy
import numpy as np
import random
import math


class PolicyHeuristic(Policy):
    def __init__(self, policy_id):
        """
        Initialize the policy with either Simulated Annealing or Dynamic Programming.
        :param policy_id: 1 for Dynamic Programming, 2 for Simulated Annealing
        """
        assert policy_id in [1, 2], "Policy ID must be 1 (Dynamic Programming) or 2 (Simulated Annealing)"
        self.policy_id = policy_id

        if self.policy_id == 1:
            # Initialization for Dynamic Programming
            pass
        elif self.policy_id == 2:
            # Initialization for Simulated Annealing
            self.initial_temp = 100
            self.cooling_rate = 0.99
            self.temp = self.initial_temp
            self.temp_min = 1
            self.max_iterations = 1000

    def get_action(self, observation, info):
        """
        Determine the action based on the chosen policy (Dynamic Programming or Simulated Annealing).
        :param observation: State of the environment (products and stocks).
        :param info: Additional environment information.
        :return: Dictionary with stock_idx, size, and position.
        """
        if self.policy_id == 1:
            return self.dp_action(observation, info)
        elif self.policy_id == 2:
            return self.sa_action(observation, info)
        
#--------------------------------------- Dynamic Programming ------------------------------------------------------------------------

    def ffh_action(self, observation, info):
        list_prods = observation["products"]
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        if all(prod["quantity"] == 0 for prod in list_prods):
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

        max_stock_w = max(stock.shape[0] for stock in observation["stocks"])
        max_stock_h = max(stock.shape[1] for stock in observation["stocks"])

        products = list(list_prods) if isinstance(list_prods, tuple) else list_prods
        
        # Sắp xếp sản phẩm theo kích thước (diện tích sản phẩm)
        products.sort(key=lambda x: x["size"][0] * x["size"][1], reverse=False)

        for prod in products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                if prod["quantity"] > 0:
                    if prod["size"][0] > max_stock_w or prod["size"][1] > max_stock_h:
                        continue

                # Iterate through all stocks to find the best placement
                for prod_size in [(prod["size"][0], prod["size"][1]), (prod["size"][1], prod["size"][0])]:
                    for i, stock in enumerate(observation["stocks"]):
                        position = self._find_best_position_(stock, prod_size)
                        if position:
                            pos_x, pos_y = position
                            stock_idx = i
                            break

                    if stock_idx != -1:
                        break
        
        if stock_idx == -1:
            return {"stock_idx": -1, "size": prod_size, "position": (0, 0)}

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
    def _find_best_position_(self, stock, prod_size):
        stock_w, stock_h = stock.shape
        prod_w, prod_h = prod_size
        # Xác minh nếu sản phẩm lớn hơn không gian kho
        if prod_w > stock_w or prod_h > stock_h:
            return None

        # Sử dụng ma trận boolean để đánh dấu vùng trống
        available_space = (stock == -1)
        for i in range(stock_w - prod_w + 1):
            for j in range(stock_h - prod_h + 1):
                sub_region = available_space[i:i + prod_w, j:j + prod_h]
                if np.all(sub_region):
                    return (i, j)       

        # Nếu không thể xếp vào stock hiện tại, trả về None để chuyển sang stock khác
        return None
#------------------------------------------ Simulated Annealing ---------------------------------------------------------------------
    def sa_action(self, observation, info):
        """
        Lấy hành động bằng cách áp dụng Simulated Annealing để tìm giải pháp tốt nhất.
        :param observation: Trạng thái hiện tại của môi trường.
        :param info: Thông tin bổ sung từ môi trường.
        :return: Từ điển chứa stock_idx, size, và position.
        """
        products = observation["products"]
        stocks = observation["stocks"]

        # Sinh giải pháp ban đầu
        current_solution = self._generate_initial_solution(products, stocks)
        best_solution = current_solution

        while self.temp > 1:
            # Sinh giải pháp lân cận
            neighbor_solution = self._generate_neighbor_solution(current_solution, products, stocks)

            # Đánh giá năng lượng của các giải pháp
            current_energy = self._evaluate_solution(current_solution, products, stocks)
            neighbor_energy = self._evaluate_solution(neighbor_solution, products, stocks)

            # Quyết định chấp nhận giải pháp lân cận
            if self._acceptance_probability(current_energy, neighbor_energy, self.temp) > random.random():
                current_solution = neighbor_solution

            # Cập nhật giải pháp tốt nhất
            if neighbor_energy < self._evaluate_solution(best_solution, products, stocks):
                best_solution = neighbor_solution

            # Giảm nhiệt độ
            self.temp *= self.cooling_rate

            # Kiểm tra nếu không thể đặt sản phẩm vào stock hiện tại
            if isinstance(best_solution, dict) and best_solution["stock_idx"] == -1:
                # Nếu không thể đặt vào stock hiện tại, chuyển sang stock tiếp theo
                next_stock_idx = (current_solution["stock_idx"] + 1) % len(stocks)
                best_solution["stock_idx"] = next_stock_idx

        # Đảm bảo trả về một dictionary chứa các thông tin cần thiết
        if isinstance(best_solution, list):
            # Nếu best_solution là list, lấy phần tử đầu tiên
            best_solution = best_solution[0]

        if not isinstance(best_solution, dict):
            raise ValueError("Invalid solution generated by SA.")

        return best_solution
    
    def _generate_initial_solution(self, products, stocks):
        """
        Sinh giải pháp ban đầu ngẫu nhiên, kiểm tra cả khả năng xoay 90 độ.
        """
        products = list(products) if isinstance(products, tuple) else products
        
        # Sắp xếp sản phẩm theo kích thước (chiều dài or chiều rộng)
        products.sort(key=lambda x: max(x["size"][0], x["size"][1]), reverse=True)
        
        for product in products:
            if product["quantity"] > 0:
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = product["size"]

                    # Thử đặt sản phẩm theo hướng gốc
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x_pos in range(stock_w - prod_w + 1):
                            for y_pos in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x_pos, y_pos), (prod_w, prod_h)):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": (prod_w, prod_h),
                                        "position": (x_pos, y_pos),
                                    }

                    # Thử xoay 90 độ
                    if stock_w >= prod_h and stock_h >= prod_w:
                        for x_pos in range(stock_w - prod_h + 1):
                            for y_pos in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x_pos, y_pos), (prod_h, prod_w)):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": (prod_h, prod_w),
                                        "position": (x_pos, y_pos),
                                    }

        # Trả về giải pháp mặc định nếu không tìm thấy
        return {"stock_idx": -1}

    def _generate_neighbor_solution(self, current_solution, products, stocks):
        """
        Sinh giải pháp lân cận bằng cách thay đổi vị trí hoặc hướng xoay.
        """
        if not current_solution or "stock_idx" not in current_solution:
            return current_solution 

        stock_idx = current_solution["stock_idx"]
        if stock_idx == -1:
            return current_solution

        stock = stocks[stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)

        prod_w, prod_h = current_solution["size"]
        x_pos = random.randint(0, stock_w - prod_w)
        y_pos = random.randint(0, stock_h - prod_h)

        if self._can_place_(stock, (x_pos, y_pos), (prod_w, prod_h)):
            return {"stock_idx": stock_idx, "size": (prod_w, prod_h), "position": (x_pos, y_pos)}
        
        return current_solution


    def _evaluate_solution(self, solution, products, stocks):
        """
        Đánh giá năng lượng của giải pháp: Diện tích lãng phí.
        """
        stock_idx = solution["stock_idx"]
        if stock_idx == -1:
            return float("inf")  # Không đặt được sản phẩm => năng lượng cực cao

        stock = stocks[stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = solution["size"]

        # Tính lãng phí: Phần còn lại của kho sau khi đặt sản phẩm
        wasted_area = (stock_w * stock_h) - (prod_w * prod_h)
        return wasted_area


    def _acceptance_probability(self, current_energy, neighbor_energy, temp):
        """
        Tính xác suất chấp nhận giải pháp kém hơn dựa trên nhiệt độ.
        """
        if neighbor_energy < current_energy:
            return 1.0
        return np.exp(((current_energy - neighbor_energy)) / (temp + 1e-9))

    # Student code here
    # You can add more functions if needed
