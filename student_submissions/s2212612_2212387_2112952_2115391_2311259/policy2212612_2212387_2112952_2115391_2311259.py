import math
import numpy as np
from policy import Policy


class Policy2212612_2212387_2112952_2115391_2311259(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        
        self.policy_id = policy_id
        if self.policy_id == 1:
            self.policy = SarsaPolicy()
        elif self.policy_id == 2:
            self.policy = GreedyRotateDecreasing()

    def get_action(self, observation, info):
        """
        Gọi hàm `get_action` của policy tương ứng.
        """
        return self.policy.get_action(observation, info)
    # Student code here
    # You can add more functions if needed

class SarsaPolicy(Policy):
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        super().__init__()
        self.sarsa_table = {}
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def get_action(self, observation, info):
        """SARSA-based action selection."""
        state_key = self._generate_state_key(observation)

        # Khởi tạo SARSA table nếu là trạng thái mới
        if state_key not in self.sarsa_table:
            self.sarsa_table[state_key] = {}

        # Áp dụng epsilon-greedy để chọn hành động
        if np.random.rand() < self.exploration_rate:
            action = self._random_action(observation)  # Truyền observation vào đây
        else:
            action = self._select_best_action_sarsa(state_key, observation)  # Truyền observation vào đây

        # Lấy phần thưởng và cập nhật SARSA table
        reward = self._calculate_reward(action, observation)
        next_state_key = self._generate_state_key(observation)
        
        # Chọn hành động tiếp theo (next_action) cho SARSA
        next_action = self._random_action(observation) if np.random.rand() < self.exploration_rate else self._select_best_action_sarsa(next_state_key, observation)

        # Cập nhật SARSA table
        self._update_sarsa_table(state_key, action, reward, next_state_key, next_action)
        return action

    def _update_sarsa_table(self, state_key, action, reward, next_state_key, next_action):
        """Cập nhật giá trị SARSA table."""
        action_key = (action["stock_idx"], tuple(action["size"]), tuple(action["position"]))
        next_action_key = (next_action["stock_idx"], tuple(next_action["size"]), tuple(next_action["position"]))

        current_q = self.sarsa_table[state_key].get(action_key, 0)
        next_q = self.sarsa_table.get(next_state_key, {}).get(next_action_key, 0)

        self.sarsa_table[state_key][action_key] = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * next_q)

    def _select_best_action_sarsa(self, state_key, observation):
        """Chọn hành động tốt nhất dựa trên giá trị SARSA."""
        actions = self.sarsa_table[state_key]
        return max(actions, key=actions.get, default=self._random_action(observation))

    def _generate_state_key(self, observation):
        """Tạo khóa đại diện cho trạng thái từ observation."""
        stocks = observation["stocks"]
        products = observation["products"]
        return str(stocks) + str(products)

    def _random_action(self, observation):
        """Chọn một action hợp lệ bất kỳ."""
        for product in observation["products"]:
            if product["quantity"] > 0:
                for stock_idx, stock in enumerate(observation["stocks"]):
                    position_x, position_y = self._find_position(stock, product["size"])
                    if position_x is not None and position_y is not None:
                        return {
                            "stock_idx": stock_idx,
                            "size": product["size"],
                            "position": (position_x, position_y)
                        }
        # Nếu không có action hợp lệ, trả về giá trị mặc định
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_position(self, stock, prod_size):
        """Tìm vị trí có thể đặt product."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None, None

    def _calculate_reward(self, action, observation):
        """Tính toán reward dựa trên diện tích đã cắt."""
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        if stock_idx == -1:
            return -5  # Trừ điểm nếu không tìm thấy action hợp lệ

        stock = observation["stocks"][stock_idx]
        if self._can_place_(stock, position, size):
            filled_ratio = np.prod(size) / np.prod(self._get_stock_size_(stock))
            return 10 + filled_ratio * 20  # Tăng phần thưởng dựa trên diện tích đã cắt
        return -10  # Phạt nếu không thể cắt

class GreedyRotateDecreasing(Policy):
    def __init__(self):
        pass

    def _get_sorted_products_(self, products):
        """
        Trả về danh sách sản phẩm có quantity > 0 được sắp xếp theo diện tích giảm dần
        """
        valid_products = []
        for i, prod in enumerate(products):
            if prod["quantity"] > 0:
                # Tính diện tích và lưu index để còn biết là sản phẩm nào
                area = int(prod["size"][0]) * int(prod["size"][1])
                valid_products.append((i, area, prod))
        
        # Sắp xếp theo diện tích giảm dần
        valid_products.sort(key=lambda x: x[1], reverse=True)
        return valid_products

    def _find_position_in_stock_(self, stock, prod_size):
        """
        Tìm vị trí đặt được sản phẩm trong stock với kích thước cụ thể
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w < prod_w or stock_h < prod_h:
            return None

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None

    def get_action(self, observation, info):
        stocks = observation["stocks"]
        sorted_products = self._get_sorted_products_(observation["products"])
        
        if not sorted_products:  # Nếu không còn sản phẩm nào để cắt
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Tìm stock đầu tiên đã được sử dụng hoặc stock mới nếu chưa có stock nào được sử dụng
        current_stock_idx = 0
        while current_stock_idx < len(stocks):
            current_stock = stocks[current_stock_idx]
            
            # Thử từng sản phẩm theo thứ tự diện tích giảm dần
            for prod_idx, area, prod in sorted_products:
                # Chuẩn bị cả hai kích thước (gốc và xoay)
                original_size = prod["size"]
                rotated_size = np.array([int(prod["size"][1]), int(prod["size"][0])])
                
                # Thử với kích thước gốc trước
                position = self._find_position_in_stock_(current_stock, original_size)
                if position is not None:
                    return {
                        "stock_idx": current_stock_idx,
                        "size": original_size,
                        "position": position
                    }
                
                # Nếu không được, thử với kích thước đã xoay
                position = self._find_position_in_stock_(current_stock, rotated_size)
                if position is not None:
                    prod["size"] = rotated_size
                    return {
                        "stock_idx": current_stock_idx,
                        "size": rotated_size,
                        "position": position
                    }
            
            # Nếu không đặt được sản phẩm nào vào stock hiện tại, chuyển sang stock tiếp theo
            current_stock_idx += 1

        # Nếu không tìm được vị trí phù hợp trong bất kỳ stock nào
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def reset(self):
        """Reset policy state"""
        pass
