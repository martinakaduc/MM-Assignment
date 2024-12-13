from policy import Policy, RandomPolicy
import numpy as np

class Policy2211257(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        if policy_id == 1:
            # Initialize parameters for Q-learning
            self.learning_rate = 0.1
            self.discount_factor = 0.95
            self.exploration_rate = 0.2
            self.q_table = {}
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._q_learning_action(observation, info)
        elif self.policy_id == 2:
            return self._column_generation_action(observation, info)

    # === Q-Learning Functions ===
    def _get_state_key(self, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        return str(stocks) + str(products)
    
    def _q_learning_action(self, observation, info):
        state_key = self._get_state_key(observation)
        # Kiểm tra nếu trạng thái chưa có trong bảng Q
        if state_key not in self.q_table:
            self.q_table[state_key] = {}  # Khởi tạo giá trị Q cho trạng thái mới
        # Xác định hành động bằng cách chọn ngẫu nhiên hoặc chọn hành động tối ưu (theo epsilon-greedy)
        if np.random.rand() < self.exploration_rate:
            action = self._random_action(observation)
        else:
            action = self._best_action(state_key, observation)
        # Cập nhật bảng Q sau mỗi lần lấy hành động
        self._update_q_table(state_key, action, observation, info)
        return action
    
    def _random_action(self, observation):
        list_prods = observation["products"]
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        return {"stock_idx": i, "size": prod_size, "position": (pos_x, pos_y)}
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    def _best_action(self, state_key, observation):
        actions = self.q_table[state_key]
        if not actions:
            return self._random_action(observation)
        best_action_key = max(actions, key=actions.get)  # Đây là tuple
        return {
            "stock_idx": best_action_key[0],
            "size": list(best_action_key[1]),
            "position": list(best_action_key[2])
        }
    
    def _update_q_table(self, state_key, action, observation, info):
        action_key = (action["stock_idx"], tuple(action["size"]), tuple(action["position"]))
        reward = self._calculate_reward(action, observation)
        next_state_key = self._get_state_key(observation)
        current_q = self.q_table[state_key].get(action_key, 0)
        max_future_q = max(self.q_table.get(next_state_key, {}).values(), default=0)
        new_q = (1 - self.learning_rate) * current_q + \
                self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[state_key][action_key] = new_q
    
    def _calculate_reward(self, action, observation):
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        if stock_idx == -1:
            return -20  # Penalize invalid actions more heavily.

        stock = observation["stocks"][stock_idx]
        if self._can_place_(stock, position, size):
            filled_area = size[0] * size[1]
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_area = stock_w * stock_h
            remaining_area = stock_area - filled_area
            
            # Reward proportional to the ratio of the filled area.
            filled_ratio = filled_area / stock_area
            waste_penalty = remaining_area / stock_area  # Higher waste leads to a penalty.
            
            return 100 * filled_ratio - 10 * waste_penalty
        return -10  # Penalize invalid placements.
        
    # === Column Generation Functions ===
    def _column_generation_action(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        sorted_products = sorted(products, key=lambda x: np.prod(x['size']), reverse=True)
        for i, stock in enumerate(stocks):
            for product in sorted_products:
                if product["quantity"] == 0:
                    continue
                pos_x, pos_y = self._find_position(stock, product["size"])
                prod_w, prod_h = product["size"]
                if pos_x is not None and pos_y is not None and self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                    return {
                        "stock_idx": i,
                        "size": product["size"],
                        "position": (pos_x, pos_y)
                    }
        return RandomPolicy().get_action(observation, info)

    def _find_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                    return x, y
        return None, None
