import numpy as np
import random
from policy import Policy

class Policy2352626(Policy):
    def __init__(self, policy_id=2, stocks=None, products=None):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        
        # Cấu hình chung
        self.stocks = stocks if stocks is not None else []
        self.products = products if products is not None else []
        
        # Cấu hình A2C
        self._init_a2c_config()
        
        # Cấu hình Heuristic
        self.heuristic_optimizer = HeuristicOptimizer()

    def _init_a2c_config(self):
        # Khởi tạo các tham số và cấu trúc cho A2C
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.1
        self.num_actions = 10
        self.value_table = {}
        self.entropy_beta = 0.01
        self.policy_table = {}
        self.actor_lr = 0.01
        self.critic_lr = 0.01
        self.baseline_table = {}
        self.gae_lambda = 0.95
        self.replay_buffer = []
        self.replay_buffer_maxlen = 50
        self.multi_step_n = 5
        self.priority_weights = []
        self.priority_alpha = 0.6
        self.priority_beta = 0.4

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._get_a2c_action(observation, info)
        else:
            return self._get_heuristic_action(observation, info)

    def _get_a2c_action(self, observation, info):
        state = self._get_state_key(observation, info)
        
        # Khởi tạo state nếu chưa tồn tại
        if state not in self.q_table:
            self.q_table[state] = {}
            self.value_table[state] = 0
            self.baseline_table[state] = 0

        # Khám phá hoặc khai thác
        if random.uniform(0, 1) < self.epsilon:
            action = self._ffd_action(observation)
        else:
            action = self._sample_action(observation, state)
        
        return action

    def _get_heuristic_action(self, observation, info):
        return self.heuristic_optimizer.get_action(observation, info)
    def _get_state_key(self, observation, info):
        return str(observation)
    def _ffd_action(self, observation):
        """First Fit Decreasing (FFD) heuristic for action selection."""
        products = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        for product in products:
            if product["quantity"] > 0:
                prod_size = product["size"]
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    for x in range(stock_w - prod_size[0] + 1):
                        for y in range(stock_h - prod_size[1] + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y),
                                }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _sample_action(self, observation, state):
        action_space = []
        for stock_idx, stock in enumerate(observation["stocks"]):
            for product_idx, product in enumerate(observation["products"]):
                if product["quantity"] > 0:
                    prod_size = product["size"]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    for x in range(stock_w - prod_size[0] + 1):
                        for y in range(stock_h - prod_size[1] + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                action_space.append({
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y),
                                })
                                if len(action_space) >= self.num_actions:
                                    break
                    if len(action_space) >= self.num_actions:
                        break
            if len(action_space) >= self.num_actions:
                break

        if action_space:
            action_indices = [self._get_action_index(action) for action in action_space]
            q_values = np.array([self.q_table[state].get(idx, 0) for idx in action_indices])
            probabilities = self._softmax(q_values)

            # Save policy probabilities for entropy calculation
            self.policy_table[state] = probabilities

            action_idx = np.random.choice(len(action_space), p=probabilities)
            return action_space[action_idx]

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def store_experience(self, observation, action, reward, next_observation, done):
        experience = (observation, action, reward, next_observation, done)
        priority = max(self.priority_weights, default=1.0)
        if len(self.replay_buffer) >= self.replay_buffer_maxlen:
            self.replay_buffer.pop(0)
            self.priority_weights.pop(0)
        self.replay_buffer.append(experience)
        self.priority_weights.append(priority)

    def sample_experiences(self, batch_size=2):
        priorities = np.array(self.priority_weights) ** self.priority_alpha
        probabilities = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.replay_buffer), batch_size, p=probabilities)
        experiences = [self.replay_buffer[i] for i in indices]

        # Importance-sampling weights
        total = len(self.replay_buffer)
        weights = (total * probabilities[indices]) ** (-self.priority_beta)
        weights /= weights.max()  # Normalize
        return experiences, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priority_weights[idx] = abs(error) + 1e-6  # Avoid zero priority

    def compute_multi_step_return(self, rewards, value_next_state):
        discounted_return = 0
        for t, reward in enumerate(rewards[::-1]):
            discounted_return = reward + self.gamma * discounted_return
        return discounted_return + (self.gamma ** self.multi_step_n) * value_next_state

    def update_q_table(self, state, action, next_state, reward, done):
        action_idx = self._get_action_index(action)

        if action_idx not in self.q_table[state]:
            self.q_table[state][action_idx] = 0

        if next_state not in self.q_table:
            self.q_table[next_state] = {}
            self.value_table[next_state] = 0

        next_q_values = self.q_table[next_state].values()
        max_next_q = max(next_q_values) if next_q_values else 0

        current_q = self.q_table[state][action_idx]
        value_next_state = 0 if done else self.value_table[next_state]
        value_current_state = self.value_table[state]

        # TD-error for Actor-Critic
        td_error = reward + self.gamma * value_next_state - value_current_state

        # Update value table
        self.value_table[state] += self.critic_lr * td_error

        # Compute Advantage
        delta = reward + self.gamma * max_next_q - current_q
        advantage = delta + self.gae_lambda * self.gamma * (value_next_state - value_current_state)

        # Update Q-table using Advantage estimation
        self.q_table[state][action_idx] += self.actor_lr * advantage

        # Incorporate entropy regularization
        entropy = -np.sum(self.policy_table[state] * np.log(self.policy_table[state] + 1e-8))
        self.q_table[state][action_idx] += self.entropy_beta * entropy

        return td_error


    def step(self, observation, action, reward, next_observation, info):
        state = self._get_state_key(observation, info)
        next_state = self._get_state_key(next_observation, info)

        if next_state not in self.q_table:
            self.q_table[next_state] = {}
            self.value_table[next_state] = 0

        self.store_experience(observation, action, reward, next_observation, info.get("done", False))

        if len(self.replay_buffer) >= 32:  # Perform batch updates when enough experiences are stored
            batch, indices, weights = self.sample_experiences()
            td_errors = []
            for (obs, act, rew, next_obs, done), weight in zip(batch, weights):
                td_error = self.update_q_table(
                    self._get_state_key(obs, {}), act, self._get_state_key(next_obs, {}), rew, done
                )
                td_errors.append(td_error * weight)  # Scale TD-error by importance-sampling weight
            self.update_priorities(indices, td_errors)

        return self.get_action(next_observation, info)

    def _get_action_index(self, action):
        stock_idx = action.get("stock_idx", -1)
        pos_x, pos_y = action.get("position", (0, 0))
        return (stock_idx, pos_x, pos_y)

    def _compute_reward(self, observation, action, info):
        filled_ratio = info.get("filled_ratio", 0)
        trim_loss = info.get("trim_loss", 0)

        placement_reward = 100 if action["stock_idx"] != -1 else -50
        efficiency_reward = 200 * filled_ratio
        trim_penalty = -1000 * trim_loss

        total_reward = placement_reward + efficiency_reward + trim_penalty

        total_reward = max(-2000, min(total_reward, 300))

        return total_reward

    def _softmax(self, q_values):
        max_q = np.max(q_values) if len(q_values) > 0 else 0
        exp_q = np.exp(q_values - max_q)  # For numerical stability
        return exp_q / np.sum(exp_q) if np.sum(exp_q) > 0 else np.ones_like(q_values) / len(q_values)
    
class HeuristicOptimizer:
    def get_action(self, observation, info):
        list_prods = observation["products"]
        # Sắp xếp sản phẩm theo diện tích giảm dần
        sorted_prods = sorted(
            [prod for prod in list_prods if prod["quantity"] > 0], 
            key=lambda x: x["size"][0] * x["size"][1], 
            reverse=True
        )

        for prod in sorted_prods:
            prod_size = prod["size"]

            # Tìm Stock phù hợp với diện tích thừa nhỏ nhất
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size

                # Kiểm tra kích thước Stock
                if stock_w < prod_w or stock_h < prod_h:
                    continue

                # Tìm vị trí tối ưu
                position = self._find_best_stock_position(stock, prod_size)
                
                if position is not None:
                    return {
                        "stock_idx": stock_idx, 
                        "size": prod_size, 
                        "position": position
                    }

        # Nếu không thể đặt sản phẩm, sử dụng RandomPolicy
        return {
            "stock_idx": random.randint(0, len(observation["stocks"]) - 1), 
            "size": sorted_prods[0]["size"], 
            "position": (0, 0)
        }

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)

    def _find_best_stock_position(self, stock, prod_size):
        """Tìm vị trí đặt sản phẩm với diện tích thừa nhỏ nhất"""
        prod_w, prod_h = prod_size
        stock_w, stock_h = self._get_stock_size_(stock)
        
        min_waste = float('inf')
        best_pos = None

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    # Tính diện tích thừa
                    waste_x = stock_w - (x + prod_w)
                    waste_y = stock_h - (y + prod_h)
                    waste = waste_x * waste_y

                    # Cập nhật vị trí có diện tích thừa nhỏ nhất
                    if waste < min_waste:
                        min_waste = waste
                        best_pos = (x, y)

        return best_pos


