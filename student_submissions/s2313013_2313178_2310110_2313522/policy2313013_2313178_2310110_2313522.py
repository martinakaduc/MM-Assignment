from policy import Policy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Policy2313013_2313178_2310110_2313522(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.use_policy  = None
        self.policy_id = policy_id

        if policy_id == 1:
            self.use_policy = Column_Generation()
        elif policy_id == 2:
            self.use_policy = REINFORCE(is_training=False)

    def get_action(self, observation, info):
        # Student code here
        return self.use_policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed



#####################################################################
# The class below is Column Generation for 2D Cutting Stock Problem #
# Author: Huynh Quoc Thang                                          #                        
# Create on: 30/11/2024                                             #
# Last update: 15/12/2024                                           #  
#####################################################################

class Column_Generation(Policy):
    def __init__(self):
        self.columns = []  # Lưu trữ các mẫu cắt
        self.duals = []  # Biến đối song song

    def getStock(self, stock):
        valid_rows = np.any(stock != -2, axis=1)
        valid_cols = np.any(stock != -2, axis=0)

        stock_w = np.sum(valid_rows)  # Chiều rộng khả dụng
        stock_h = np.sum(valid_cols)  # Chiều cao khả dụng

        return stock_w, stock_h

    def canPlace(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        stock_w, stock_h = stock.shape

        if pos_x + prod_w > stock_w or pos_y + prod_h > stock_h:
            return False

        return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        if not products or not stocks:
            return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}

        self.columns, self.duals = self._solve_master_problem(products)
        new_column = self._solve_subproblem(products, self.duals, stocks)

        if new_column:
            self.columns.append(new_column)

        action = self._select_best_action(products, stocks)
        return action

    def _solve_master_problem(self, products):
        duals = [1.0 for _ in products]  

        if not self.columns:
            self.columns = [[1 if i == j else 0 for i in range(len(products))] for j in range(len(products))]

        return self.columns, duals

    def _solve_subproblem(self, products, duals, stocks):
        """
        Giải bài toán phụ để tìm mẫu cắt mới.
        """
        best_pattern = None
        best_reduced_cost = 0

        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self.getStock(stock)

            for product_idx, product in enumerate(products):
                prod_w, prod_h = product["size"]
                if product["quantity"] > 0:
                    # Kiểm tra trường hợp xoay
                    possible_sizes = [(prod_w, prod_h), (prod_h, prod_w)] if prod_w != prod_h else [(prod_w, prod_h)]

                    for size in possible_sizes:
                        if size[0] <= stock_w and size[1] <= stock_h:
                            pattern = [1 if i == product_idx else 0 for i in range(len(products))]
                            reduced_cost = duals[product_idx] - 1

                            if reduced_cost < best_reduced_cost:
                                best_reduced_cost = reduced_cost
                                best_pattern = pattern

        return best_pattern if best_reduced_cost < 0 else None

    def _select_best_action(self, products, stocks):
        """
        Lựa chọn hành động tối ưu từ các mẫu hiện tại.
        """
        for stock_idx, stock in enumerate(stocks):
            stock_w, stock_h = self.getStock(stock)

            for product_idx, product in enumerate(products):
                prod_w, prod_h = product["size"]
                if product["quantity"] > 0:
                    # Kiểm tra cả hai khả năng sản phẩm xoay hoặc không xoay
                    possible_sizes = [(prod_w, prod_h), (prod_h, prod_w)] if prod_w != prod_h else [(prod_w, prod_h)]

                    for size in possible_sizes:
                        if size[0] <= stock_w and size[1] <= stock_h:
                            for x in range(stock_w - size[0] + 1):
                                for y in range(stock_h - size[1] + 1):
                                    if self.canPlace(stock, (x, y), size):
                                        return {"stock_idx": stock_idx, "size": list(size), "position": (x, y)}

        new_column = self._solve_subproblem(products, self.duals, stocks)

        if new_column:
            self.columns.append(new_column)
            return self._select_best_action(products, stocks)

        return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}  


##################################################################################################
# The class below is REINFORCE algorithms of Reinforcement Learning for 2D Cutting Stock Problem #
# Author: Ta Tien Tai                                                                            #                        
# Create on: 05/12/2024                                                                          #
# Last update: 15/12/2024                                                                        #  
##################################################################################################

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        logits = self.network(x)
        return torch.nn.functional.log_softmax(logits, dim=-1)

class REINFORCE(Policy):
    def __init__(self, is_training=True, model_path='student_submissions\s2313013_2313178_2310110_2313522\model.pth'):
        self.state_dim = 200
        self.action_dim = 100
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon = 0.999
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.entropy_weight = 0.01
        self.cnt_step = 0
        self.threshold_update = 64
        self.sum_baseline = 0
        self.is_training = is_training
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        
        if not is_training:
            self.load_model(model_path)
        
        self.stocks = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []  

    def load_model(self, model_path):
        try:
            self.policy_network.load_state_dict(torch.load(model_path))
            print(f"Model has been load from {model_path}")
        except FileNotFoundError:
            print(f"Cannot find weight file at {model_path}, start training with random weight.")
        
    def save_model(self, model_path):
        torch.save(self.policy_network.state_dict(), model_path)
        print(f"Model has been save at {model_path}")
    
    def train(self, env, episodes=100):
        print("Training begin...")
        observation, info = env.reset(seed=42)
        ep = 0
        while ep < episodes:
            action = self.get_action(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print("Episode: ", ep, " - ", info)
                observation, info = env.reset(seed=ep)
                ep += 1
        
        print("Training end")
        if self.is_training:
            self.save_model('model.pth')

    def _standardize(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        std = np.where(std == 0, 1, std)
        
        normalized_data = (data - mean) / std
        return normalized_data

    def _encode_observation(self, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        
        encode_stocks = []
        for stock in stocks:
            stock_array = np.array(stock)
            available = (stock_array == -1).sum()
            encode_stocks.append(available) 

        encode_features = []
        for prod in products:
            size = prod["size"]
            quantity = prod["quantity"]
            encode_features.append(size[0] * size[1] * quantity)
 
        encode_stocks = np.array(encode_stocks, dtype=np.float32)
        encode_features = np.array(encode_features, dtype=np.float32)
 
        encode_stocks = self._standardize(encode_stocks)
        encode_features = self._standardize(encode_features)

        encode_stocks = np.pad(encode_stocks, 
                                (0, max(0, 100 - len(encode_stocks))), 
                                mode='constant')[:100]
        
        encode_features = np.pad(encode_features, 
                                (0, max(0, 100 - len(encode_features))), 
                                mode='constant')[:100]
        
        encode_state = np.concatenate([encode_stocks, encode_features])
        
        return torch.FloatTensor(encode_state).to(self.device)

    def _calculate_reward(self, stock_idx, prod_size):
        cutted_stock = self.stocks[stock_idx]
        remain_space = np.sum(cutted_stock == -1)
        prod_square = prod_size[0] * prod_size[1]
        reward = prod_square / remain_space

        stock_w, stock_h = self._get_stock_size_(cutted_stock)
        watse_square = (prod_square - (stock_w * stock_h)) / (stock_w * stock_h)
        reward += watse_square

        return reward

    def get_action(self, observation, info):
        self.stocks = observation["stocks"]
        state = self._encode_observation(observation)
        
        if np.random.random() < self.epsilon and self.is_training:
            action_idx = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                log_probs = self.policy_network(state)
                probs = torch.exp(log_probs)
                action_idx = torch.multinomial(probs, 1).item()
        
        log_prob = self.policy_network(state)[action_idx]
        
        return self._decode_action(state, action_idx, log_prob, observation)

    def _store_transition(self, state, action, reward, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)

    def _decode_action(self, state, action_idx, log_prob, observation):
        stocks = observation["stocks"]
        products = observation["products"]

        remain_products = []
        for prod_idx, prod_info in enumerate(products):
            if prod_info["quantity"] > 0:
                remain_products.append((prod_idx, prod_info))
        
        if len(remain_products) == 0:
            return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}

        prod_idx, prod = remain_products[action_idx % len(remain_products)]
        prod_size = prod["size"]
        
        sorted_stocks = sorted(
            [(self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1], idx, stock) 
            for idx, stock in enumerate(stocks)], 
            reverse=True
        )
        
        for area, stock_idx, stock in sorted_stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size  
            
            if stock_w < prod_w or stock_h < prod_h:
                continue  

            available = np.sum(stock == -1)
            if prod_h * prod_w > available:
                continue
            
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    reverse_prod_size = (prod_size[1], prod_size[0])
                    if self._can_place_(stock, (x, y), prod_size):
                        
                        if(self.is_training):
                          reward = self._calculate_reward(stock_idx, prod_size)
                          self._store_transition(state, action_idx, reward, log_prob)
                          self.sum_baseline += reward

                          self.cnt_step += 1
                          if self.cnt_step % self.threshold_update == 0:
                              self.update_policy()
                        
                        return {
                            "stock_idx": stock_idx,  
                            "size": prod_size,
                            "position": (x, y)
                        }

        return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}

    def _reset_mem(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.log_probs[:]

    def update_policy(self):
        if len(self.rewards) == 0:
            return

        returns = []
        base_line = self.sum_baseline / self.cnt_step
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = np.array(returns, dtype=np.float32)
        returns = torch.tensor(returns, device=self.device)

        entropy = 0
        for log_prob in self.log_probs:
            entropy -= torch.exp(log_prob) * log_prob 

        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R * base_line)
        entropy_loss = self.entropy_weight * entropy  
        total_loss = torch.stack(policy_loss).sum() + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)  
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._reset_mem()