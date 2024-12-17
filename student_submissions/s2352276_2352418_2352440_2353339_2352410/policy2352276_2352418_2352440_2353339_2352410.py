from policy import Policy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
class DoubleDeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)
class Policy2352276_2352418_2352440_2353339_2352410(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2" 
        
        self.policy_id = policy_id
        self.patterns = []  
        self.demands = [] 
        self.stock_size = (0, 0)  
        
        if policy_id == 1:
            pass  # You can add specific initialization for policy_id 1 here
        elif policy_id == 2:
            learning_rate=1e-3 
            gamma=0.99
            epsilon=1.0
            epsilon_decay=0.995,
            super().__init__()
            self.state_dim = 1000  # Estimated state dimension
            self.action_dim = 10000  # Estimated action dimension

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.q_network = DoubleDeepQNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network = DoubleDeepQNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())

            self.optimizer = optim.SGD(self.q_network.parameters(), lr=learning_rate)
            self.loss_fn = nn.MSELoss()

            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay

            self.memory = []
            self.batch_size = 256
            pass  # You can add specific initialization for policy_id 2 here
    def get_action(self, observation, info):
        if self.policy_id == 1:
            list_prods = list(observation["products"])
            list_prods.sort(key=lambda x: x["size"][0] * x["size"][1], reverse=True)

            sorted_stocks = sorted(
                enumerate(observation["stocks"]),
                key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],
                reverse=True
            )

            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    rotated_size = (prod_size[1], prod_size[0])
                    best_option = None
                    best_stock_area = -1

                    for i, stock in sorted_stocks:
                        stock_w, stock_h = self._get_stock_size_(stock)
                        orientations = [prod_size, rotated_size]
                        for orientation in orientations:
                            prod_w, prod_h = orientation
                            if stock_w >= prod_w and stock_h >= prod_h:
                                for y in range(stock_h - prod_h + 1):
                                    for x in range(stock_w - prod_w + 1):
                                        if self._can_place_(stock, (x, y), orientation):
                                            stock_area = stock_w * stock_h
                                            if stock_area > best_stock_area:
                                                best_stock_area = stock_area
                                                best_option = {"stock_idx": i, "size": (int(prod_w), int(prod_h)), "position": (x, y)}
                                            break
                                    if best_option:
                                        break
                            if best_option:
                                break
                        if best_option and best_stock_area != -1:
                            break

                    if best_option:
                        # Debug
                        # print(f"Selected product size: {best_option['size']}, stock index: {best_option['stock_idx']}, position: {best_option['position']}")
                        return best_option
        elif self.policy_id == 2:
            state = self._preprocess_state(observation)

            if random.random() < self.epsilon:
                # Exploration: Random action
                list_prods = observation["products"]
                for prod in list_prods:
                    if prod["quantity"] > 0:
                        # Original orientation [w,h]
                        original_size = prod["size"]
                        # Rotated orientation [h,w]
                        rotated_size = np.array([prod["size"][1], prod["size"][0]])

                        # Try both orientations
                        for i, stock in enumerate(observation["stocks"]):
                            for orientation_idx, current_size in enumerate([original_size, rotated_size]):
                                stock_w, stock_h = self._get_stock_size_(stock)

                                if stock_w < current_size[0] or stock_h < current_size[1]:
                                    continue

                                for x in range(stock_w - current_size[0] + 1):
                                    for y in range(stock_h - current_size[1] + 1):
                                        if self._can_place_(stock, (x, y), current_size):
                                            # print(f"Placing at stock {i}, position ({x}, {y}), size {current_size}, rotation {orientation_idx}")  # Debug line
                                            return {
                                                "stock_idx": i,
                                                "size": current_size,
                                                "rotation": orientation_idx,
                                                "position": (x, y)
                                            }

                return None

            # Exploitation: Use Q-network
            with torch.no_grad():
                q_values = self.q_network(state)
                action_idx = q_values.argmax().item()

            # Decode action_idx to stock, product, and placement
            stock_idx = action_idx // (len(observation['products']) * 200)
            action_idx %= len(observation['products']) * 200
            prod_idx = action_idx // 200
            placement_idx = action_idx % 200

            stock = observation['stocks'][stock_idx]
            prod = observation['products'][prod_idx]

            # Decode rotation and placement
            rotation = placement_idx // 100
            placement_idx %= 100

            # Get product size based on rotation
            if rotation == 1:
                size = np.array([prod['size'][1], prod['size'][0]])
            else:
                size = prod['size']

            placement = self._column_generation(stock, size, placement_idx, rotation)

            self.epsilon *= self.epsilon_decay

            return {
                "stock_idx": stock_idx,
                "size": size,
                "rotation": rotation,
                "position": placement
            }
    def _preprocess_state(self, observation):
        stocks = np.concatenate([stock.flatten() for stock in observation['stocks']])
        products = np.concatenate([
            np.array([p['quantity'], p['size'][0], p['size'][1], p.get('rotation', 0)])
            for p in observation['products']
        ])
        state = np.concatenate([stocks, products])
        return torch.FloatTensor(state).to(self.device)
    def _column_generation(self, stock, prod_size, placement_idx, rotation):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        
        x = (placement_idx // 10) * (stock_w - prod_w) // 10
        y = (placement_idx % 10) * (stock_h - prod_h) // 10
        
        return (x, y)
    
    def learn(self, state, action, reward, next_state, done):
        # Store experience
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states)
        next_states = torch.stack(next_states)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q-values
        current_q_values = self.q_network(states)
        next_q_values = self.target_network(next_states)
        
        # Double DQN target
        next_actions = self.q_network(next_states).argmax(dim=1)
        max_next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        targets = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        # Compute loss
        predicted_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.loss_fn(predicted_q_values, targets.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update of target network
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(0.005 * param.data + 0.995 * target_param.data)