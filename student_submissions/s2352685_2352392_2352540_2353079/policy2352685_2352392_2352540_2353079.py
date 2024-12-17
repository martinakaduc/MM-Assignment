import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from policy import Policy

class DoubleDeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class Policy2352685_2352392_2352540_2353079(Policy):
    def __init__(self, policy_id=1, learning_rate=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        super().__init__()
        self.policy_id = policy_id  # Định nghĩa tham số policy_id
        self.state_dim = 1000  # Estimated state dimension
        self.action_dim = 10000  # Estimated action dimension
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Đối với policy_id = 1 (DDQN), khởi tạo Q-network
        self.q_network = DoubleDeepQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network = DoubleDeepQNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.SGD(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        self.memory = []
        self.batch_size = 128

    def _preprocess_state(self, observation):
        state = []
        for stock in observation['stocks']:
            stock_state = stock.flatten()
            state.extend(stock_state)

        for product in observation['products']:
            state.extend([product['quantity'], 
                          product['size'][0],  # Width
                          product['size'][1],  # Height
                          0 if 'rotation' not in product else product['rotation']
                          ])
        
        return torch.FloatTensor(state).to(self.device)

    def get_action(self, observation, info):
        if self.policy_id == 1:  # Policy for DDQN
            return self._ddqn_get_action(observation)
        elif self.policy_id == 2:  # Policy for FFD
            return self._ffd_get_action(observation)

    def _ddqn_get_action(self, observation):
        state = self._preprocess_state(observation)
        
        if random.random() < self.epsilon:
            # Exploration: Random action
            list_prods = observation["products"]
            for prod in list_prods:
                if prod["quantity"] > 0:
                    original_size = prod["size"]
                    rotated_size = np.array([prod["size"][1], prod["size"][0]])  # Rotated orientation
                    for i, stock in enumerate(observation["stocks"]):
                        for orientation_idx, current_size in enumerate([original_size, rotated_size]):
                            stock_w, stock_h = self._get_stock_size_(stock)
                            
                            if stock_w < current_size[0] or stock_h < current_size[1]:
                                continue

                            for x in range(stock_w - current_size[0] + 1):
                                for y in range(stock_h - current_size[1] + 1):
                                    if self._can_place_(stock, (x, y), current_size):
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
        
        rotation = placement_idx // 100
        placement_idx %= 100
        
        if rotation == 1:
            size = np.array([prod['size'][1], prod['size'][0]])  # Swap width and height
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

    def _ffd_get_action(self, observation):
        list_prods = sorted(
            observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True
        )
        stock_idx, pos_x, pos_y = -1, None, None
        prod_size = None

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                best_fit = None
                min_leftover_area = float("inf")

                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        leftover_area = (stock_w * stock_h) - (prod_w * prod_h)
                        if leftover_area < min_leftover_area:
                            pos_x, pos_y = self._find_position_(stock, prod_size)
                            if pos_x is not None and pos_y is not None:
                                best_fit = (i, pos_x, pos_y, prod_size)
                                min_leftover_area = leftover_area

                    if stock_w >= prod_h and stock_h >= prod_w:
                        leftover_area = (stock_w * stock_h) - (prod_h * prod_w)
                        if leftover_area < min_leftover_area:
                            pos_x, pos_y = self._find_position_(stock, prod_size[::-1])
                            if pos_x is not None and pos_y is not None:
                                best_fit = (i, pos_x, pos_y, prod_size[::-1])
                                min_leftover_area = leftover_area

                if best_fit:
                    stock_idx, pos_x, pos_y, prod_size = best_fit
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

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