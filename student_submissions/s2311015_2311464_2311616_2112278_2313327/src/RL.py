import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from policy import Policy


class ActorNetwork(nn.Module):
    def __init__(self, obs_shape, action_shape):
        super().__init__()
        # Modify network to output more detailed action probabilities
        self.fc = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_shape),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

class CriticNetwork(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        # Enhanced critic network
        self.fc = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.fc(x)

class ActorCriticPolicy(Policy, nn.Module):
    def __init__(self, obs_shape=1000175, action_shape=5000, lr=1e-3, gamma=0.99, entropy_coeff=0.01, filled_ratio=0):
        super().__init__()
        nn.Module.__init__(self)  # Initialize nn.Module
        self.gamma = gamma  # discount factor
        self.entropy_coeff = entropy_coeff
        self.actor = ActorNetwork(obs_shape, action_shape)
        self.critic = CriticNetwork(obs_shape)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)
        self.filled_ratio = filled_ratio
        # Track learning history for more intelligent action selection
        self.action_history = []
        self.reward_history = []
        self.learning_rate = lr
    
    def _preprocess_observation(self, observation, max_stocks=100, max_products=25):
        stocks = observation['stocks']
        products = observation['products']
        # padded_stocks = [np.pad(stock, ((0, max_stocks - stock.shape[0]),
        #                                 (0, max_stocks - stock.shape[1])),
        #                         mode='constant', constant_values=-2).flatten()
        #                  for stock in stocks[:max_stocks]]
        # while len(padded_stocks) < max_stocks:
        #     padded_stocks.append(np.full((max_stocks * max_stocks), -2))
        padded_stocks = []
        for stock in stocks[:max_stocks]:
            padded_stocks.append(stock.flatten())
            area = self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]
            padded_stocks.append(np.array([area], dtype=np.int64)) 
        stocks_tensor = np.concatenate(padded_stocks)

        padded_products = [[int(product["size"][0]), int(product["size"][1]), product["quantity"]]
                           if product["quantity"] > 0 else [0, 0, -1]
                           for product in products[:max_products]]
        while len(padded_products) < max_products:
            padded_products.append([0, 0, -1])
        products_tensor = np.concatenate(padded_products)

        obs_tensor = np.concatenate([stocks_tensor, products_tensor])
        return torch.tensor(obs_tensor, dtype=torch.float32)
    def get_action(self, observation, info):
        obs_tensor = self._preprocess_observation(observation)
        
        with torch.no_grad():
            action_probs = self.actor(obs_tensor)
        
        # Enhanced action selection considering historical performance
        action_candidates = torch.argsort(action_probs, descending=True)
        
        best_action = None
        best_score = float('-inf')
        
        for action_idx in action_candidates:
            # Decode action with more comprehensive analysis
            decoded_action = self._decode_action(action_idx.item(), observation)
            
            if decoded_action is None:
                continue
            
            # Score action based on multiple criteria
            self.filled_ratio = info["filled_ratio"]
            action_score = self._score_action(decoded_action, observation)
            
            if action_score > best_score:
                best_score = action_score
                best_action = decoded_action
        
        return best_action

    def _decode_action(self, action, observation):
        prods = observation['products']
        l = len(prods)
        stock_index = action // (l * 2)
        prod_index = (action % (l * 2)) // 2
        rotate = action % 2

        if stock_index >= 100 or prod_index >= l:
            return None
        
        stock = observation['stocks'][stock_index]
        prod = prods[prod_index]
        stock_w, stock_h = self._get_stock_size_(stock)
        if (rotate == 1):
            prod_h, prod_w = prod['size']
        else:
            prod_w, prod_h = prod['size']
        if prod['quantity'] == 0 or stock_w < prod_w or stock_h < prod_h:
            return None
        
        for i in range(stock_w - prod_w + 1):
            for j in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (i, j), (prod_w, prod_h)):
                    return {'stock_idx': stock_index, 'position': (i, j), 'size': (prod_w, prod_h)}
        return None
    
    def _encode_action(self, action, orig_state):
        size = action['size']
        prods = orig_state['products']
        prod_idx = -1
        for i in range(len(prods)):
            if (prods[i]['size'][0] == size[0] and prods[i]['size'][1] == size[1]) or (prods[i]['size'][0] == size[1] and prods[i]['size'][1] == size[0]):
                prod_idx = i
                break
        rotation_offset = 0 if prods[prod_idx]['size'][0] == size[0] else 1
        return (action['stock_idx'] * len(prods) * 2) + (prod_idx * 2) + rotation_offset
    def _score_action(self, action, observation):
        stock_idx = action['stock_idx']
        stock = observation['stocks'][stock_idx]
        prod_w, prod_h = action['size']

        # Get stock dimensions and current state
        stock_w, stock_h = self._get_stock_size_(stock)
        total_stock_area = stock_w * stock_h
        stock_state = np.sum(stock == -1)  # Used area
        prod_area = prod_w * prod_h

        # Priority 1: Stock area score (maximize larger unused stock area)
        used_ratio = stock_state / total_stock_area
        stock_area_score = total_stock_area * (1 - used_ratio) + total_stock_area + prod_area

        # Priority 2: Chance of fulfilling current stock_state with products
        fulfillment_score = 0
        if stock_state % prod_area == 0:
            fulfillment_score += 5  # Bonus for perfect fit

        # Priority 3: Complete fill bonus
        complete_fill_bonus = 0
        if stock_state + prod_area == total_stock_area:
            complete_fill_bonus += 50  # Significant bonus for completely filling the stock

        # Additional position efficiency
        pos_x, pos_y = action['position']

        # Check if placement is possible
        if not self._can_place_(stock, action['position'], (prod_w, prod_h)):
            return float('-inf')  # Impossible placement

        # Historical reward consideration
        historical_reward = self._get_historical_reward(action)
        position_efficiency = self._evaluate_position_efficiency(stock, action['position'], action['size'])

        # Combine scores with carefully tuned weights
        total_score = (
            stock_area_score +      # Prioritize larger, less utilized stocks
            20 * fulfillment_score +     # Prefer actions that can fully utilize space
            10 * position_efficiency +   # Prefer centered placements
            10 * historical_reward +     # Consider past performance
            10 * complete_fill_bonus            # Significant bonus for complete fill
        )
        total_score -= self.filled_ratio * 10000

        return total_score
    def _calculate_reward(self, action, observation):
        stock_idx = action['stock_idx']
        stock = observation['stocks'][stock_idx]
        prod_w, prod_h = action['size']
        
        # Calculate space utilization (maximize this)
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Penalize unfit placements
        if not self._can_place_(stock, action['position'], (prod_w, prod_h)):
            return -1  # penalize bad placements
        stock_state = np.sum(stock == -1)
        prod_state = prod_w * prod_h
        # Positive reward for good placement
        reward = -0.1 * (prod_w * prod_h) / (stock_w * stock_h) + -1 * (stock_state - prod_state) - self.filled_ratio * 10000  # higher reward for better utilization
        
        return reward
    def _get_historical_reward(self, action):
        """
        Retrieve historical performance for similar actions
        """
        similar_actions = [
            reward for (hist_action, reward) in zip(self.action_history, self.reward_history)
            if (hist_action['stock_idx'] == action['stock_idx'] and hist_action['size'] == action['size'])
        ]
        
        return np.mean(similar_actions) if similar_actions else 0

    def _evaluate_position_efficiency(self, stock, position, size):
        """
        Evaluate the efficiency of product placement position
        """
        x, y = position
        prod_w, prod_h = size
        
        # Check surrounding areas and potential future placements
        surrounding_efficiency = self._check_surrounding_areas(stock, position, size)
        
        # Proximity to stock edges can be beneficial
        edge_proximity_score = self._calculate_edge_proximity(stock, position, size)
        
        return (surrounding_efficiency + edge_proximity_score) / 2

    def _check_surrounding_areas(self, stock, position, size):
        """
        Analyze potential for future placements around current position
        """
        x, y = position
        prod_w, prod_h = size
        remaining_space = 0
        
        # Check surrounding areas
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                if self._can_place_(stock, (x + dx, y + dy), size):
                    remaining_space += 1
        
        return remaining_space / 4  # Normalize

    def _calculate_edge_proximity(self, stock, position, size):
        x, y = position
        prod_w, prod_h = size
        stock_w, stock_h = self._get_stock_size_(stock)

        # Calculate distances to edges
        left_distance = x
        right_distance = stock_w - (x + prod_w)
        top_distance = y
        bottom_distance = stock_h - (y + prod_h)

        # Prefer placement close to any edge
        edge_proximity_score = min(left_distance, right_distance, top_distance, bottom_distance)
        return edge_proximity_score

    def update(self, transitions, orig_state):
        states, actions, rewards, next_states, dones = transitions
        
        # Update rewards based on the new reward function
        updated_rewards = []
        for idx, action in enumerate(actions):
            reward = self._calculate_reward(action, orig_state)
            updated_rewards.append(reward)
        
        rewards = torch.tensor(updated_rewards, dtype=torch.float32)
        states = torch.stack([self._preprocess_observation(state) for state in states])
        next_states = torch.stack([self._preprocess_observation(next_state) for next_state in next_states])

        encoded_actions = torch.tensor([self._encode_action(action, orig_state) for action in actions], dtype=torch.int64)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute values and advantages
        values = self.critic(states).view(-1)
        next_values = self.critic(next_states).view(-1)
        targets = (rewards + self.gamma * next_values * (1 - dones)).view(-1)
        advantages = (targets - values).view(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Actor update with policy gradient
        action_probs = self.actor(states)
        log_probs = torch.log(action_probs.gather(1, encoded_actions.unsqueeze(1)).squeeze())
        entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1).mean()
        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coeff * entropy
        
        self.optimizer_actor.zero_grad()
        loss = actor_loss.backward()
        print(loss)
        self.optimizer_actor.step()

        # Critic update
        critic_loss = nn.MSELoss()(values, targets.detach())
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update action and reward history for future decision-making
        self.action_history.extend(actions)
        self.reward_history.extend(updated_rewards)
        
        # Limit history size to prevent memory issues
        max_history_size = 1000
        self.action_history = self.action_history[-max_history_size:]
        self.reward_history = self.reward_history[-max_history_size:]