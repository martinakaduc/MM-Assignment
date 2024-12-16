import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import math
import time
from abc import abstractmethod
"""
TO_DO LIST:
- Create a self-observation. We just need to encode first when the environment reset.
Modify the self-observation easily with access by stock_index and product_index.
- Enhance the masking mechanism. We introduce a big negative reward when choose an invalid combination.
- The reward will by locally in 1 stocks.
If we reuse stocks, it will be a positive reward and otherwise.
When placing stock, we calculate the compactness, and retrieve a postive bonus base on compactness.

"""
class Policy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)


class ProductEncoder(nn.Module):
    """Encodes the input sequence using 1D convolution."""
    def __init__(self, input_size, hidden_size, device):
        super(ProductEncoder, self).__init__()
        self.device = device
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)
    """
    Parameter: x - A torch tensor with shape (2*max_num_product, 3)
    Return: output - A torch tensor with shape (2*max_num_product, hidden_size)
    Usage:
        Extract feature from tuple 3: width - height - quantity of each product and its rotation
    """
    def forward(self, x):
        # print("Product: ", x.shape)
        x = x.to(self.device)
        x = x.unsqueeze(0).permute(0,2,1)
        # print("Product: ", x.shape)
        output = self.conv(x)
        # print("Product: ", output.shape)
        output = output.squeeze(0).permute(1, 0)
        # print("Product: ", output.shape)
        return output


class StockEncoder(nn.Module):
    """Encodes the states using 2D Convolution."""
    def __init__(self, hidden_size, map_size, device):
        super(StockEncoder, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(1, hidden_size // 4, stride=2, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_size // 4, hidden_size // 2, stride=2, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=(math.ceil(map_size[0] / 4), math.ceil(map_size[1] / 4)))

    """
    Parameter: input - A tensor with shape (num_stock, max_w, max_h)
    Return: output - A tensor with shape (num_stock, hidden_size)
    Usage:
        Extract features from stock.
    """
    def forward(self, input):
        # Ensure the input tensor is compatible with Conv2D
        # output = torch.where(input != -1, torch.tensor(0.0, dtype=torch.float32, device=input.device), torch.tensor(1.0, dtype=torch.float32, device=input.device))
        input = input.to(self.device)
        output = input.unsqueeze(1)  # Add channel dimension (N, C=1, H, W)
        # print("Stock: ", output.shape)
        output = F.leaky_relu(self.conv1(output))
        # print("Stock: ", output.shape)
        output = F.leaky_relu(self.conv2(output))
        # print("Stock: ", output.shape)
        output = self.conv3(output).squeeze(-1).squeeze(-1)  # Remove height and width dimensions
        # print("Stock: ", output.shape)
        return output


class DQN(nn.Module):
    """
    Parameter: x - A 1D tensor of concatenated product's feature and stock's feature
    Return: output - A tensor with shape (2*num_stock*max_product_type) to compute the compatible score of stock - product

    """

    """
    Parameter: x - A 1D tensor of 1 stock - 1 product
    Return: output - A tensor with shape (1,) to estimate the compatible score of stock - product.
    """
    def __init__(self, input_size, hidden_size, output_size, device):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, False)
        self.fc3 = nn.Linear(hidden_size, output_size, False)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.shape)
        return x


class Masking():
    """
    Provides a masking mechanism to ensure only valid actions are taken.
    """
    def __init__(self, num_actions, num_products, device):
        self.num_actions = num_actions
        self.num_products = num_products
        self.num_sheets = num_actions // (2*num_products)
        self.device = device
        self.mask = torch.zeros(num_actions, dtype=torch.float32).to(self.device)
        self.count = 0

    def update_mask(self, sheet_index, product_index, rotate, attribute):
        """
        Update the mask to block invalid actions based on attributes.

        Args:
            sheet_index (int): Index of the selected stock sheet.
            product_index (int): Index of the selected product.
            rotate (int): 0 for no rotation, 1 for rotation.
            attribute (str): The error type ("Product_Out_of_range", "Product_Error", "Stock_Error", "Complete").
        """
        invalid_actions = []

        if attribute == "Product_Out_of_range":
            # Mask all actions involving products with indices >= product_index
            for p_idx in range(product_index, self.num_products):
                invalid_actions.extend([
                    sheet_index * (2 * self.num_products) + p_idx * 2,  # Without rotation
                    sheet_index * (2 * self.num_products) + p_idx * 2 + 1  # With rotation
                ])

        elif attribute == "Product_Error":
            # Mask only actions involving the specific product
            invalid_actions.extend([
                sheet_index * (2 * self.num_products) + product_index * 2,  # Without rotation
                sheet_index * (2 * self.num_products) + product_index * 2 + 1  # With rotation
            ])

        elif attribute == "Stock_Error":
            # Incrementally block actions in the same stock to refine search
            if self.count < 10:  # Allow up to 10 incremental masking steps
                invalid_actions.append(
                    sheet_index * (2 * self.num_products) + product_index * 2 + rotate
                )
                self.count += 1
            else:
                # If the maximum attempts for this stock are reached, mask all actions on this stock
                for p_idx in range(self.num_products):
                    invalid_actions.extend([
                        sheet_index * (2 * self.num_products) + p_idx * 2,  # Without rotation
                        sheet_index * (2 * self.num_products) + p_idx * 2 + 1  # With rotation
                    ])
                self.count = 0  # Reset the counter for the next stock

        elif attribute == "Complete":
            # Reduce the likelihood of choosing this stock for the next step
            for p_idx in range(self.num_products):
                self.mask[sheet_index * (2 * self.num_products) + p_idx * 2] -= 0.5
                self.mask[sheet_index * (2 * self.num_products) + p_idx * 2 + 1] -= 0.5

        # Apply the invalid actions to the mask
        for action in invalid_actions:
            self.mask[action] = 1

        # Debugging: Ensure that at least one action remains valid
        if torch.all(self.mask == 1):
            print("Warning: All actions are masked. Resetting to prevent invalid state.")
            self.mask.fill_(0)  # Clear the mask if everything is invalid, allowing fallback exploration

    def reset_mask(self):
        """Reset the mask to allow all actions."""
        self.mask = torch.zeros_like(self.mask).to(self.device)

    def apply_mask(self, logits):
        """
        Apply the mask to the logits to ensure invalid actions are not selected.

        :param logits: Output from the model before applying softmax.
        :return: Masked logits.
        """
        return logits + (self.mask * -1e6)

    def reset_mask(self):
        """Reset the mask to allow all actions."""
        self.mask = torch.zeros_like(self.mask)

    def apply_mask(self, logits):
        """
        Apply the mask to the logits to ensure invalid actions are not selected.

        :param logits: Output from the model before applying softmax.
        :return: Masked logits.
        """
        return logits + (self.mask * -1e9)

import torch
import numpy as np


class EnvironmentSimulate:
    def __init__(self, num_stock, num_product, max_w, max_h):
        """
        Initialize the environment simulation.

        Args:
            num_stock (int): Number of stocks.
            num_product (int): Number of products.
            max_w (int): Maximum width of a stock.
            max_h (int): Maximum height of a stock.
        """
        self.num_stock = num_stock
        self.num_product = num_product
        self.max_w = max_w
        self.max_h = max_h
        self.stock_features = torch.zeros((num_stock, max_w, max_h), dtype=torch.float32)
        self.product_features = torch.zeros((2 * num_product, 3), dtype=torch.float32)

    def get(self):
        """
        Get the current stock and product features.

        Returns:
            tuple: (stock_features, product_features)
        """
        return self.stock_features, self.product_features

    def get_tensor(self):
        """
        Get a flattened tensor of stock and product features.

        Returns:
            torch.Tensor: Concatenated tensor of stock and product features.
        """
        return torch.cat((self.stock_features.flatten(), self.product_features.flatten()))

    def reset(self, observation):
        """
        Reset the environment simulation based on the initial observation.

        Args:
            observation (dict): Initial environment observation containing:
                - "stocks": A list of 2D arrays representing stocks.
                - "products": A list of dictionaries with size and quantity.

        Returns:
            tuple: Updated (stock_features, product_features).
        """
        if "stocks" not in observation or "products" not in observation:
            raise ValueError("Observation must contain 'stocks' and 'products' keys.")

        # Process stocks
        stock_data = np.array(observation["stocks"], dtype=np.float32)
        self.stock_features = torch.tensor(stock_data)
        self.stock_features = torch.where(self.stock_features == -2, 0.0, 1.0)

        # Process products
        product_features = []
        for product in observation["products"]:
            width, height = product.get("size", (0, 0))
            quantity = product.get("quantity", 0)
            product_features.append([width, height, quantity])  # Original orientation
            product_features.append([height, width, quantity])  # Rotated orientation

        # Fill remaining slots with placeholders
        while len(product_features) < 2 * self.num_product:
            product_features.append([0, 0, 0])

        self.product_features = torch.tensor(product_features, dtype=torch.float32)

        return self.stock_features, self.product_features


    def update(self, action):
        if action is None:
            return
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        prod_w, prod_h = size
        x, y = position
        product_idx = None
        for i, product in enumerate(self.product_features):
            width, height, quantity = product[0], product[1], product[2]
            if quantity == 0:
                continue
            if width == prod_w and height == prod_h:
                product_idx = i
                break
        if product_idx is not None:
            if torch.all(self.stock_features[stock_idx][x : x + prod_w, y : y + prod_h] == 1):
                self.stock_features[stock_idx][x : x + prod_w, y : y + prod_h] = 0
                self.product_features[product_idx, 2] -= 1
                self.product_features[product_idx ^ 1, 2] -= 1
        return self.stock_features, self.product_features


class PPO(nn.Module, Policy):
    def __init__(self, num_stock, max_w, max_h, num_product, hidden_size, mode : str, device,
                 lr=3e-4, gamma=0.99, clip_epsilon=0.2, value_loss_coef=0.5, entropy_coef=0.01):
        nn.Module.__init__(self)
        Policy.__init__(self)

        #Attribute
        self.num_product = num_product
        self.num_stock = num_stock
        self.max_w = max_w
        self.max_h = max_h
        self.mode = mode
        self.device = device

        #Support Classes
        self.observation = EnvironmentSimulate(num_stock, num_product, max_w, max_h)
        self.policy_net = DQN(num_stock*128 + 2*num_product*32, hidden_size, 2*num_stock*num_product, device).to(self.device)
        self.value_net = DQN(num_stock*128 + 2*num_product*32, hidden_size, 1, device).to(self.device)
        self.product_encoder = ProductEncoder(3, 32, device).to(self.device)
        self.stock_encoder = StockEncoder(128, [max_w, max_h], device).to(self.device)
        self.masking = Masking(2*num_stock*num_product, num_product, device)

        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)

        # Hyperparameters for PPO
        self.lr = lr
        self.gamma = gamma  # Discount factor
        self.clip_epsilon = clip_epsilon  # PPO clip epsilon for objective function
        self.value_loss_coef = value_loss_coef  # Coefficient for value loss
        self.entropy_coef = entropy_coef  # Coefficient for entropy loss


    def encode(self):
        stock_tensor, product_tensor = self.observation.get()
        stock_tensor = self.stock_encoder(stock_tensor)
        product_tensor = self.product_encoder(product_tensor)
        act_input = torch.cat((torch.flatten(stock_tensor), torch.flatten(product_tensor)))
        return act_input

    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        return np.all(stock[x:x + w, y:y + h] == -1)

    def place_product(self, observation, sheet_index, product_index, rotate):
        stock = observation["stocks"][sheet_index]
        if product_index >= len(observation["products"]):
            attribute = "Product_Out_of_range"
            return None, attribute

        prod_info = observation["products"][product_index]
        if observation["products"][product_index]["quantity"] == 0:
            attribute = "Product_Error"
            return None, attribute

        prod_w, prod_h = prod_info["size"]
        if rotate == 1:
            prod_w, prod_h = prod_h, prod_w

        max_w, max_h = stock.shape
        best_pos_x, best_pos_y = None, None
        best_distance = float('inf')

        for pos_x in range(max_w - prod_w + 1):
            for pos_y in range(max_h - prod_h + 1):
                if stock[pos_x][pos_y] != -1:
                    continue
                if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                    distance = (pos_x + prod_w/2)**2 + (pos_y + prod_h/2)**2
                    if distance < best_distance:
                        best_pos_x = pos_x
                        best_pos_y = pos_y
                        best_distance = distance

        if best_pos_x is None:
            attribute = "Stock_Error"
            return None, attribute
        
        attribute = "Complete"
        return {
            "stock_idx": sheet_index,
            "size": [prod_w, prod_h],
            "position": [best_pos_x, best_pos_y],
        }, attribute

    def inner_get_action(self, observation, info):
        """
        Given the current state, get the action from the policy network.
        Chuyển về chọn 1 lần 1 action.
        Trong lúc train thì không bật masking lên
        Trong lúc test thì bật mask
        """
        act_input = self.encode()  # Get the input features
        logits = self.policy_net(torch.softmax(act_input, dim=-1))  # Get the logits (raw action probabilities)
        dist_before_masked = torch.distributions.Categorical(logits=logits)
        logits = self.masking.apply_mask(logits)
        dist = torch.distributions.Categorical(logits=logits)
        if self.mode == "Train":
            action = dist.sample()  # Sample an action from the distribution
        elif self.mode == "Test":
            action = torch.argmax(logits)

        sheet_index = action // (2 * self.num_product)
        product_index = (action % (2 * self.num_product)) // 2
        rotate = action % 2  # 0 for no rotation, 1 for rotation
        placed, attribute = self.place_product(observation, sheet_index, product_index, rotate)
        self.observation.update(placed)
        # If placement fails, return a default invalid action in test mode
        self.masking.update_mask(sheet_index, product_index, rotate, attribute)
        if placed is None:
            return {
                "stock_idx": 1,
                "size" : [1000,1000],
                "position": [-1, -1]
            }, action, dist.log_prob(action), dist.entropy
        return placed, action, dist_before_masked.log_prob(action), dist_before_masked.entropy()

    ##Method for calling from main.py##
    def get_action(self, observation, info):
        placed, _, _, _ = self.inner_get_action(observation,info)
        while placed["size"] == [1000,1000]:
            placed, _, _, _ = self.inner_get_action(observation,info)
        return placed
    
    def model_reset(self, observation):
        self.masking.reset_mask()
        self.observation.reset(observation)
    def calculate_reward(self, observation, action, info):
        """
        Calculate reward based on area usage, compactness, and empty space.
        """
        stock = observation["stocks"][action["stock_idx"]]
        product_width, product_height = action["size"]

        # Kiểm tra chiều rộng sản phẩm, nếu không hợp lệ trả về 0
        if product_width ==  1000 or product_height == 1000:
            return 0  # Nếu có vấn đề với kích thước sản phẩm, trả về phần thưởng 0

        area_total = stock.shape[0] * stock.shape[1] - np.sum(stock == -2)

        # Tính diện tích đã sử dụng
        area_used = np.sum(stock >= 0)  # Kiểm tra các ô không bị trống (hoặc không phải giá trị mặc định)
        area_utilization = area_used / area_total  # Tỷ lệ sử dụng diện tích của stock

        reward_area_utilization = area_utilization  # Khuyến khích sử dụng diện tích hiệu quả

        # Tổng hợp phần thưởng
        reward = 1 + reward_area_utilization - info['trim_loss']
        return reward 

    def compute_returns_and_advantages(self, rewards, values, dones, last_value):
        advantages = []
        returns = []
        last_advantage = 0.0
        last_return = last_value
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + (1 - dones[t]) * self.gamma * last_value - values[t]
                last_return = rewards[t] + (1 - dones[t]) * self.gamma * last_value
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                last_return = rewards[t] + (1 - dones[t]) * self.gamma * last_return

            advantage = delta + self.gamma * 0.9 * (1 - dones[t]) * last_advantage
            last_advantage = advantage
            advantages.insert(0, advantage)
            returns.insert(0, last_return)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        return returns, advantages



    def update(self, states, actions, log_probs_old, returns, advantages):
        """Update the policy and value network using PPO."""
        states = torch.tensor(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        log_probs_old = torch.tensor(log_probs_old).to(self.device)
        returns = returns.clone().detach().to(self.device)
        advantages = advantages.clone().detach().to(self.device)

        # Get new action probabilities and values
        logits = self.policy_net(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs_new = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.value_net(states).squeeze()

        # Compute the ratio of new and old probabilities
        ratio = torch.exp(log_probs_new - log_probs_old)

        # Clipped objective for PPO
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value function loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # Total loss: Policy loss + Value loss + Entropy loss
        torch.autograd.set_detect_anomaly(True)
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Perform the optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def evaluate(self, states, actions):
        """
        Evaluate the policy network given a batch of states and actions.

        :param states: A batch of states
        :param actions: A batch of actions
        :return: Log probabilities, entropy, and state value estimates
        """
        logits = self.policy_net(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        values = self.value_net(states).squeeze()
        return log_probs, entropy, values


