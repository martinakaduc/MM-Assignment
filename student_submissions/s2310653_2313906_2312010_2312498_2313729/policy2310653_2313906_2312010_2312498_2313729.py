from policy import Policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np


class Policy2310653_2313906_2312010_2312498_2313729(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = FFDHPolicy()
        elif policy_id == 2:
            self.policy = RLPolicy()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed

class FFDHPolicy(Policy):
    def __init__(self):
        pass


    def get_action(self, observation, info):
        return self._apply_ffdh_area_strategy(observation)


    def _apply_ffdh_area_strategy(self, observation):
        """
        FFDH: Sắp xếp sản phẩm giảm dần theo diện tích và tìm vị trí đặt đầu tiên.
        """

        # Sắp xếp sản phẩm theo diện tích giảm dần
        products = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True,
        )

        for product in products:
            if product["quantity"] > 0:
                product_size = product["size"]

                # Duyệt qua từng tấm và kiểm tra xoay hoặc không xoay
                for stock_idx, stock in enumerate(observation["stocks"]):
                    for rotation in [False, True]:
                        size = product_size[::-1] if rotation else product_size
                        position = self._find_best_fit(stock, size)
                        if position:
                            return self._create_action(stock_idx, size, position, rotation)

                # Không tìm được vị trí nào trên các tấm hiện có
       
                return None  # Không có hành động khả thi

    def _find_best_fit(self, stock, product_size):
        """
        Tìm vị trí tốt nhất để đặt sản phẩm lên tấm.
        """
        stock_width, stock_height = self._get_stock_size_(stock)
        product_width, product_height = product_size
        best_position = None
        min_wasted_area = float("inf")

        for x in range(stock_width - product_width + 1):
            for y in range(stock_height - product_height + 1):
                if self._can_place_(stock, (x, y), product_size):
                    wasted_area = self._calculate_wasted_area(stock, (x, y), product_size)
                    if wasted_area < min_wasted_area:
                        min_wasted_area = wasted_area
                        best_position = (x, y)

        return best_position

    def _calculate_wasted_area(self, stock, position, product_size):
        """
        Tính diện tích trống còn lại trên tấm nếu đặt sản phẩm tại vị trí.
        """
        x, y = position
        product_width, product_height = product_size
        wasted_area = 0

        for i in range(x, x + product_width):
            for j in range(y, y + product_height):
                if stock[i, j] == -1:
                    wasted_area += 1

        return wasted_area

    # def _can_place_(self, stock, position, product_size):
    #     """
    #     Kiểm tra xem sản phẩm có thể đặt vào vị trí trên tấm hay không.
    #     """
    #     x, y = position
    #     product_width, product_height = product_size
    #     stock_width, stock_height = self._get_stock_size_(stock)

    #     if x + product_width > stock_width or y + product_height > stock_height:
    #         return False

    #     for i in range(product_width):
    #         for j in range(product_height):
    #             if stock[x + i, y + j] != -1:
    #                 return False
    #     return True
    

    def _can_place_(self, stock, position, product_size):
        """
        Kiểm tra xem sản phẩm có thể đặt vào vị trí trên tấm hay không.
        """
        x, y = position
        product_width, product_height = product_size
        stock_width, stock_height = self._get_stock_size_(stock)

        # Kiểm tra nếu sản phẩm vượt ra khỏi kích thước tấm
        if x + product_width > stock_width or y + product_height > stock_height:
            return False

        # Kiểm tra các ô trong vùng đặt sản phẩm
        for i in range(product_width):
            for j in range(product_height):
                # Đảm bảo các chỉ số nằm trong phạm vi hợp lệ
                if x + i >= stock_height or y + j >= stock_width or stock[x + i, y + j] != -1:
                    return False
        return True


    def _get_stock_size_(self, stock):
        """
        Lấy kích thước của tấm hiện tại.
        """
        return stock.shape[1], stock.shape[0]

    def _create_action(self, stock_idx, size, position, rotated):
        """
        Tạo hành động để đặt sản phẩm lên tấm.
        """
        return {
            "stock_idx": stock_idx,
            "size": size,
            "position": position,
            "rotated": rotated,
        }

# Sap xep theo chieu cao (Tot hon)

# {'filled_ratio': 0.19, 'trim_loss': 0.3587157879908043}
# {'filled_ratio': 0.08, 'trim_loss': 0.2233059871750764}
# {'filled_ratio': 0.18, 'trim_loss': 0.2368002432245921}
# {'filled_ratio': 0.1, 'trim_loss': 0.14625770544403788}
# {'filled_ratio': 0.34, 'trim_loss': 0.1357619397680044}

# Sap xep theo dien tich ( Tot hon nhieu)
# {'filled_ratio': 0.19, 'trim_loss': 0.3553882983059738}
# {'filled_ratio': 0.08, 'trim_loss': 0.21290169993042757}
# {'filled_ratio': 0.17, 'trim_loss': 0.19578219132563987}
# {'filled_ratio': 0.09, 'trim_loss': 0.051662039804688656}
# {'filled_ratio': 0.33, 'trim_loss': 0.10832707802167041}

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

class PPOAgent:
    def __init__(self, input_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, K_epochs=4):
        self.policy = PolicyNetwork(input_dim, action_dim)
        self.policy_old = PolicyNetwork(input_dim, action_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.value_network = ValueNetwork(input_dim)
        self.params = list(self.policy.parameters()) + list(self.value_network.parameters())
        self.optimizer = optim.Adam(self.params, lr=lr, eps=1e-5)
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

    def select_action(self, state, info):
        state = torch.FloatTensor(state)
        logits = self.policy_old(state)
        dist = Categorical(logits=logits)
        # action = dist.sample()
        # return action.item(), dist.log_prob(action).item()
        return dist

    def update(self, memory):
        # Convert lists to tensors
        old_states = torch.stack(memory.states, dim=0)
        old_actions = torch.LongTensor(memory.actions)
        old_logprobs = torch.FloatTensor(memory.logprobs)
        with torch.no_grad():
            old_values = self.value_network(old_states).squeeze()
        rewards = torch.FloatTensor(memory.rewards)
        is_terminals = torch.FloatTensor(memory.is_terminals)

        # Compute discounted rewards
        discounted_rewards = []
        G = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                G = 0
            G = reward + self.gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)

        # PPO update
        for _ in range(self.K_epochs):
            # Get current policy log probabilities
            logits = self.policy(old_states)
            if torch.isnan(logits).any():
                print(old_states)
                print(logits)
                raise ValueError("Logits contain NaN")
            dist = Categorical(logits=logits)
            # dist = F.log_softmax(logits, dim=-1)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()

            # Get state values
            state_values = self.value_network(old_states).squeeze()

            # Compute advantages
            advantages = discounted_rewards - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # Compute ratios
            ratios = torch.exp(logprobs - old_logprobs)

            # Compute surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss_original = (state_values - discounted_rewards).pow(2)
            # Clipped value predictions
            clipped_values = old_values.detach() + torch.clamp(state_values - old_values.detach(), -self.eps_clip, self.eps_clip)
            value_loss_clipped = (clipped_values - discounted_rewards).pow(2)
            # Use the minimum of the two
            value_loss = torch.max(value_loss_original, value_loss_clipped).mean()

            # Compute total loss
            loss = policy_loss + 0.5 * value_loss - 0.01 * dist_entropy.mean()

            # Update policy and value network
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.params, 0.5)
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
    
    def save_model(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=True)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class RLPolicy(Policy):
    def __init__(self):
        # Student code here
        self.memory = Memory()
        self.agent = PPOAgent(input_dim=4, action_dim=2) # placeholder
        self.loop_count = 0
        self.update = True

    def get_action(self, observation, info):
        # Student code here
        state = self._preprocess_observation(observation)
        if self.loop_count == 0:
            input_dim = state.shape[0]
            action_dim = 2 * len(observation['stocks']) * 25
            self.agent = PPOAgent(input_dim=input_dim, action_dim=action_dim)
            # model_path = "ppo_model.pth"
            # try:
            #     self.agent.load_model(model_path)
            # except FileNotFoundError:
            #     print(f"No model found at {model_path}, starting from scratch.")
        action_dist = self.agent.select_action(state, info)
        actions = torch.argsort(action_dist.probs, descending=True)
        action = None
        decoded_action = None
        action_idx = 0
        for i in range(len(actions)):
            action = actions[i].item()
            decoded_action = self._decode_action(action, observation)
            if decoded_action is not None:
                action_idx = i
                break
        if decoded_action is None:
            decoded_action = {'stock_idx': 0, 'size': (0, 0), 'position': (0, 0)}
        if not self.update:
            self.loop_count += 1
            return decoded_action
        logprob = action_dist.log_prob(torch.tensor(action)).item()
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(logprob)
        self.memory.rewards.append(self._calculate_reward(decoded_action, observation, info) - 0.1 * action_idx)
        if info['trim_loss'] == 1:
            self.memory.is_terminals.append(True)
        else:
            self.memory.is_terminals.append(False)
        self.loop_count += 1
        if self.loop_count % 10 == 0:
            self.agent.update(self.memory)
            self.memory.clear_memory()
        return decoded_action

    def _calculate_reward(self, action, observation, info):
        stock_idx = action['stock_idx']
        stock = observation['stocks'][stock_idx]
        prod_w, prod_h = action['size']
        # filled_ratio = info['filled_ratio']
        # trim_loss = info['trim_loss']
        # Calculate space utilization (maximize this)
        stock_w, stock_h = self._get_stock_size_(stock)

        # Penalize unfit placements
        if not self._can_place_(stock, action['position'], (prod_w, prod_h)):
            return -10  # penalize bad placements
        stock_left = np.sum(stock == -1)
        prod_area = prod_w * prod_h
        # Positive reward for good placement
        reward = 10 * prod_area / (stock_w * stock_h) - 5 * (stock_left - prod_area) / stock_left
        return reward
    
    def _preprocess_observation(self, observation):
        stocks = observation['stocks']
        products = observation['products']
        max_stocks = 100
        max_products = 25
        
        padded_stocks = [np.pad(stock, ((0, max_stocks - stock.shape[0]),
                                        (0, max_stocks - stock.shape[1])),
                                mode='constant', constant_values=-2).flatten()
                         for stock in stocks[:max_stocks]]
        while len(padded_stocks) < max_stocks:
            padded_stocks.append(np.full((max_stocks * max_stocks), -2))
        stocks_tensor = np.concatenate(padded_stocks)

        padded_products = [[int(product["size"][0]), int(product["size"][1]), product["quantity"]]
                            if product["quantity"] > 0 else [0, 0, 0]
                            for product in products[:max_products]]
        while len(padded_products) < max_products:
            padded_products.append([0, 0, 0])
        products_tensor = np.concatenate(padded_products)

        obs_tensor = np.concatenate([stocks_tensor, products_tensor])
        return torch.tensor(obs_tensor, dtype=torch.float32)

    def _decode_action(self, action, observation):
        num_stocks = len(observation['stocks'])
        prods = observation['products']
        l = len(prods)
        stock_index = action // (l * 2)
        prod_index = (action // 2) % l
        rotate = action % 2

        if stock_index >= num_stocks or prod_index >= l:
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
                    return {'stock_idx': stock_index, 'size': (prod_w, prod_h), 'position': (i, j)}
        return None