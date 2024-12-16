from policy import Policy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Policy2311070_2310574_2310548_2310993(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = FirstFitPolicy()
        elif policy_id == 2:
            self.policy = ActorCriticPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

class FirstFitPolicy(Policy):
    def __init__(self):
        self.level = []
        for _ in range(100):
            self.level.append([])

    def get_action(self, observation, info):
        list_prods = observation["products"]
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        list_prod = []
        # Iterate through the list of products
        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue
            # Add both orientations (rotated and unrotated) to the list of possible products
            list_prod.append({'size': prod["size"], 'quantity': prod["quantity"]})
            list_prod.append({'size': prod["size"][::-1], 'quantity': prod["quantity"]})
        if list_prod == []:
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        # Choose the product with the largest height to place first
        prod = max(list_prod, key=lambda prod: prod["size"][1])
        pos_x, pos_y = None, None
        # Loop through all stocks
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_idx = i
            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            if stock_w < prod_w or stock_h < prod_h:
                continue
            if self._can_place_(stock, (0, 0), prod_size):
                pos_x, pos_y = 0, 0
                self.level[i] = [0]
                self.level[i].append(prod_h)
                break 
            # Loop through all the levels in the stock to find an empty space
            for j in range(len(self.level[i]) - 1):
                h = self.level[i][j + 1] - self.level[i][j]
                w = 0
                while not self._can_place_(stock, (w, self.level[i][j]), (1, h)):
                    w += 1   
                if w + prod_w > stock_w:
                    # Check if the current level is the last one
                    if j == len(self.level[i]) - 2:
                        if self.level[i][j + 1] + prod_h > stock_h:
                            break
                        if self._can_place_(stock, (0, self.level[i][j + 1]), prod_size):
                            pos_x, pos_y = 0, self.level[i][j + 1]
                            self.level[i].append(self.level[i][j + 1] + prod_h)
                            break
                    else:
                        continue
                if self._can_place_(stock, (w, self.level[i][j]), prod_size):
                    pos_x, pos_y = w, self.level[i][j]
                    break
            if pos_x is not None and pos_y is not None:
                break   
            else:
                continue
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        x = self.network(x)
        return torch.nn.functional.log_softmax(x, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.network(x)

class ActorCriticPolicy(Policy):
    def __init__(self):
        self.state_dim = 576
        self.action_dim = 25
        self.alpha = 0.01
        self.beta = 0.01
        self.epsilon = 1.0

        self.actor = ActorNetwork(self.state_dim, self.action_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.alpha)
        self.critic = CriticNetwork(self.state_dim)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.beta)

        self.reward = 0
        self.idx = 0
        self.trim_loss =  0
        self.state = None

    def encode_state(self, observation):
        used_stock = 0
        stock_feature = []
        # Add stock features (width, height, filled ratio, placeable nearest position) to the list
        for stock in observation["stocks"]:
            if (stock >= 0).sum() > 0:
                used_stock += 1
            stock_w, stock_h = self._get_stock_size_(stock)
            filled_ratio = (stock >= 0).sum() / (stock_w * stock_h)
            pos_x, pos_y = -1, -1
            for x in range(stock_w):
                for y in range(stock_h):
                    if self._can_place_(stock, (x, y), (1, 1)):
                        pos_x, pos_y = x, y
                        break
                if pos_x != -1 and pos_y != -1:
                    break
            stock_feature.extend([stock_w / 10, stock_h / 10, filled_ratio, pos_x / 10, pos_y / 10]) 
        stock_feature.append(used_stock / 10)
        # Add product features (width, height, quantity) to the list
        prod_feature = []
        for prod in observation["products"]:
            prod_w, prod_h = prod["size"]
            quantity = prod["quantity"]
            prod_feature.extend([prod_w / 10, prod_h / 10, quantity / 10])

        stock_feature = np.array(stock_feature, dtype=np.float32)
        prod_feature = np.array(prod_feature, dtype=np.float32)
        # Pad the product feature array to fix the length
        prod_feature = np.pad(prod_feature, (0, max(0, 75 - len(prod_feature))), mode='constant')[:75]
        # Concatenate stock and product features
        state = np.concatenate([stock_feature, prod_feature])
        return torch.FloatTensor(state)

    def get_action(self, observation, info):
        state = self.encode_state(observation)
        # Update the policy
        cutted_stock = 0
        for stock in observation["stocks"]:
            if (stock >= 0).sum() > 0:
                cutted_stock += 1
        if cutted_stock > 0:
            self.update_policy(self.state, state, observation)
        self.state = state
        # Calculate the trim loss of the stock
        trim_loss = []
        for stock in observation["stocks"]:
            if (stock >= 0).sum() == 0:
                continue
            area = (stock != -2).sum()
            empty = (stock == -1).sum()
            trim_loss.append(empty / area)
        self.trim_loss = np.mean(trim_loss).item() if trim_loss else 1

        idx = 0
        with torch.no_grad():
            prob = self.actor(state)
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            idx = np.random.randint(0, self.action_dim)
        else:
            idx = torch.argmax(prob).item()

        self.idx = idx
        return self.decode_action(idx, observation)

    def decode_action(self, idx, observation):
        # If the index of the action >= the number of products, 
        # penalize the reward and pick a random product index
        if idx >= len(observation["products"]):
            self.reward -= 1000
            idx = np.random.randint(0, len(observation["products"]))

        prod = observation["products"][idx]
        # If the product quantity is equal to 0, penalize and pick a random product index
        if (prod["quantity"] == 0):
            self.reward -= 1000
            while (prod["quantity"] == 0):
                idx = np.random.randint(0, len(observation["products"]))
                prod = observation["products"][idx]

        prod_w, prod_h = prod["size"]
        stock_idx = None
        position = None
        prod_size = None
        min_waste = float('inf')
        # Loop through all stocks
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            # Check both orientations of the product
            for rot in [(prod_w, prod_h), (prod_h, prod_w)]:
                prod_w, prod_h = rot
                if prod_w <= stock_w and prod_h <= stock_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                # Calculate the waste if the product is placed at this position
                                waste = self.calculate_waste(stock, (x, y), (prod_w, prod_h))
                                if waste < min_waste:
                                    min_waste = waste
                                    stock_idx = i
                                    position = (x, y)
                                    prod_size = rot
            if position is not None:
                break
        if position is not None:
            return {"stock_idx": stock_idx, "size": prod_size, "position": position}

        self.reward -= 1000
        return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}

    def calculate_reward(self, observation):
        trim_loss = []
        # Calculate the trim loss of the stock
        for stock in observation["stocks"]:
            if (stock >= 0).sum() == 0:
                continue
            area = (stock != -2).sum()
            empty = (stock == -1).sum()
            trim_loss.append(empty / area)
        new_trim_loss = np.mean(trim_loss).item() if trim_loss else 1
        # If the new trim loss < the previous trim loss, reward the agent
        # Otherwise, penalize the agent
        if (new_trim_loss < self.trim_loss):
            self.reward += 10
        else:
            self.reward += -10
        return self.reward

    def update_policy(self, state, next_state, observation):
        V_state = self.critic(state)
        V_next = self.critic(next_state)
        # Compute the critic's loss function (mean squared error)
        reward = self.calculate_reward(observation)
        TD_error = torch.tensor(reward, dtype=torch.float32) + self.beta*V_next - V_state
        critic_loss = 1/2 * (V_state - (reward + self.beta*V_next)) ** 2
        # Update the critic network
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True) 
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()
        # Compute the actor's loss function
        with torch.no_grad():
            V_state = self.critic(state)
            V_next = self.critic(next_state)
        TD_error = torch.tensor(reward, dtype=torch.float32) + self.beta*V_next - V_state
        log_prob = self.actor(state)
        log_prob = log_prob[self.idx]
        actor_loss = - log_prob * TD_error
        # Update the actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()
        # Decay the epsilon value to reduce the exploration
        self.epsilon = max(0.01, self.epsilon * 0.999)
        self.reward = 0

    def calculate_waste(self, stock, position, prod_size):
        # Calculate the waste if the product is placed at this position
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        stock_w, stock_h = self._get_stock_size_(stock)
        area = 0
        for i in range(max(0, pos_x-1), min(stock_w, pos_x+prod_w)):
            for j in range(max(0, pos_y-1), min(stock_h, pos_y+prod_h)):
                if stock[i,j] == -1:
                    area += 1
        return area - prod_w * prod_h
