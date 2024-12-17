from policy import Policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler

STATE_SIZE = 1000004
STOCK_SIZE = 100
MAX_X = 100
MAX_Y = 100

class Policy2312080_2312105_2312265_2311289_2313429(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        

        if policy_id == 1:
            self.policy = best_fit_policy()
        elif policy_id == 2:
            self.policy = RL_policy()
            part_files = [
                "weights.pth.part1",
                "weights.pth.part2",
                "weights.pth.part3",
                "weights.pth.part4",
                "weights.pth.part5"
            ]
            with open('weights.pth', 'wb') as out_file:
                for part in part_files:
                    with open(part, 'rb') as part_file:
                        out_file.write(part_file.read())
            self.policy.load("weights.pth")

    def get_action(self, observation, info):
        return self.policy.get_action(observation=observation, info=info)


class best_fit_policy(Policy):

    def __init__(self):
        pass

    def get_action(self, observation, info):
        list_prods = observation["products"]

        list_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quality > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
                # Sort stocks with area descending
                # Create a list of tuples (index, stock, area)
                stocks_with_index = [(i, stock, self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]) 
                                     for i, stock in enumerate(observation["stocks"])]
                # Sort by area while keeping original indices
                stocks_with_index.sort(key=lambda x: x[2], reverse=True)
                # Unpack sorted stocks and indices
                sorted_stocks = [(t[0], t[1]) for t in stocks_with_index]
                
                for original_idx, stock in sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = original_idx
                            break

                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = None, None
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break
                        if pos_x is not None and pos_y is not None:
                            stock_idx = original_idx
                            break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
# * REINFORCEMENT SECTION:
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, max_stocks , max_x, max_y):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        
        self.stock_head = nn.Linear(32, max_stocks)
        self.x_head = nn.Linear(32, max_x)
        self.y_head = nn.Linear(32, max_y)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        stock_logits = self.stock_head(x) 
        x_logits = self.x_head(x)
        y_logits = self.y_head(x)
        
        return stock_logits, x_logits, y_logits

class RL_policy(Policy):
    def __init__(self, lr=1e-3, gamma=0.99, max_steps=1000, weight_decay=1e-4, device=None):
        """
        @Params:
            lr (float): learning rate
            gamma (float): discount reward factor
            weight_decay (float): L2 regularization param
            max_steps (int): max step for each episode
        """

        # * Network params
        self.lr = lr
        self.gamma = gamma
        self.max_steps = max_steps
        self.current_steps = 0
        self.weight_decay = weight_decay
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # * Init neuron network
        self.policy_network = PolicyNetwork(STATE_SIZE, STOCK_SIZE, MAX_X, MAX_Y) 
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.policy_network.to(self.device)

        # * Store log probabilities and rewards
        self.log_probs = []
        self.rewards = []

        self.scaler = torch.amp.GradScaler("cuda")
        
        
    def compute_waste_utilization(self, stocks) -> np.array:
        '''
            @param: stocks - danh sách các mảng stock hiện tại
            @return: np.array chứa [waste_amount, utilization_rate]
        '''
        total_area = 0
        total_placed_area = 0

        for stock in stocks:
            stock_area = np.sum(stock != -2)
            total_area += stock_area

            placed_area = np.sum(stock == 0)
            total_placed_area += placed_area

        waste_amount = total_area - total_placed_area
        utilization_rate = total_placed_area / total_area if total_area > 0 else 0.0

        return np.array([waste_amount, utilization_rate])

        
    def init_state(self, prod_size, stocks) -> np.array:
        '''
            @Param: observation
            Initial state from current observation
        '''
        stocks_vector = np.concatenate([stock.flatten() for stock in stocks])
        
        side_feature_vector = self.compute_waste_utilization(stocks=stocks)

        state = np.concatenate([
            stocks_vector,
            prod_size,
            side_feature_vector
        ])

        return state
    
    def compute_reward(self, waste, utilization):
        alpha = 1.5  
        beta = 2.0   

        max_waste = 10**3
        normalized_waste = min(waste / max_waste, 1.0)  

        normalized_utilization = utilization 

        reward = -alpha * normalized_waste + beta * normalized_utilization

        reward = min(reward, 10.0)

        return reward + 25

    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                state = self.init_state(prod_size=prod_size, stocks=observation["stocks"])

                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                stock_logits, x_logits, y_logits = self.policy_network(state_tensor)

                stock_dist = torch.distributions.Categorical(logits=stock_logits)
                x_dist = torch.distributions.Categorical(logits=x_logits)
                y_dist = torch.distributions.Categorical(logits=y_logits)

                stock_id = stock_dist.sample()
                x = x_dist.sample()
                y = y_dist.sample()

                log_prob = (
                    stock_dist.log_prob(stock_id) + 
                    x_dist.log_prob(x) + 
                    y_dist.log_prob(y)
                )

                if self._can_place_(stock=observation["stocks"][stock_id.item()], position=(x.item(), y.item()), prod_size=prod_size):
                    stock_idx = stock_id.item()
                    pos_x = x.item()
                    pos_y = y.item()

                    waste, utilization = self.compute_waste_utilization(observation["stocks"])
                    reward = self.compute_reward(waste=waste, utilization=utilization)

                    self.log_probs.append(log_prob)
                    self.rewards.append(reward)
                else:
                    penalty = -50
                    self.log_probs.append(log_prob)
                    self.rewards.append(penalty)
                    stocks_with_index = [(i, stock, self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]) 
                                    for i, stock in enumerate(observation["stocks"])]

                    stocks_with_index.sort(key=lambda x: x[2], reverse=True)

                    sorted_stocks = [(t[0], t[1]) for t in stocks_with_index]
                    
                    for original_idx, stock in sorted_stocks:
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size
                        if stock_w >= prod_w and stock_h >= prod_h:
                            pos_x, pos_y = None, None
                            for x_try in range(stock_w - prod_w + 1):
                                for y_try in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x_try, y_try), prod_size):
                                        pos_x, pos_y = x_try, y_try
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = original_idx
                                break

                        if stock_w >= prod_h and stock_h >= prod_w:
                            pos_x, pos_y = None, None
                            for x_try in range(stock_w - prod_h + 1):
                                for y_try in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x_try, y_try), prod_size[::-1]):
                                        prod_size = prod_size[::-1]
                                        pos_x, pos_y = x_try, y_try
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break
                            if pos_x is not None and pos_y is not None:
                                stock_idx = original_idx
                                break

                    if pos_x is not None and pos_y is not None:
                        break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}


    def update_params(self, batch_size=32):
        discounted_rewards = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.device)
        

        if discounted_rewards.std() == 0:
            discounted_rewards = discounted_rewards - discounted_rewards.mean()
        else:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        

        for i in range(0, len(self.log_probs), batch_size):
            batch_log_probs = self.log_probs[i:i + batch_size]
            batch_rewards = discounted_rewards[i:i + batch_size]

            policy_loss = []
            for log_prob, R in zip(batch_log_probs, batch_rewards):
                policy_loss.append(-log_prob * R)
            

            policy_loss = torch.stack(policy_loss).sum()

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                self.scaler.scale(policy_loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        # Reset log_probs and rewards after update
        self.log_probs = []
        self.rewards = []

    def store(self, filepath):
        checkpoint = {
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Weights and optimizer states saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.policy_network.to(self.device)
        
        # print(f"Weights and optimizer states loaded from {filepath}")






                    