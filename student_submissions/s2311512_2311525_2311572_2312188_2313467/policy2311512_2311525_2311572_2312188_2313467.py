from policy import Policy
import numpy as np
import random 

class Policy2311512_2311525_2311572_2312188_2313467(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            ffd = FFD()
            return ffd.get_action(observation, info)
        elif self.policy_id == 2:
            q = Q()
            return q.get_action(observation, info)
#class FFD
class FFD(Policy):
    def __init__(self):
        self.processed_stocks = -1

    def get_action(self, observation, info):
        list_prods = observation["products"]
        list_stocks = enumerate(observation["stocks"])

        # Sort products by their area in decreasing order
        self.sorted_prods = sorted(
            list_prods,
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )
        # Sort stocks by their area in decreasing order
        self.sorted_stocks = sorted(
            list_stocks,
            key=lambda item: self.get_stock_area(item[1]),
            reverse=True
        )
        # Reset processed_stocks counter if new products and stocks are found
        if self.sorted_stocks[0][1][0][0] == -1:
            self.processed_stocks = -1
        idx = -1
        # Loop through sorted products
        for i, stock in self.sorted_stocks:
            idx += 1
            # If a stock has been processed completely, don't touch it again to increase speed
            if stock[0][0] == -1:
                self.processed_stocks += 1
            available_space = np.count_nonzero(stock == -1)
            # Move on to a stock that has not yet been fully processed.
            if idx < self.processed_stocks:              
                continue
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod in self.sorted_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    # If the product exceeds the available space, move on to the next product
                    if prod_size[0] * prod_size[1] > available_space:
                        continue
                    # Check both orientations of the product
                    for orien in [prod_size, prod_size[::-1]]:
                        if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                            max_x = stock_w - orien[0] + 1
                            max_y = stock_h - orien[1] + 1
                            for x in range(max_x):
                                for y in range(max_y):
                                    if self._can_place_(stock, (x, y), orien):
                                        return {"stock_idx": i, "size": orien, "position": (x, y)}
            
                                    
        return {"stock_idx": -1, "size": [0,0], "position": (0,0)}
    
    def get_stock_area(self, stock):
        w, h = self._get_stock_size_(stock)
        return w * h
#class q learning
class Q(Policy):
    def __init__(self, num_actions=3, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__()
        self.q_table = {}  # Dictionary to store Q-values
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.num_actions = num_actions  # Number of possible actions

    def _get_state_key(self, observation, info):
        return str(observation)

    def get_action(self, observation, info):
        state = self._get_state_key(observation, info)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        if random.uniform(0, 1) < self.epsilon:
            action = self._random_action(observation, state)
        else:
            action = self._best_action(observation, state)

        # Update Q-value directly after taking the action
        self.update_q_table(state, action, observation, info)

        return action

    def _random_action(self, observation, state):
        action_space = self._generate_action_space(observation)
        if action_space:
            return random.choice(action_space)
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _best_action(self, observation, state):
        action_space = self._generate_action_space(observation)
        if action_space:
            action_indices = range(len(action_space))
            best_index = np.argmax(self.q_table[state][:len(action_space)])
            return action_space[best_index]
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _generate_action_space(self, observation):
        """
        Generate all possible actions based on the observation.
        Handles a large number of actions dynamically.
        """
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
        return action_space[:self.num_actions]

    def _action_to_index(self, action):
        """
        Convert an action to a unique index for the Q-table.
        For larger spaces, ensure efficient mapping.
        """
        # Example: Use a hash or a tuple-based approach for larger spaces
        action_hash = hash((action["stock_idx"], tuple(action["size"]), tuple(action["position"])))
        return abs(action_hash) % self.num_actions

    def update_q_table(self, state, action, observation, info):
        action_idx = self._action_to_index(action)
        if action_idx >= self.num_actions:
            action_idx = self.num_actions - 1  # Clamp to max index

        reward = self._compute_reward(observation, action, info)
        next_state = self._get_state_key(observation, info)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        current_q = self.q_table[state][action_idx]
        max_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action_idx] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def _compute_reward(self, observation, action, info):
        filled_ratio = info.get("filled_ratio", 0.0)
        trim_loss = info.get("trim_loss", 0.0)
        max_trim_loss_penalty = 50.0
        max_filled_ratio_bonus = 100.0
        invalid_action_penalty = -10.0
        valid_action_reward = 20.0

        trim_loss_penalty = max_trim_loss_penalty * (trim_loss / (trim_loss + 1))
        filled_ratio_bonus = max_filled_ratio_bonus * filled_ratio

        if action["stock_idx"] == -1:
            return invalid_action_penalty

        product_size = action["size"]
        position = action["position"]

        for stock in observation["stocks"]:
            if self._can_place_(stock, position, product_size):
                return valid_action_reward + filled_ratio_bonus - trim_loss_penalty

        return -1 - trim_loss_penalty