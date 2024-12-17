from policy import Policy
import numpy as np
import random

class Policy2210348_2213893_2212309(Policy):
    def __init__(self, policy_id=1, stocks=None, products=None):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.stocks = stocks if stocks is not None else []  # Stock data
        self.products = products if products is not None else []  # Product data
        self.policy_table = {}  # Policy probabilities for each state-action pair
        self.value_table = {}  # Value function table for PPO
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.2  # PPO clipping range
        self.actor_lr = 0.01  # Actor learning rate
        self.critic_lr = 0.01  # Critic learning rate
        self.replay_buffer = []  # Experience buffer
        self.replay_buffer_maxlen = 50  # Maximum buffer size
        self.num_actions = 10  # Number of possible actions
        self.entropy_coefficient = 0.01  # Encourage exploration
        self.value_coefficient = 0.5  # Value loss weighting
        self.advantage_normalization = True  # Normalize advantages

    def _get_state_key(self, observation, info):
        return str(observation)

    def get_action(self, observation, info):
        if self.policy_id == 1:
            state = self._get_state_key(observation, info)

            if state not in self.policy_table:
                # Initialize policy probabilities and value table for the state
                self.policy_table[state] = np.ones(self.num_actions) / self.num_actions
                self.value_table[state] = 0.0

            if random.uniform(0, 1) < 0.1:  # Exploration
                action = self._random_action(observation)
            else:  # Exploitation
                action_probs = self.policy_table[state]
                action_idx = np.random.choice(len(action_probs), p=action_probs)
                action = self._decode_action(action_idx, observation)
            return action
        else:
            return self.Bestfit(observation, info)

    def Bestfit(self, observation, info):
        list_prods = sorted(
            observation["products"],
            key=lambda p: p["size"][0] * p["size"][1],  # Sort by area
            reverse=True,
        )

        def best_fit():
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    best_score = float("inf")
                    best_placement = None

                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)

                        # Skip stocks smaller than the product
                        if stock_w < prod_size[0] or stock_h < prod_size[1]:
                            continue

                        # Try original orientation
                        placement = self._find_best_position(stock, prod_size, stock_w, stock_h)
                        if placement:
                            score = self._calculate_remaining_space(stock, prod_size, stock_w, stock_h, placement)
                            if score < best_score:
                                best_score = score
                                best_placement = {"stock_idx": i, "size": prod_size, "position": placement}

                        # Try rotated orientation
                        if prod_size[0] != prod_size[1]:  # If not square
                            rotated_size = prod_size[::-1]
                            if stock_w >= rotated_size[0] and stock_h >= rotated_size[1]:
                                placement = self._find_best_position(stock, rotated_size, stock_w, stock_h)
                                if placement:
                                    score = self._calculate_remaining_space(stock, rotated_size, stock_w, stock_h, placement)
                                    if score < best_score:
                                        best_score = score
                                        best_placement = {
                                            "stock_idx": i,
                                            "size": rotated_size,
                                            "position": placement,
                                        }

                    if best_placement:
                        return best_placement

            # No valid placement found
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        return best_fit()

    def _find_best_position(self, stock, prod_size, stock_w, stock_h):
        """
        Find the best position for the product in the stock.
        """
        for x in range(stock_w - prod_size[0] + 1):
            for y in range(stock_h - prod_size[1] + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None

    def _calculate_remaining_space(self, stock, prod_size, stock_w, stock_h, position):
        """
        Calculate the remaining free space after placing the product.
        """
        used_area = prod_size[0] * prod_size[1]
        total_area = stock_w * stock_h
        remaining_area = total_area - used_area
        return remaining_area

    def _random_action(self, observation):
        """Generate a random valid action."""
        for stock_idx, stock in enumerate(observation["stocks"]):
            for product in observation["products"]:
                if product["quantity"] > 0:
                    prod_size = product["size"]
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

    def _decode_action(self, action_idx, observation):
        """Decode the action index into the actual action."""
        stocks = observation["stocks"]
        stock_idx = action_idx // self.num_actions
        position = (action_idx % self.num_actions, action_idx % self.num_actions)
        return {
            "stock_idx": stock_idx,
            "size": [1, 1],  # Simplified for demonstration
            "position": position,
        }

    def step(self, observation, action, reward, next_observation, info):
        state = self._get_state_key(observation, info)
        next_state = self._get_state_key(next_observation, info)

        # Store experience
        self.store_experience(state, action, reward, next_state, info.get("done", False))

        if len(self.replay_buffer) >= 32:  # Update after sufficient experiences
            self.update_policy()

        return self.get_action(next_observation, info)

    def store_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.replay_buffer) >= self.replay_buffer_maxlen:
            self.replay_buffer.pop(0)
        self.replay_buffer.append(experience)

    def update_policy(self):
        states, actions, rewards, next_states, dones = zip(*self.replay_buffer)

        advantages = []
        returns = []
        for i in range(len(rewards)):
            G = 0
            advantage = 0
            for t in range(i, len(rewards)):
                G += rewards[t] * (self.gamma ** (t - i))
                if t + 1 < len(rewards):
                    advantage += rewards[t] + self.gamma * self.value_table.get(next_states[t], 0) - self.value_table[states[t]]
            returns.append(G)
            advantages.append(advantage)

        # Normalize advantages if enabled
        if self.advantage_normalization:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        for state, action, advantage, G in zip(states, actions, advantages, returns):
            action_idx = self._get_action_index(action)

            # Update policy probabilities with entropy regularization
            old_prob = self.policy_table[state][action_idx]
            entropy = -np.sum(self.policy_table[state] * np.log(self.policy_table[state] + 1e-8))
            new_prob = old_prob * np.exp(self.actor_lr * advantage + self.entropy_coefficient * entropy)
            self.policy_table[state] = np.clip(new_prob, old_prob - self.epsilon, old_prob + self.epsilon)

            # Normalize policy probabilities
            self.policy_table[state] /= np.sum(self.policy_table[state])

            # Update value function with added loss weighting
            value_loss = self.value_coefficient * (G - self.value_table[state]) ** 2
            self.value_table[state] += self.critic_lr * (G - self.value_table[state]) - value_loss

    def _get_action_index(self, action):
        stock_idx = action.get("stock_idx", -1)
        pos_x, pos_y = action.get("position", (0, 0))
        return stock_idx * self.num_actions + pos_x  # Simplified encoding

    def _compute_reward(self, observation, action, info):
        filled_ratio = info.get("filled_ratio", 0)
        trim_loss = info.get("trim_loss", 0)

        placement_reward = 100 if action["stock_idx"] != -1 else -50
        efficiency_reward = 500 * filled_ratio
        trim_penalty = -500 * trim_loss

        # Penalize extreme trim loss and reward significant filled ratios
        if filled_ratio >= 0.1:
            efficiency_reward += 200
        if trim_loss > 0.8:
            trim_penalty -= 300

        total_reward = placement_reward + efficiency_reward + trim_penalty
        return max(-2000, min(total_reward, 500))


