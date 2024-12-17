from policy import Policy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from scipy.optimize import milp
from scipy.optimize import Bounds, LinearConstraint
import random


class ActorNetwork(nn.Module):
    def __init__(self, input_size, action_space):
        super().__init__()
        # Define the neural network architecture using Sequential
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),  # Fully connected layer from input to 256 neurons
            nn.LayerNorm(256),           # Normalize the layer output to stabilize training
            nn.ReLU(),                   # Apply ReLU activation function
            nn.Dropout(0.1),             # Dropout for regularization, reducing overfitting

            nn.Linear(256, 128),         # Second fully connected layer
            nn.LayerNorm(128),           # Layer normalization
            nn.ReLU(),                   # ReLU activation function
            nn.Dropout(0.1),             # Dropout for regularization

            nn.Linear(128, action_space) # Final layer mapping to action space size
        )
        
    def forward(self, x):
        # Forward pass through the network
        logits = self.network(x)
        
        # Use temperature scaling to control output sharpness
        temperature = 1.0
        
        # Apply softmax to convert logits into probabilities
        return nn.functional.softmax(logits / temperature, dim=-1)
    

class CriticNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        
        # Define the neural network architecture using Sequential
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),  # Fully connected layer from input to 256 neurons
            nn.LayerNorm(256),           # Normalize output to stabilize training
            nn.ReLU(),                   # Apply ReLU activation function
            nn.Dropout(0.1),             # Dropout for regularization
            
            nn.Linear(256, 128),         # Second fully connected layer
            nn.LayerNorm(128),           # Layer normalization
            nn.ReLU(),                   # ReLU activation function
            nn.Dropout(0.1),             # Dropout for regularization
            
            nn.Linear(128, 1)            # Final layer outputting a single value (value estimate)
        )
        
    def forward(self, x):
        # Forward pass through the network
        return self.network(x)


class A2CPolicy(Policy):
    def __init__(self, observation):
        learning_rate = 0.01

        # Extract stock-related information from the environment
        self.num_stock = len(observation["stocks"])          # Number of stocks
        self.stock_size = observation["stocks"][0].size      # Size of each stock feature vector
        self.action_space = 5000                             # Action space size
        self.max_product_type = 25                           # Maximum number of product types
        self.gamma = 0.99                                    # Discount factor for rewards

        # Exploration parameters for epsilon-greedy action selection
        self.epsilon = 0.5                                  # Initial exploration probability
        self.epsilon_min = 0.01                             # Minimum exploration probability
        self.epsilon_decay = 0.995                          # Decay rate for exploration

        # Device selection (CPU/GPU)
        self.device = torch.device("cpu")

        # Define fixed input size based on environment features
        self.input_size = self.max_product_type * 3 + self.num_stock * self.stock_size

        # Initialize actor and critic networks
        self.actor = ActorNetwork(self.input_size, self.action_space).to(self.device)
        self.critic = CriticNetwork(self.input_size).to(self.device)

        # Define optimizers for both networks
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Entropy coefficient to encourage exploration by penalizing low-entropy policies
        self.entropy_coef = 0.01

        # Initialize memory for storing experiences
        self.states = []             # States observed
        self.actions = []            # Actions taken
        self.rewards = []            # Rewards received
        self.next_states = []        # Next states observed
        self.dones = []              # Done flags indicating episode termination

        # Experience replay buffer settings
        self.batch_size = 32         # Number of samples for training in each batch

        # Track valid actions and last modified stock
        self.valid_actions = []      # Stores valid actions for the current state
        self.last_stock_idx = None   # Tracks which stock was modified last

    def _state_to_tensor(self, observation):
        # Initialize product features with zeros
        # Each product has 3 features: quantity, width, and height
        products_feature = np.zeros(self.max_product_type * 3)
        idx = 0
        
        # Extract features for each product in the observation
        for p in observation["products"]:
            if idx >= self.max_product_type * 3:  # Limit to maximum product types
                break
            if p["quantity"] > 0:  # Consider only available products
                products_feature[idx] = p["quantity"]  # Product quantity
                products_feature[idx + 1] = p["size"][0]  # Product width
                products_feature[idx + 2] = p["size"][1]  # Product height
                idx += 3  # Move to the next product slot

        # Initialize stock features with zeros
        # Each stock has a flattened 4x4 grid representation
        stocks_feature = np.zeros(self.num_stock * self.stock_size)

        # Extract features for each stock
        for i, stock in enumerate(observation["stocks"]):
            flat_stock = stock.flatten()  # Flatten the stock's 4x4 grid
            stocks_feature[i * self.stock_size:(i + 1) * self.stock_size] = flat_stock

        # Combine product and stock features into a single state vector
        state = np.concatenate([products_feature, stocks_feature])

        # Convert state to a PyTorch tensor and move to the specified device
        return torch.FloatTensor(state).to(self.device)

    def _get_valid_actions(self, observation):
        # Check if valid actions are empty or if this is the first call
        if not self.valid_actions or self.last_stock_idx is None:
            # Initialize valid actions list
            self.valid_actions = []
            
            # Iterate through each stock in the environment
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)  # Extract stock dimensions
                
                # Iterate through each product
                for prod_idx, prod in enumerate(observation["products"]):
                    if prod["quantity"] <= 0:  # Skip products with no available quantity
                        continue

                    prod_w, prod_h = prod["size"]  # Extract product dimensions

                    # Check normal orientation (width x height)
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                    self.valid_actions.append((prod_idx, stock_idx, x, y))

                    # Check rotated orientation (height x width)
                    if prod_w != prod_h and stock_w >= prod_h and stock_h >= prod_w:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                                    self.valid_actions.append(
                                        (prod_idx, stock_idx, x, y, True)
                                    )

        else:
            # Update valid actions for the last modified stock
            updated_valid_actions = []
            for action in self.valid_actions:
                prod_idx, stock_idx, x, y, *rotated = action

                # Check if the product still has a positive quantity
                if observation["products"][prod_idx]["quantity"] <= 0:
                    continue

                # If the stock was the last modified, validate the action again
                if stock_idx == self.last_stock_idx:
                    stock = observation["stocks"][stock_idx]
                    prod_w, prod_h = (
                        observation["products"][prod_idx]["size"]
                        if not rotated else observation["products"][prod_idx]["size"][::-1]
                    )
                    if not self._can_place_(stock, (x, y), (prod_w, prod_h)):
                        continue

                # If the action is still valid, keep it
                updated_valid_actions.append(action)

            # Update the valid actions list
            self.valid_actions = updated_valid_actions

        return self.valid_actions

    def _action_to_index(self, action):
        prod_idx, stock_idx, x, y = action[:4]  # Extract action components
        max_pos = 50  # Maximum possible position value
        
        # Compute a unique action index using a modular hash-like formula
        # This ensures the action index stays within the action space range
        return (prod_idx * max_pos * max_pos + stock_idx * max_pos + x) % self.action_space

    def _get_next_observation(self, action, observation):
        # Create deep copies of stocks and products to avoid modifying the original observation
        _stocks = copy.deepcopy(observation["stocks"])
        _products = copy.deepcopy(observation["products"])

        # Extract action details
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        width, height = size
        x, y = position

        # Find the product index in the product list
        product_idx = None
        for i, product in enumerate(_products):
            # Check if product size matches in either orientation
            if np.array_equal(product["size"], size) or np.array_equal(product["size"], size[::-1]):
                if product["quantity"] == 0:  # Skip if product is out of stock
                    continue
                product_idx = i  # Record matching product index
                break

        if product_idx is not None and 0 <= stock_idx < self.num_stock:
            stock = _stocks[stock_idx]

            # Calculate stock dimensions
            stock_width = np.sum(np.any(stock != -2, axis=1))  # Non-blocked width
            stock_height = np.sum(np.any(stock != -2, axis=0))  # Non-blocked height

            # Check if the product fits within the stock's available space
            if (
                x >= 0 and y >= 0 and
                x + width <= stock_width and
                y + height <= stock_height
            ):
                # Ensure the placement area is empty (-1 means empty)
                if np.all(stock[x:x + width, y:y + height] == -1):
                    # Place the product in the stock
                    stock[x:x + width, y:y + height] = product_idx
                    
                    # Decrease the product's available quantity
                    _products[product_idx]["quantity"] -= 1

        # Return updated observation
        return {"stocks": _stocks, "products": _products}

    
    def get_action(self, observation, info):
        # Retrieve all valid actions based on the current observation
        valid_actions = self._get_valid_actions(observation)
        if not valid_actions:
            # Return a null action if no valid actions are available
            return {"stock_idx": None, "size": (None, None), "position": (None, None)}
        
        # Convert the current state into a tensor for model input
        state = self._state_to_tensor(observation)

        # Epsilon-greedy action selection
        separate_rng = np.random.default_rng()  # Independent random number generator
        rd = separate_rng.random()
        if rd < self.epsilon:  # Exploration: choose a random action
            chosen_action = valid_actions[np.random.choice(len(valid_actions))]
        else:  # Exploitation: choose action based on the policy network
            with torch.no_grad():
                action_probs = self.actor(state)  # Predict action probabilities
            action_probs = action_probs.cpu().numpy()

            # Find the best action from valid actions
            max_prob = float('-inf')
            chosen_action = None
            for action in valid_actions:
                action_idx = self._action_to_index(action[:4])  # Map action to index
                prob = action_probs[action_idx]
                if prob > max_prob:  # Track the highest-probability action
                    max_prob = prob
                    chosen_action = action

        # Decay epsilon for reduced exploration over time
        self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-1.0 / self.epsilon_decay))

        if chosen_action is None:  # Safety check
            return {"stock_idx": None, "size": (None, None), "position": (None, None)}

        # Update the last modified stock
        self.last_stock_idx = chosen_action[1]

        # Prepare action format for the environment
        prod_idx = chosen_action[0]
        if len(chosen_action) == 5 and chosen_action[4]:  # If the product is rotated
            prod_size = observation["products"][prod_idx]["size"]
            return_action = {
                "stock_idx": chosen_action[1],
                "size": (prod_size[1], prod_size[0]),  # Swap width and height
                "position": (chosen_action[2], chosen_action[3])
            }
        else:  # Normal orientation
            return_action = {
                "stock_idx": chosen_action[1],
                "size": observation["products"][prod_idx]["size"],
                "position": (chosen_action[2], chosen_action[3])
            }

        # Simulate the environment step with the chosen action
        next_observation = self._get_next_observation(return_action, observation)

        # Calculate the reward from the environment
        reward = self.calculate_reward(observation, return_action, next_observation)

        # Convert the action into a tuple format for storage
        action_tuple = (prod_idx, return_action["stock_idx"],
                        return_action["position"][0], return_action["position"][1])

        # Check if all products are placed (termination condition)
        terminated = all([product["quantity"] == 0 for product in next_observation["products"]])

        # Store the experience in memory for training
        self.remember(observation, action_tuple, reward, next_observation, terminated)

        # Trigger training when enough experiences are collected or upon termination
        if len(self.states) >= 32 or (terminated and len(self.states) >= 2):
            self.train()

        # Reset valid actions and stock tracking upon termination
        if terminated:
            self.valid_actions = []
            self.last_stock_idx = None

        return return_action

    def remember(self, state, action, reward, next_state, done):
        # Convert the current state to a tensor and store it in memory
        self.states.append(self._state_to_tensor(state))
        
        # Convert the action into its corresponding index and store it
        self.actions.append(self._action_to_index(action))
        
        # Store the reward received after taking the action
        self.rewards.append(reward)
        
        # Convert the next state to a tensor and store it in memory
        self.next_states.append(self._state_to_tensor(next_state))
        
        # Store whether the episode ended after the action
        self.dones.append(done)

    def train(self):
        # Collect stored experiences into tensors
        states = torch.stack(self.states)
        actions = torch.tensor(self.actions).to(self.device)
        rewards = torch.tensor(self.rewards).float().to(self.device)
        next_states = torch.stack(self.next_states)
        dones = torch.tensor(self.dones).float().to(self.device)

        # Calculate returns and advantages using Generalized Advantage Estimation (GAE)
        returns = []
        advantages = []
        next_value = self.critic(next_states[-1]).detach()
        next_advantage = 0
        
        for r, d, value in zip(reversed(rewards), reversed(dones), 
                            reversed(self.critic(states).detach())):
            next_value = r + self.gamma * next_value * (1 - d)
            next_advantage = next_value - value.item()
            returns.insert(0, next_value)
            advantages.insert(0, next_advantage)

        # Convert to tensors
        returns = torch.tensor(returns).to(self.device)
        advantages = torch.tensor(advantages).to(self.device)

        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Forward pass through networks
        action_probs = self.actor(states)
        state_values = self.critic(states)

        # Calculate entropy bonus for exploration encouragement
        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1).mean()

        # Compute policy loss using selected action probabilities
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1))
        log_probs = torch.log(selected_action_probs + 1e-10).squeeze()
        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy

        # Compute value loss using Mean Squared Error
        critic_loss = nn.functional.mse_loss(state_values.squeeze(), returns)

        # Update the Actor Network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update the Critic Network
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Clear stored experiences
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []


    def calculate_reward(self, observation, action_dict, next_observation):
        if action_dict is None:
            return -1.0  # Penalty for invalid or no action taken
        
        # Calculate area utilization for all stocks
        total_area = 0
        used_area = 0
        for stock in next_observation["stocks"]:
            if np.sum(stock >= 0) > 0:  # Consider only stocks with at least one product
                stock_w, stock_h = self._get_stock_size_(stock)
                total_area += stock_w * stock_h
                used_area += np.sum(stock >= 0)  # Count filled cells
        
        utilization = used_area / (total_area + 1e-5)  # Prevent division by zero

        # Calculate number of products placed
        prev_products = sum(p["quantity"] for p in observation["products"])
        next_products = sum(p["quantity"] for p in next_observation["products"])
        products_placed = prev_products - next_products

        # Compute reward using weighted contributions
        reward = (
            utilization * 25.0  # Reward for maximizing stock utilization
            + products_placed * 0.5  # Reward for placing products
            + (next_products == 0) * 5.0  # Bonus for completing all products
            - (1 - utilization) * 0.5  # Minor penalty for unused stock area
        )
        
        return reward


class MILPPolicy(Policy):
    def __init__(self):
        """
        Initializes the MILPPolicy with necessary variables and configurations.
        """
        # Stores generated patterns from the MILP solution process
        self.patterns = []

        # Precomputed actions determined by the MILP model
        self.result_actions = []

        # Tracks the index of the next action to return from the precomputed list
        self.current_action_index = 0

    @staticmethod
    def fits_in_stock(stock_width, stock_height, product_width, product_height, x, y):
        """
        Checks if a product can fit into a stock area at the specified (x, y) position.
        """
        return x + product_width <= stock_width and y + product_height <= stock_height


    @staticmethod
    def rectangles_overlap(r1, r2):
        """
        Determines whether two rectangles overlap.
        """
        x1, y1, w1, h1 = r1
        x2, y2, w2, h2 = r2
        return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


    @staticmethod
    def get_demand_from_obs(observation):
        """
        Extracts and aggregates product demand from the observation.
        """
        products = observation["products"]
        demand = {}
        for product in products:
            width, height = product["size"]
            size = tuple(sorted((width, height)))  # Normalize size to ensure consistency
            quantity = product["quantity"]
            if size not in demand:
                demand[size] = 0
            demand[size] += quantity

        list_demand = [{"size": np.array([size[0], size[1]]), "quantity": quantity} for size, quantity in demand.items()]
        return sorted(list_demand, key=lambda x: (x["size"][0], x["size"][1]))


    @staticmethod
    def get_stock_info_from_obs(observation):
        """
        Extracts and aggregates stock area information from the observation.
        """
        stocks = observation["stocks"]
        stock_counts = {}
        for stock in stocks:
            # Calculate effective width and height of each stock
            stock_width = np.sum(np.any(stock != -2, axis=1))  # Width: rows with active slots
            stock_height = np.sum(np.any(stock != -2, axis=0))  # Height: columns with active slots
            stock_counts[(stock_width, stock_height)] = stock_counts.get((stock_width, stock_height), 0) + 1

        return [{"size": np.array([size[0], size[1]]), "quantity": quantity} for size, quantity in stock_counts.items()]

    def generate_random_pattern(self, products, stocks):
        """
        Generates random placement patterns for the given products and stocks.
        """
        # Initialize map_pattern for existing patterns
        map_pattern = {tuple(s["size"]): s["patterns"] for s in self.patterns}

        for stock_i, stock in enumerate(stocks):
            stock_size = tuple(stock["size"])

            # Initialize patterns for the current stock if not present
            if stock_size not in map_pattern:
                map_pattern[stock_size] = []

            stock_width, stock_height = stock["size"]
            remaining_demand = {tuple(p["size"]): p["quantity"] for p in products}

            max_patterns = 5  # Limit patterns generated per stock

            while any(remaining_demand.values()):  # Check if there's still demand
                product_count = [0] * len(products)
                placements = []
                can_be_placed = {tuple(p["size"]): True for p in products}

                while any(can_be_placed[tuple(p["size"])] for p in products):
                    # Randomly select a product
                    i, product = random.choice(list(enumerate(products)))
                    if not can_be_placed[tuple(product["size"])]:
                        continue

                    width, height = product["size"]

                    # Check if product has been fully placed
                    if product_count[i] >= product["quantity"]:
                        can_be_placed[tuple(product["size"])] = False
                        continue

                    placed = False

                    # Search for a valid position to place the product
                    for y in range(stock_height):
                        for x in range(stock_width):
                            for orientation in [(width, height), (height, width)]:
                                pw, ph = orientation
                                placement = (x, y, pw, ph)

                                # Check fit and overlap
                                if (
                                    self.fits_in_stock(stock_width, stock_height, pw, ph, x, y)
                                    and all(
                                        not self.rectangles_overlap(placement, placed_rect)
                                        for placed_rect in placements
                                    )
                                ):
                                    # Record placement and update demand
                                    placements.append(placement)
                                    if remaining_demand[tuple(product["size"])] > 0:
                                        remaining_demand[tuple(product["size"])] -= 1
                                    product_count[i] += 1
                                    placed = True
                                    break
                            if placed:
                                break

                    if not placed:
                        can_be_placed[tuple(product["size"])] = False

                # Record unique patterns
                if not any(
                    pattern["product_count"] == product_count
                    for pattern in map_pattern[stock_size]
                ):
                    pattern = {"product_count": product_count, "placement": placements}
                    map_pattern[stock_size].append(pattern)

                max_patterns -= 1

        # Update class attribute with generated patterns
        self.patterns = [{"size": size, "patterns": patterns} for size, patterns in map_pattern.items()]

    def MILP(self, c, A, b, m, n):
        """
        Solves the Mixed-Integer Linear Programming (MILP) problem.
        """
        # Define linear constraints
        constraints = LinearConstraint(A, -np.inf, b)

        # Set variable bounds
        bounds = Bounds(np.zeros(n), np.inf)

        # Specify integer variables
        integrality = np.ones(n, dtype=int)

        # Solve the MILP problem
        result = milp(c, constraints=constraints, bounds=bounds, integrality=integrality)
        
        # Return the optimization result
        return result


    def prepare_result_actions(self, observation):
        """
        Precomputes a list of actions based on the observation and MILP results.
        """

        # Extract demand and stock data
        products = self.get_demand_from_obs(observation)
        stocks = self.get_stock_info_from_obs(observation)

        # Generate random patterns
        GENERATE_TIMES = 1
        for _ in range(GENERATE_TIMES):
            self.generate_random_pattern(products, stocks)

        # Prepare MILP constraints
        demand_constraints = [pattern["product_count"] for s in self.patterns for pattern in s["patterns"]]
        demand = [product["quantity"] for product in products]

        num_stock_constraints = [[0 for _ in range(len(stocks))] for s in self.patterns for pattern in s["patterns"]]
        pattern_idx = 0
        for i, s in enumerate(self.patterns):
            for pattern in s["patterns"]:
                num_stock_constraints[pattern_idx][i] = 1
                pattern_idx += 1

        num_stock = [stock["quantity"] for stock in stocks]
        target = [s["size"][0] * s["size"][1] for s in self.patterns for pattern in s["patterns"]]

        # Convert to numpy arrays for MILP
        constraints = np.hstack((-np.array(demand_constraints), np.array(num_stock_constraints))).T
        r_constraints = np.hstack((-np.array(demand), np.array(num_stock)))
        target = np.array(target)

        # Solve the MILP problem
        result = self.MILP(target, constraints, r_constraints, len(constraints), len(target))

        if not result.success:
            print("The problem is infeasible.")
            return

        # Round MILP results to ensure integer solutions
        result.x = np.round(result.x)

        # Extract selected patterns
        result_patterns = []
        pattern_idx = 0
        oristocks = observation["stocks"]
        for s in self.patterns:
            for pattern in s["patterns"]:
                while result.x[pattern_idx] > 0:
                    result_patterns.append({"size": s["size"], "patterns": [pattern], "stock_idx": -1})
                    result.x[pattern_idx] -= 1
                pattern_idx += 1

        # Map patterns to available stocks
        used_stock_idx = set()
        for pattern in result_patterns:
            stock_width, stock_height = pattern["size"]
            for i, stock in enumerate(oristocks):
                width = np.sum(np.any(stock != -2, axis=1))
                height = np.sum(np.any(stock != -2, axis=0))
                if (stock_width, stock_height) == (width, height) and i not in used_stock_idx:
                    pattern["stock_idx"] = i
                    used_stock_idx.add(i)
                    break

        # Prepare the list of actions from patterns
        self.result_actions = []
        for result_pattern in result_patterns:
            stock_idx = result_pattern["stock_idx"]
            placements = result_pattern["patterns"][0]["placement"]
            for placement in placements:
                x, y, pw, ph = placement
                prod_size = (pw, ph)
                self.result_actions.append({"stock_idx": stock_idx, "size": prod_size, "position": (x, y)})

        # Reset action index
        self.current_action_index = 0

    def get_action(self, observation, info):
        """
        Returns the next action from the precomputed list of result actions.
        """

        # Ensure actions are prepared
        if self.current_action_index == 0:
            self.prepare_result_actions(observation)

        # Check if only one product is left
        if sum(product["quantity"] for product in observation["products"]) == 1:
            action = self.result_actions[self.current_action_index]
            for product in observation["products"]:
                # Match product size with the action size (considering both orientations)
                if product["quantity"] == 1 and (
                    action["size"] == (product["size"][0], product["size"][1]) or
                    action["size"] == (product["size"][1], product["size"][0])
                ):
                    # Reset precomputed actions and patterns
                    self.patterns = []
                    self.result_actions = []
                    self.current_action_index = 0
                    return action

        # Return the next action from the list
        action = self.result_actions[self.current_action_index]
        self.current_action_index += 1

        return action

class Policy2310510_2312506_2310167_2313914(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.policy = None
        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy is None:
            if self.policy_id == 1:
                self.policy = A2CPolicy(observation)
            
            elif self.policy_id == 2:
                self.policy = MILPPolicy()
        return self.policy.get_action(observation, info)