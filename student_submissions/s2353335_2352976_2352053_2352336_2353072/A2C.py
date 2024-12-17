import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from policy import Policy
import matplotlib.pyplot as plt

class CuttingStockMetrics:
    def __init__(self):
        # Episode-level metrics
        self.episode_metrics = {
            'filled_ratios': [],
            'waste_ratios': [],
            'completed_products': [],
            'invalid_actions': [],
            'episode_lengths': [],
            'rewards': [],
            'edge_utilization': [],
            'corner_placements': [],
            'largest_waste_area': [],
            'product_completion_order': []
        }
        
        # Add episode tracking
        self.episode_history = {
            'episode_filled_ratios': [],  # Track filled ratio per episode
            'episode_rewards': [],        # Track total reward per episode
            'episode_numbers': []         # Track episode numbers
        }
        
        # Best scores
        self.best_scores = {
            'best_filled_ratio': 0.0,
            'best_reward': float('-inf'),
            'best_episode': -1
        }
        
        # Running averages
        self.running_averages = {
            'filled_ratio': deque(maxlen=10),
            'waste_ratio': deque(maxlen=10),
            'reward': deque(maxlen=10)
        }

    def add_episode_data(self, episode_number, filled_ratio, total_reward):
        """Record data for a completed episode"""
        self.episode_history['episode_numbers'].append(episode_number)
        self.episode_history['episode_filled_ratios'].append(filled_ratio)
        self.episode_history['episode_rewards'].append(total_reward)
        
        # Update best scores
        if filled_ratio > self.best_scores['best_filled_ratio']:
            self.best_scores['best_filled_ratio'] = filled_ratio
        if total_reward > self.best_scores['best_reward']:
            self.best_scores['best_reward'] = total_reward
            self.best_scores['best_episode'] = episode_number

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        # Double the action dimension to account for rotations
        self.action_dim = action_dim * 2  # Each position now has a rotated variant
        
        self.fc1 = nn.Linear(state_dim, 64)
        self.ln1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 32)
        self.ln2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, self.action_dim)
        
        # Orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        logits = self.fc3(x)
        
        if logits.size(0) == 1:
            logits = logits.squeeze(0)
            
        return logits

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 32),  # Giảm từ 256 xuống 128
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),  # Thêm dropout
            nn.Linear(32, 8),   # Giảm từ 64 xuống 32
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # Initialize weights với gain nhỏ hơn
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.01)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.network(state)

class ActorCriticPolicy2(Policy):
    def __init__(self):
        super().__init__()
        # State and action dimensions
        max_stocks = 100
        self.max_products = None
        self.state_dim = None
        self.action_dim = max_stocks * 25 * 2  # Double for rotations
        
        # Training flag
        self.training = True
        
        # Determine device
        self.device = (
            "mps" if torch.backends.mps.is_available() 
            else "cuda" if torch.cuda.is_available() 
            else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Basic Actor-Critic parameters
        self.gamma = 0.99
        self.gae_lambda = 0.95  # Added GAE parameter
        self.entropy_coef = 0.01
        # self.clip_epsilon = 0.2  # Không cần thiết trong A2C
        self.num_epochs = 1  # Chỉ cập nhật một lần cho A2C
        self.batch_size = 128  # Có thể điều chỉnh hoặc bỏ qua
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        
        # Training setup
        self.steps = 0
        self.prev_filled_ratio = 0.0
        
        # Model saving
        self.model_path = "saved_models/"
        os.makedirs(self.model_path, exist_ok=True)
        
        # Initialize metrics
        self.metrics = CuttingStockMetrics()
        
        # Initialize product tracking
        self.initial_products = None
        self.prev_total_products = None
        
        # Initialize reward tracking
        self.last_reward = 0
        self.reward_history = deque(maxlen=10)
        
        # Initialize memory for A2C
        self.memory = A2CMemory()

    def initialize_networks(self, observation):
        """Initialize networks after getting first observation"""
        if self.max_products is None:
            self.max_products = len(observation["products"])
            self.initial_products = sum(prod["quantity"] for prod in observation["products"])
            self.prev_total_products = self.initial_products
            
            stock_features = 100 * 3
            product_features = self.max_products * 3
            global_features = 2
            self.state_dim = stock_features + product_features + global_features
            
            # Initialize networks
            self.actor = ActorNetwork(self.state_dim, self.action_dim).to(self.device)
            self.critic = CriticNetwork(self.state_dim).to(self.device)
            
            # Optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
            
            # Initialize state normalization
            self.state_mean = torch.zeros(self.state_dim).to(self.device)
            self.state_std = torch.ones(self.state_dim).to(self.device)

    def preprocess_observation(self, observation, info):
        """Convert observation to state tensor with fixed dimensions"""
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Stock features - ensure 100 stocks
        stock_features = []
        for stock in stocks[:100]:
            stock_w, stock_h = self._get_stock_size_(stock)
            used_space = np.sum(stock != -1)
            total_space = stock_w * stock_h
            stock_features.extend([
                stock_w / 10.0,  # Normalized width
                stock_h / 10.0,  # Normalized height
                used_space / total_space  # Utilization ratio
            ])
        
        # Pad if lacking stocks
        while len(stock_features) < 300:
            stock_features.extend([0, 0, 0])
        
        # Product features - ensure max_products
        prod_features = []
        for prod in products[:self.max_products]:
            if prod["quantity"] > 0:
                prod_features.extend([
                    prod["size"][0] / 10.0,  # Normalized width
                    prod["size"][1] / 10.0,  # Normalized height
                    min(prod["quantity"], 10) / 10.0  # Normalized quantity
                ])
        
        # Pad if lacking products
        while len(prod_features) < self.max_products * 3:
            prod_features.extend([0, 0, 0])
        
        # Global features
        global_features = [
            info.get('filled_ratio', 0),
            self.steps / 1000.0  # Normalized step count
        ]
        
        # Combine all features
        state = np.array(stock_features + prod_features + global_features, dtype=np.float32)
        return torch.FloatTensor(state)

    def get_action(self, observation, info):
        """
        Determines the action to take based on the current observation and policy.
        Prioritizes placing products in existing partially filled stocks to minimize stock usage.
        
        Parameters:
            observation (dict): Current state observation.
            info (dict): Additional information (if any).

        Returns:
            action: The selected action for placing the product.
        """
        # Initialize networks if first time
        if self.max_products is None:
            self.initialize_networks(observation)
        
        remaining_products = sum(prod["quantity"] for prod in observation["products"])

        # Use structured placement more consistently early on
        if remaining_products > 30:
            placement_action = self._get_structured_placement(observation)
            if placement_action is not None:
                self.steps += 1
                return placement_action
        
        # Largest-first strategy
        largest_product = self._get_largest_product(observation)
        if largest_product:
            best_stock_idx = self.find_best_fitting_stock(observation, largest_product["size"])
            
            if best_stock_idx is not None:
                state = self.preprocess_observation(observation, info)
                state = self.normalize_state(state).unsqueeze(0)
                
                with torch.no_grad():
                    logits = self.actor(state).squeeze(0)
                    
                    # Strongly bias towards best_stock_idx actions
                    boost_factor = 400.0  # High boost to prioritize existing stock
                    stock_actions = torch.arange(25, device=self.device) + (best_stock_idx * 25)
                    logits[stock_actions] += boost_factor
                    
                    # Adaptive temperature
                    temperature = max(1.0 - (self.steps / 20000), 0.1)
                    logits = logits / temperature

                    logits = torch.clamp(logits, min=-20.0, max=20.0)
                    
                    # Handle NaN logits
                    if torch.isnan(logits).any():
                        logits = torch.zeros_like(logits)

                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()

                    if self.training:
                        log_prob = dist.log_prob(action)
                        value = self.critic(state)
                        self.memory.add(state.cpu(), action.item(), 0, value.item(), log_prob.item(), False)
            
            # Convert to actual placement action
            placement_action = self.convert_action(action.item(), observation)
            
            # If invalid action, fallback to structured placement
            if placement_action is None:
                placement_action = self._get_structured_placement(observation)
            
            self.steps += 1
            return placement_action

        # Fallback in case no largest product is found
        placement_action = self._get_structured_placement(observation)
        self.steps += 1
        return placement_action


    def normalize_state(self, state):
    # Convert to float32 and move to device
        state = state.to(dtype=torch.float32, device=self.device)
        state_mean = self.state_mean.to(dtype=torch.float32, device=self.device)
        state_std = self.state_std.to(dtype=torch.float32, device=self.device)
        return (state - state_mean) / (state_std + 1e-8)

    def update_state_normalizer(self, state):
        # Convert state to float32 before calculations
        state = state.to(dtype=torch.float32)
        self.state_mean = 0.99 * self.state_mean.to(dtype=torch.float32) + 0.01 * state.mean()
        self.state_std = 0.99 * self.state_std.to(dtype=torch.float32) + 0.01 * state.std()

    def update_policy(self, reward=None, done=None, info=None):
        """A2C Update: Compute advantage and update networks"""
        if not self.training:
            return
        
        if reward is not None and self.memory.size > 0:
            self.memory.rewards[-1] = reward
            self.memory.is_terminals[-1] = done
            
            if info and 'observation' in info:
                correct_filled_ratio = self.calculate_filled_ratio(info['observation'])
                self.prev_filled_ratio = correct_filled_ratio
        
        if done and self.memory.size > 0:
            states, actions, rewards, values, old_log_probs, dones = self.memory.generate_batches()
            
            if states is None:
                return
                
            # Move states to device first
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            
            # Compute returns and advantages
            returns = []
            advantages = []
            R = 0
            
            for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
                if d:
                    R = 0
                R = r + self.gamma * R
                advantage = R - v
                returns.append(R)
                advantages.append(advantage)
            
            # Single conversion to tensors
            returns = torch.tensor(list(reversed(returns)), dtype=torch.float32, device=self.device)
            advantages = torch.tensor(list(reversed(advantages)), dtype=torch.float32, device=self.device)
            actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
            old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            # Now states is guaranteed to be on MPS device
            logits = self.actor(states)
            logits = torch.clamp(logits, min=-20.0, max=20.0)
            if torch.isnan(logits).any():
                logits = torch.zeros_like(logits)
            current_values = self.critic(states)
            logits = logits - logits.mean(dim=-1, keepdim=True)
            logits = logits / (logits.std(dim=-1, keepdim=True) + 1e-8)
            # Compute losses
            dist = torch.distributions.Categorical(logits=logits)
            current_log_probs = dist.log_prob(actions)
            
            # Actor loss
            actor_loss = -(advantages * current_log_probs).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(current_values.squeeze(), returns)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            # Clear memory
            self.memory.clear()

    def save_model(self, filename):
        if not filename.endswith('.pt'):
            filename += '.pt'
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'state_mean': self.state_mean,
            'state_std': self.state_std,
        }, os.path.join(self.model_path, filename))
    
    def load_model(self, filename):
        try:
            checkpoint = torch.load(os.path.join(self.model_path, filename))
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.state_mean = checkpoint['state_mean']
            self.state_std = checkpoint['state_std']
            return True
        except:
            return False
    
    def calculate_reward(self, action, observation, info):
        """Calculate normalized reward based on action and observation."""
        
        if action is None:
            return -1.0  # Scaled from -10.0 to -1.0

        reward = 0.0
        current_filled_ratio = self.calculate_filled_ratio(observation)
        filled_ratio_change = current_filled_ratio - self.prev_filled_ratio

        # Get action details
        stock = observation["stocks"][action["stock_idx"]]
        pos_x, pos_y = action["position"]
        size_w, size_h = action["size"]
        stock_w, stock_h = self._get_stock_size_(stock)
        piece_area = size_w * size_h

        # 1. Scattered Placement Penalty (Scaled)
        adjacent_count = self._count_adjacent_pieces(stock, pos_x, pos_y, size_w, size_h)
        if adjacent_count == 0 and np.sum(stock != -1) > 0:
            distance = self._get_distance_to_filled(stock, pos_x, pos_y)
            # Original: scatter_penalty = -5.0 * min(8, (1.5 ** min(distance, 4)))
            # Scaled: divide by 25.0
            scatter_penalty = -5.0 * min(8, (1.5 ** min(distance, 4))) / 25.0  # Scales to ~-1.0
            reward += scatter_penalty

        # 2. Adjacency Reward (Scaled)
        if adjacent_count > 0:
            # Original: adjacency_reward = 2.0 * min(5, (1.2 ** min(adjacent_count, 4)))
            # Scaled: divide by 4.0
            adjacency_reward = 2.0 * min(5, (1.2 ** min(adjacent_count, 4))) / 4.0  # Scales to ~+2.5 /4 = +0.625
            reward += adjacency_reward

        # 3. Top-Down Fill Violation Penalty (Scaled)
        empty_cells_above = 0
        for x in range(pos_x, pos_x + size_w):
            for y in range(0, pos_y):
                if stock[y, x] == -1:
                    empty_cells_above += 1
        if empty_cells_above > 0:
            # Original: vertical_penalty = -1.0 * min(10, (1.2 ** min(empty_cells_above, 5)))
            # Scaled: divide by 10.0
            vertical_penalty = -1.0 * min(10, (1.2 ** min(empty_cells_above, 5))) / 10.0  # Scales to ~-1.0
            reward += vertical_penalty

        # 4. Edge and Corner Bonuses (Scaled)
        if pos_x == 0 and pos_y == 0:
            reward += 8.0 / 8.0  # +1.0
        elif pos_x == 0 or pos_y == 0:
            reward += 4.0 / 8.0  # +0.5

        # 5. New Stock Penalty (Scaled)
        if np.sum(stock != -1) == piece_area:
            used_stocks = sum(1 for s in observation["stocks"] if np.any(s != -1))
            if piece_area < stock_w * stock_h * 0.3:
                # Original: new_stock_penalty = -5.0 * min(8, (1.2 ** min(used_stocks, 5)))
                # Scaled: divide by 25.0
                new_stock_penalty = -5.0 * min(8, (1.2 ** min(used_stocks, 5))) / 25.0  # Scales to ~-1.6
                reward += new_stock_penalty

        # 6. Area Utilization (Scaled)
        # Assuming filled_ratio_change is between -1 and +1
        # Original: filled_ratio_reward = filled_ratio_change * 30.0
        # Scaled: multiply by 3.0
        filled_ratio_reward = filled_ratio_change * 3.0  # Scales to [-3.0, +3.0]
        reward += filled_ratio_reward

        # 7. Isolation Penalty (Scaled)
        empty_neighbors = self._count_empty_neighbors(stock, pos_x, pos_y, size_w, size_h)
        if empty_neighbors > 0:
            # Original: isolation_penalty = -2.0 * min(5, (1.2 ** min(empty_neighbors, 4)))
            # Scaled: divide by 10.0
            isolation_penalty = -2.0 * min(5, (1.2 ** min(empty_neighbors, 4))) / 10.0  # Scales to ~-1.0
            reward += isolation_penalty

        # Update previous filled ratio
        self.prev_filled_ratio = current_filled_ratio

        # Optionally, clip the reward to a specific range to prevent extreme values
        reward = max(-5.0, min(5.0, reward))  # Example clipping between -5 and +5

        return reward


    def _count_adjacent_pieces(self, stock, pos_x, pos_y, size_w, size_h):
        """Enhanced adjacent piece counting with stronger weights for direct adjacency"""
        count = 0
        padding = 1
        
        # Check all sides and corners
        for dx in range(-padding, size_w + padding):
            for dy in range(-padding, size_h + padding):
                # Skip the cells where the piece will be placed
                if 0 <= dx < size_w and 0 <= dy < size_h:
                    continue
                    
                check_x = pos_x + dx
                check_y = pos_y + dy
                
                if (0 <= check_x < stock.shape[1] and 
                    0 <= check_y < stock.shape[0]):
                    if stock[check_y, check_x] != -1:
                        # Much higher weight for direct adjacency vs diagonal
                        if dx in [0, size_w] or dy in [0, size_h]:
                            count += 2.0  # Increased from 1.5
                        else:
                            count += 0.25  # Reduced diagonal weight
        
        return count

    def calculate_stock_filled_ratio(self, stock):
        """Calculate filled ratio for a single stock"""
        stock_w, stock_h = self._get_stock_size_(stock)
        total_area = stock_w * stock_h
        used_area = np.sum(stock != -1)
        return used_area / total_area


    def calculate_space_utilization(self, stock, pos_x, pos_y, size_w, size_h):
        """Calculate how efficiently the remaining space can be used"""
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Calculate remaining rectangles after placement
        remaining_areas = []
        
        # Right rectangle
        if pos_x + size_w < stock_w:
            right_w = stock_w - (pos_x + size_w)
            right_h = stock_h
            remaining_areas.append(right_w * right_h)
        
        # Top rectangle
        if pos_y + size_h < stock_h:
            top_w = size_w
            top_h = stock_h - (pos_y + size_h)
            remaining_areas.append(top_w * top_h)
        
        # Calculate utilization score
        total_remaining = sum(remaining_areas)
        stock_area = stock_w * stock_h
        used_area = size_w * size_h
        
        # Reward based on how much usable space remains
        space_efficiency = used_area / stock_area
        remaining_ratio = total_remaining / stock_area
        
        # Return weighted score
        return space_efficiency * 3.0 + remaining_ratio * 2.0

    def calculate_stock_penalty(self, observation):
        used_stocks = sum(1 for stock in observation['stocks'] if np.any(stock > 0))
        stock_penalty = -0.2 * used_stocks
        return stock_penalty

    def _is_good_pattern(self, pos_x, pos_y, size_w, size_h, stock):
        """Evaluate if the placement creates a good cutting pattern"""
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Check edge alignment
        is_edge_aligned = (pos_x == 0 or pos_x + size_w == stock_w or
                          pos_y == 0 or pos_y + size_h == stock_h)
        
        # Calculate remaining usable space
        used_area = size_w * size_h
        total_area = stock_w * stock_h
        remaining_ratio = 1 - (used_area / total_area)
        
        # Check if the remaining space is still usable
        min_dimension = min(stock_w, stock_h)
        has_usable_space = (remaining_ratio >= 0.3 and  # At least 30% space left
                           min_dimension >= min(size_w, size_h))  # Can fit similar pieces
        
        return is_edge_aligned and has_usable_space

    def _get_random_valid_action(self, observation, allow_rotation=True):
        """Get a random valid action with optional rotation"""
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            for prod in observation["products"]:
                if prod["quantity"] > 0:
                    # Try both normal and rotated orientations
                    orientations = [(prod["size"][0], prod["size"][1])]
                    if allow_rotation:
                        orientations.append((prod["size"][1], prod["size"][0]))
                    
                    for prod_w, prod_h in orientations:
                        for _ in range(10):
                            pos_x = np.random.randint(0, stock_w - prod_w + 1)
                            pos_y = np.random.randint(0, stock_h - prod_h + 1)
                            
                            if self._can_place_(stock, (pos_x, pos_y), (prod_w, prod_h)):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": (prod_w, prod_h),
                                    "position": (pos_x, pos_y)
                                }
        
        return None

    def evaluate_cutting_pattern(self, observation, action, info):
        """Evaluate the quality of a cutting pattern"""
        if action is None:
            return None
        
        stock = observation['stocks'][action['stock_idx']]
        stock_w, stock_h = self._get_stock_size_(stock)
        pos_x, pos_y = action['position']
        size_w, size_h = action['size']
        
        # Calculate metrics
        edge_contact = 0
        if pos_x == 0 or pos_x + size_w == stock_w:
            edge_contact += 1
        if pos_y == 0 or pos_y + size_h == stock_h:
            edge_contact += 1
        
        is_corner = (pos_x == 0 or pos_x + size_w == stock_w) and \
                   (pos_y == 0 or pos_y + size_h == stock_h)
        
        self.metrics.episode_metrics['edge_utilization'].append(edge_contact)
        self.metrics.episode_metrics['corner_placements'].append(int(is_corner))
        
        return {
            'edge_contact': edge_contact,
            'is_corner': is_corner,
            'piece_area': size_w * size_h,
            'position_quality': pos_x == 0 or pos_y == 0  # Preference for edge placement
        }
    def plot_training_progress(self):
        """Enhanced plot training metrics with episode tracking"""
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(221)
        plt.plot(self.metrics.episode_history['episode_numbers'], 
                 self.metrics.episode_history['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # Plot filled ratios
        plt.subplot(222)
        plt.plot(self.metrics.episode_history['episode_filled_ratios'])
        plt.title('Filled Ratios')
        plt.xlabel('Episode')
        plt.ylabel('Ratio')
        
        # Plot edge utilization
        plt.subplot(223)
        plt.plot(self.metrics.episode_metrics['edge_utilization'])
        plt.title('Edge Utilization')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        
        # Plot corner placements
        plt.subplot(224)
        plt.plot(self.metrics.episode_metrics['corner_placements'])
        plt.title('Corner Placements')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_path, 'training_progress.png'))
        plt.close()

    def log_episode_summary(self, steps, filled_ratio, episode_reward, observation):
        """Enhanced log episode summary with metrics tracking"""
        # Check if all products have been processed
        remaining_products = sum(prod["quantity"] for prod in observation["products"])
        
        if remaining_products == 0:  # Only log when all products are placed
            # Add episode data to metrics
            current_episode = len(self.metrics.episode_history['episode_numbers'])
            self.metrics.add_episode_data(current_episode, filled_ratio, episode_reward)
            
    def can_place_product(self, stock_grid, product_size):
        """
        Check if the product can be placed in the stock grid.
        """
        stock_h, stock_w = stock_grid.shape
        product_w, product_h = product_size

        rotations = [(product_w, product_h), (product_h, product_w)]

        for rot_w, rot_h in rotations:
            if rot_w > stock_w or rot_h > stock_h:
                continue

            for y in range(stock_h - rot_h + 1):
                for x in range(stock_w - rot_w + 1):
                    region = stock_grid[y:y + rot_h, x:x + rot_w]
                    if np.all(region == -1):
                        return True
        return False
    def find_best_fitting_stock(self, observation, product_size):
        """
        Optimized stock selection prioritizing existing partially filled stocks.
        Selects the stock that can accommodate the product with minimal residual space.
        Only selects a new stock if no existing stock can accommodate the product.
        
        Parameters:
            observation (dict): Contains the current state, including stocks.
            product_size (tuple): (width, height) of the product.

        Returns:
            int or None: Index of the best-fitting stock, or None if no suitable stock is found.
        """
        best_stock_idx = None
        minimal_residual = float('inf')
        
        product_w, product_h = product_size

        # Iterate through all stocks to find suitable partially filled stocks
        for idx, stock in enumerate(observation["stocks"]):
            stock_grid = stock  # Assuming stock is a 2D numpy array
            stock_w, stock_h = self._get_stock_size_(stock)
            used_area = np.sum(stock != -1)
            total_area = stock_w * stock_h
            free_area = total_area - used_area

            # Only consider partially filled stocks
            if 0 < used_area < total_area:
                # Check if the product can fit in this stock
                if self.can_place_product(stock_grid, product_size):
                    residual_space = self.calculate_residual_space(stock_grid, product_size)

                    # Select the stock with minimal residual space
                    if residual_space < minimal_residual:
                        minimal_residual = residual_space
                        best_stock_idx = idx

        # If a suitable partially filled stock is found, return it
        if best_stock_idx is not None:
            return best_stock_idx

        # If no suitable partially filled stock, consider empty stocks
        empty_stocks = [idx for idx, stock in enumerate(observation["stocks"]) 
                    if np.sum(stock != -1) == 0]
        smallest_sufficient_stock = None
        smallest_sufficient_area = float('inf')
        
        for idx in empty_stocks:
            stock = observation["stocks"][idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_area = stock_w * stock_h
            
            # Check if the product can fit in this empty stock
            if self.can_place_product(stock, product_size):
                if stock_area < smallest_sufficient_area:
                    smallest_sufficient_area = stock_area
                    smallest_sufficient_stock = idx

        return smallest_sufficient_stock


    def _count_small_products(self, stock):
        """Count number of small products (isolated pieces) in the stock"""
        unique_products = np.unique(stock[stock != -1])
        small_products = 0
        
        for prod_id in unique_products:
            prod_mask = stock == prod_id
            if np.sum(prod_mask) < 20:  # Consider products smaller than 20 cells
                small_products += 1
            
        return small_products

    def evaluate_placement_pattern(self, stock, pos_x, pos_y, size_w, size_h):
        """Enhanced pattern evaluation with better small product handling"""
        stock_w, stock_h = self._get_stock_size_(stock)
        score = 0
        piece_area = size_w * size_h
        is_small_piece = piece_area < 20
        
        # Special handling for small pieces
        if is_small_piece:
            # 1. Strongly encourage corner and edge placements (50%)
            if pos_x == 0 and pos_y == 0:  # Corner
                score += 3.0
            elif pos_x == 0 or pos_y == 0:  # Edge
                score += 2.0
            
            # 2. Reward clustering with other small pieces (30%)
            nearby_small = self._count_nearby_small_pieces(stock, pos_x, pos_y, size_w, size_h)
            score += nearby_small * 1.5
            
            # 3. Penalize scattered placements (20%)
            distance_to_filled = self._get_distance_to_filled(stock, pos_x, pos_y)
            score -= distance_to_filled * 0.5
        else:
            # Logic for larger pieces
            # 1. Edge alignment (40%)
            if pos_x == 0 or pos_x + size_w == stock_w:
                score += 2.0
            if pos_y == 0 or pos_y + size_h == stock_h:
                score += 2.0
            
            # 2. Corner bonus (30%)
            if (pos_x == 0 or pos_x + size_w == stock_w) and \
               (pos_y == 0 or pos_y + size_h == stock_h):
                score += 3.0
            
            # 3. Space utilization (30%)
            used_space = np.sum(stock != -1)
            total_space = stock_w * stock_h
            utilization = used_space / total_space
            score += utilization * 2.0
        
        return score
    def calculate_residual_space(self, stock_grid, product_size):
        """
        Calculate residual free space after placing the product.

        Parameters:
            stock_grid (np.ndarray): 2D array representing the stock.
            product_size (tuple): (width, height) of the product.

        Returns:
            int: Residual free area.
        """
        stock_h, stock_w = stock_grid.shape
        used_area = np.sum(stock_grid != -1)
        total_area = stock_w * stock_h
        product_area = product_size[0] * product_size[1]
        residual_space = total_area - (used_area + product_area)

        return residual_space
    def _count_nearby_small_pieces(self, stock, pos_x, pos_y, size_w, size_h):
        """Count small pieces within a 3-cell radius"""
        padding = 3
        x_start = max(0, pos_x - padding)
        x_end = min(stock.shape[1], pos_x + size_w + padding)
        y_start = max(0, pos_y - padding)
        y_end = min(stock.shape[0], pos_y + size_h + padding)
        
        region = stock[y_start:y_end, x_start:x_end]
        unique_products = np.unique(region[region != -1])
        
        small_pieces = 0
        for prod_id in unique_products:
            prod_mask = region == prod_id
            if np.sum(prod_mask) < 20:
                small_pieces += 1
            
        return small_pieces

    def _get_distance_to_filled(self, stock, pos_x, pos_y):
        """Calculate Manhattan distance to nearest filled cell"""
        filled_positions = np.where(stock != -1)
        if len(filled_positions[0]) == 0:
            return 0
        
        distances = abs(filled_positions[0] - pos_y) + abs(filled_positions[1] - pos_x)
        return np.min(distances) if len(distances) > 0 else 0

    def _get_structured_placement(self, observation):
        """Get structured placement for initial or few remaining products"""
        # Find the largest product first
        largest_product = None
        largest_area = 0
        for prod in observation["products"]:
            if prod["quantity"] > 0:
                area = prod["size"][0] * prod["size"][1]
                if area > largest_area:
                    largest_area = area
                    largest_product = prod
        
        if not largest_product:
            return None
        
        # Try to place in the first available stock
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            
            # If stock is empty, try corners first
            if np.all(stock == -1):
                # Try corners in this order: top-left, top-right, bottom-left, bottom-right
                corners = [
                    (0, 0),
                    (stock_w - largest_product["size"][0], 0),
                    (0, stock_h - largest_product["size"][1]),
                    (stock_w - largest_product["size"][0], stock_h - largest_product["size"][1])
                ]
                
                for pos_x, pos_y in corners:
                    if self._can_place_(stock, (pos_x, pos_y), largest_product["size"]):
                        return {
                            "stock_idx": stock_idx,
                            "size": largest_product["size"],
                            "position": (pos_x, pos_y)
                        }
            
            # If no corners available, try edges
            edges = []
            # Top edge
            for x in range(0, stock_w - largest_product["size"][0] + 1):
                edges.append((x, 0))
            # Left edge
            for y in range(0, stock_h - largest_product["size"][1] + 1):
                edges.append((0, y))
            # Bottom edge
            for x in range(0, stock_w - largest_product["size"][0] + 1):
                edges.append((x, stock_h - largest_product["size"][1]))
            # Right edge
            for y in range(0, stock_h - largest_product["size"][1] + 1):
                edges.append((stock_w - largest_product["size"][0], y))
            
            # Try each edge position
            for pos_x, pos_y in edges:
                if self._can_place_(stock, (pos_x, pos_y), largest_product["size"]):
                    return {
                        "stock_idx": stock_idx,
                        "size": largest_product["size"],
                        "position": (pos_x, pos_y)
                    }
        
        # If no structured placement is possible, return None
        return None

    def _get_largest_product(self, observation):
        """Find the product with the largest area that still has remaining quantity"""
        largest_product = None
        largest_area = 0
        
        for product in observation["products"]:
            if product["quantity"] > 0:  # Only consider products with remaining quantity
                area = product["size"][0] * product["size"][1]
                if area > largest_area:
                    largest_area = area
                    largest_product = product
        # print(largest_product)
        return largest_product

    def _get_greedy_action(self, observation):
        """Implements greedy placement strategy with enhanced rotation logic"""
        best_action = None
        best_score = float('-inf')
        max_attempts = 1000
        attempts = 0
        
        # Sort products by area (largest first)
        products = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],
            reverse=True
        )
        
        if not products:
            return None
        
        # Try each product
        for prod in products:
            # Always try both orientations for each product
            orientations = [
                list(prod["size"]),  # Original orientation
                [prod["size"][1], prod["size"][0]]  # Rotated orientation
            ]
            
            for prod_size in orientations:
                prod_w, prod_h = prod_size
                
                # Try each stock
                for stock_idx, stock in enumerate(observation["stocks"]):
                    if attempts >= max_attempts:
                        return best_action if best_action is not None else None
                        
                    stock_w, stock_h = self._get_stock_size_(stock)
                    
                    # Skip if product can't fit in either orientation
                    if stock_w < prod_w or stock_h < prod_h:
                        continue
                    
                    # Try placement at each position
                    for y in range(stock_h - prod_h + 1):
                        for x in range(stock_w - prod_w + 1):
                            attempts += 1
                            if attempts >= max_attempts:
                                return best_action if best_action is not None else None
                                
                            if self._can_place_(stock, (x, y), prod_size):
                                score = self._calculate_placement_score(
                                    stock, x, y, prod_w, prod_h, stock_w, stock_h
                                )
                                
                                # Give small bonus for rotated placement if it improves utilization
                                if prod_size != list(prod["size"]):
                                    if score > 0:  # Only boost positive scores
                                        score *= 1.05
                                
                                if score > best_score:
                                    best_score = score
                                    best_action = {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)
                                    }
        
        return best_action

    def _calculate_placement_score(self, stock, pos_x, pos_y, prod_w, prod_h, stock_w, stock_h):
        score = 0
        
        # Calculate center coordinates
        center_x = stock_w // 2
        center_y = stock_h // 2
        
        # Calculate distance from placement to center
        dist_to_center = abs(pos_x + prod_w/2 - center_x) + abs(pos_y + prod_h/2 - center_y)
        
        # First check if stock is already in use
        stock_utilization = np.sum(stock != -1) / (stock_w * stock_h)
        
        if stock_utilization > 0:
            # Priority 1: Complete partially filled stocks
            score += 50
            
            # Priority 2: Heavily reward multiple adjacencies
            adjacent_count = self._count_adjacent_pieces(stock, pos_x, pos_y, prod_w, prod_h)
            if adjacent_count >= 2:
                # Exponential bonus for 2 or more adjacencies
                adjacency_bonus = 60 * (2.0 ** (adjacent_count - 1))  # Significantly increased multiplier
                score += adjacency_bonus
            else:
                # Linear bonus for single adjacency
                score += 30 * adjacent_count
            
            # Additional super bonus for 3 or more adjacencies
            if adjacent_count >= 3:
                score += 100  # Extra bonus for excellent fits
            
            # Priority 3: Perfect fit bonus
            if self._is_perfect_fit(stock, pos_x, pos_y, prod_w, prod_h):
                score += 50  # Increased perfect fit bonus
            
            # Priority 4: Center filling for better space utilization
            if stock_utilization < 0.6:
                center_bonus = max(0, 25 - dist_to_center)
                score += center_bonus
            
        else:
            # For empty stocks, still prefer adjacencies over corners
            if stock_utilization < 0.3:
                # Further reduced corner bonuses
                if (pos_x == 0 or pos_x + prod_w == stock_w) and (pos_y == 0 or pos_y + prod_h == stock_h):
                    score += 10  # Reduced from 15
                elif pos_x == 0 or pos_x + prod_w == stock_w or pos_y == 0 or pos_y + prod_h == stock_h:
                    score += 5   # Reduced from 10
        
        # Increased penalties for waste creation
        gaps_above = self._count_gaps_above(stock, pos_x, pos_y, prod_w)
        score -= gaps_above * 15  # Increased penalty
        
        # Stronger penalty for creating isolated areas
        isolated_area_penalty = self._calculate_isolation_penalty(stock, pos_x, pos_y, prod_w, prod_h)
        score -= isolated_area_penalty * 20  # Increased penalty
        
        # Enhanced penalty for creating small gaps
        small_gap_penalty = self._calculate_small_gap_penalty(stock, pos_x, pos_y, prod_w, prod_h)
        score -= small_gap_penalty * 25  # Increased penalty
        
        return score

    def _is_perfect_fit(self, stock, pos_x, pos_y, prod_w, prod_h):
        """Enhanced perfect fit detection with better alignment checking"""
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        perfect_fits = 0
        alignment_quality = 0
        
        for dx, dy in directions:
            check_x = pos_x + dx * prod_w
            check_y = pos_y + dy * prod_h
            
            if (0 <= check_x < stock.shape[1] and 
                0 <= check_y < stock.shape[0]):
                # Check for full side alignment
                if dx != 0:  # Horizontal alignment
                    aligned_cells = sum(1 for y in range(pos_y, pos_y + prod_h)
                                     if 0 <= check_x < stock.shape[1] and
                                     stock[y, check_x] != -1)
                    if aligned_cells == prod_h:
                        perfect_fits += 1
                        alignment_quality += aligned_cells / prod_h
                
                elif dy != 0:  # Vertical alignment
                    aligned_cells = sum(1 for x in range(pos_x, pos_x + prod_w)
                                     if 0 <= check_y < stock.shape[0] and
                                     stock[check_y, x] != -1)
                    if aligned_cells == prod_w:
                        perfect_fits += 1
                        alignment_quality += aligned_cells / prod_w
        
        return perfect_fits >= 2 or alignment_quality >= 1.5  # More flexible perfect fit criteria

    def _calculate_small_gap_penalty(self, stock, pos_x, pos_y, prod_w, prod_h):
        """Calculate penalty for creating small gaps between pieces"""
        penalty = 0
        padding = 2  # Check 2 cells around the placement
        
        for dx in range(-padding, prod_w + padding):
            for dy in range(-padding, prod_h + padding):
                check_x = pos_x + dx
                check_y = pos_y + dy
                
                if (0 <= check_x < stock.shape[1] and 
                    0 <= check_y < stock.shape[0]):
                    # Check if this creates a small gap
                    if stock[check_y, check_x] == -1:
                        gap_size = self._get_empty_area_size(stock, check_x, check_y)
                        if 0 < gap_size < 4:  # Penalize very small gaps
                            penalty += 1
        
        return penalty

    def _calculate_isolation_penalty(self, stock, pos_x, pos_y, prod_w, prod_h):
        """Calculate penalty for creating small isolated areas"""
        stock_w, stock_h = self._get_stock_size_(stock)
        penalty = 0
        
        # Check surrounding areas for potential isolation
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dx, dy in directions:
            check_x = pos_x + dx * prod_w
            check_y = pos_y + dy * prod_h
            
            # Check if the adjacent area would become isolated
            if (0 <= check_x < stock_w and 0 <= check_y < stock_h):
                area_size = self._get_empty_area_size(stock, check_x, check_y)
                if 0 < area_size < prod_w * prod_h:  # If area is smaller than current piece
                    penalty += 1
        
        return penalty

    def _get_empty_area_size(self, stock, start_x, start_y):
        """Calculate size of connected empty area starting from given position"""
        if start_x < 0 or start_y < 0 or start_x >= stock.shape[1] or start_y >= stock.shape[0]:
            return 0
        if stock[start_y, start_x] != -1:
            return 0
        
        visited = set()
        stack = [(start_x, start_y)]
        area = 0
        
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            
            visited.add((x, y))
            area += 1
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_x, new_y = x + dx, y + dy
                if (0 <= new_x < stock.shape[1] and 0 <= new_y < stock.shape[0] and 
                    stock[new_y, new_x] == -1 and (new_x, new_y) not in visited):
                    stack.append((new_x, new_y))
    
        return area

    def _count_gaps_above(self, stock, pos_x, pos_y, width):
        """Count empty cells above the placement position"""
        gaps = 0
        for x in range(pos_x, pos_x + width):
            for y in range(0, pos_y):
                if stock[y, x] == -1:
                    gaps += 1
        return gaps

    def _count_empty_neighbors(self, stock, pos_x, pos_y, size_w, size_h):
        """Count empty neighboring cells around the placement"""
        empty_count = 0
        padding = 1
        
        for dx in range(-padding, size_w + padding):
            for dy in range(-padding, size_h + padding):
                check_x = pos_x + dx
                check_y = pos_y + dy
                
                # Skip the cells where the piece will be placed
                if 0 <= dx < size_w and 0 <= dy < size_h:
                    continue
                    
                # Check if position is within bounds
                if (0 <= check_x < stock.shape[1] and 
                    0 <= check_y < stock.shape[0]):
                    if stock[check_y, check_x] == -1:
                        empty_count += 1
    
        return empty_count

    def calculate_filled_ratio(self, observation):
        """Calculate the correct filled ratio based only on used stocks"""
        total_used_area = 0
        total_stock_area = 0
        
        for stock in observation['stocks']:
            # Check if stock is used (has any non-negative values)
            if np.any(stock != -1):
                stock_w, stock_h = self._get_stock_size_(stock)
                stock_area = stock_w * stock_h
                used_area = np.sum(stock != -1)  # Count non-empty cells
                
                total_used_area += used_area
                total_stock_area += stock_area
        
        # Calculate ratio only if we have used stocks
        if total_stock_area > 0:
            return total_used_area / total_stock_area
        return 0.0

    def convert_action(self, action_idx, observation):
        """Convert network output to placement parameters with enhanced rotation"""
        max_stocks = len(observation["stocks"])
        # Determine if action is rotated (second half of action space)
        is_rotated = action_idx >= (max_stocks * 25)
        if is_rotated:
            action_idx -= (max_stocks * 25)
        
        stock_idx = min(action_idx // 25, max_stocks - 1)
        position = action_idx % 25
        pos_x = position // 5
        pos_y = position % 5
        
        # Find best product placement considering rotation
        best_action = None
        best_pattern_score = float('-inf')
        
        for prod in observation["products"]:
            if prod["quantity"] > 0:
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)
                
                # Try both original and rotated orientations
                orientations = [
                    list(prod["size"]),  # Original orientation
                    [prod["size"][1], prod["size"][0]]  # Rotated orientation
                ]
                
                for prod_size in orientations:
                    # Scale position to actual stock size
                    scaled_x = min(int(pos_x * stock_w / 5), stock_w - prod_size[0])
                    scaled_y = min(int(pos_y * stock_h / 5), stock_h - prod_size[1])
                    
                    if self._can_place_(stock, (scaled_x, scaled_y), prod_size):
                        return {
                            "stock_idx": stock_idx,
                            "size": prod_size,
                            "position": (scaled_x, scaled_y)
                        }
        
        # Fallback to random valid action if needed
        if best_action is None:
            best_action = self._get_structured_placement(observation)
        
        return best_action

class EpisodeEvaluator:
    def __init__(self):
        self.metrics = {
            'episode_number': 0,
            'steps': 0,
            'filled_ratio': 0.0,
            'total_reward': 0.0,
            'total_waste': 0.0,
            'waste_per_stock': 0.0,
            'num_stocks_used': 0
        }
    
    def calculate_waste(self, observation):
        """Calculate waste for all used stocks"""
        total_waste = 0
        used_stocks = 0
        
        for stock in observation['stocks']:
            if np.any(stock > 0):  # Stock has been used
                stock_area = stock.shape[0] * stock.shape[1]
                empty_area = np.sum(stock == -1)  # Count cells with -1
                waste = empty_area               # Waste is the empty area
                total_waste += waste
                used_stocks += 1
            
        return {
            'total_waste': total_waste,
            'num_stocks': used_stocks,
            'waste_per_stock': total_waste / used_stocks if used_stocks > 0 else 0
        }

    def evaluate_episode(self, observation, info, episode_data):
        """Calculate comprehensive episode quality score with correct filled ratio"""
        waste_metrics = self.calculate_waste(observation)
        
        # Calculate correct filled ratio
        total_used_area = 0
        total_stock_area = 0
        for stock in observation['stocks']:
            if np.any(stock != -1):
                stock_w, stock_h = self._get_stock_size_(stock)
                stock_area = stock_w * stock_h
                used_area = np.sum(stock != -1)
                total_used_area += used_area
                total_stock_area += stock_area
        
        correct_filled_ratio = total_used_area / total_stock_area if total_stock_area > 0 else 0.0
        
        self.metrics.update({
            'episode_number': episode_data['episode_number'],
            'steps': episode_data['steps'],
            'filled_ratio': correct_filled_ratio,
            'total_reward': episode_data['total_reward'],
            'total_waste': waste_metrics['total_waste'],
            'waste_per_stock': waste_metrics['waste_per_stock'],
            'num_stocks_used': waste_metrics['num_stocks']
        })
        
        return self.get_summary()
    
    def get_summary(self):
        """Return formatted summary of episode performance"""
        summary = f"\n{'='*20} Episode {self.metrics['episode_number']} Quality Report {'='*20}\n"
        summary += f"Steps: {self.metrics['steps']}\n"
        summary += f"Filled Ratio: {self.metrics['filled_ratio']:.3f}\n"
        summary += f"Total Waste: {self.metrics['total_waste']}\n"
        summary += f"Number of Stocks Used: {self.metrics['num_stocks_used']}\n"
        summary += f"Waste per Stock: {self.metrics['waste_per_stock']:.1f}\n"
        summary += f"Total Reward: {self.metrics['total_reward']:.2f}\n"
        summary += "="*70
        return summary

class A2CMemory:
    def __init__(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._values = []
        self._log_probs = []
        self._is_terminals = []
    
    @property
    def states(self):
        return self._states
    
    @property
    def actions(self):
        return self._actions
        
    @property
    def rewards(self):
        return self._rewards
        
    @property
    def values(self):
        return self._values
        
    @property
    def log_probs(self):
        return self._log_probs
        
    @property
    def is_terminals(self):
        return self._is_terminals
    @property 
    def size(self):
        return len(self._rewards)
    def add(self, state, action, reward, value, log_prob, done):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._values.append(value)
        self._log_probs.append(log_prob) 
        self._is_terminals.append(done)

    def clear(self):
        self._states.clear()
        self._actions.clear()
        self._rewards.clear()
        self._values.clear()
        self._log_probs.clear()
        self._is_terminals.clear()
        
    def generate_batches(self):
        if len(self._states) == 0:
            return None, None, None, None, None, None
            
        states = torch.stack(self._states)
        actions = torch.tensor(self._actions)
        rewards = torch.tensor(self._rewards)
        values = torch.tensor(self._values)
        log_probs = torch.tensor(self._log_probs)
        is_terminals = torch.tensor(self._is_terminals)
        
        return states, actions, rewards, values, log_probs, is_terminals

