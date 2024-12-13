import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import scipy.optimize as optimize
from policy import Policy

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        logits = self.network(x)
        return torch.nn.functional.log_softmax(logits, dim=-1)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.network(x)  

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if policy_id == 1:
            self.state_dim = 200
            self.action_dim = 100
            self.learning_rate = 0.001
            self.gamma = 0.99
            self.epsilon = 1.0
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
            self.value_coef = 0.5
            self.entropy_coef = 0.01
            
            self.actor = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)
            self.critic = CriticNetwork(self.state_dim).to(self.device)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
            
            self.values = []
            self.states = []
            self.actions = []
            self.rewards = []
            self.log_probs = []

        elif policy_id == 2:
            self.max_iterations = 100
            self.trim_loss_threshold = 0.05
            self.optimal_patterns = None
            self.initial_patterns = []
            self.pattern_costs = []
            self.pattern_performance = []

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._get_action_policy1(observation, info)
        else:
            return self._get_action_policy2(observation, info)

    def _encode_state(self, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        
        stocks_features = []
        total_space = 0
        used_space = 0
        
        for stock in stocks:
            stock_array = np.array(stock)
            available = (stock_array == -1).sum()
            used = (stock_array >= 0).sum()
            total_space += available + used
            used_space += used
            stocks_features.extend([available/100.0, used/100.0])
            
        stocks_features.append(used_space/max(1, total_space))
        
        products_features = []
        total_demand = 0
        for prod in products:
            size = prod["size"]
            quantity = prod["quantity"]
            total_demand += quantity * size[0] * size[1]
            products_features.extend([size[0]/10.0, size[1]/10.0, quantity/10.0])
            
        stocks_features = np.array(stocks_features, dtype=np.float32)
        products_features = np.array(products_features, dtype=np.float32)
        
        stocks_features = np.pad(stocks_features, 
                            (0, max(0, 100 - len(stocks_features))), 
                            mode='constant')[:100]
        
        products_features = np.pad(products_features, 
                                (0, max(0, 100 - len(products_features))), 
                                mode='constant')[:100]
        
        state = np.concatenate([stocks_features, products_features])
        return torch.FloatTensor(state).to(self.device)

    def _get_action_policy1(self, observation, info):
        state = self._encode_state(observation)
        
        with torch.no_grad():
            state_value = self.critic(state)
            self.values.append(state_value)
        
        if np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                log_probs = self.actor(state)
                probs = torch.exp(log_probs)
                action_idx = torch.multinomial(probs, 1).item()
        
        log_prob = self.actor(state)[action_idx]
        self.log_probs.append(log_prob)
        self.states.append(state)
        self.actions.append(action_idx)
        
        return self._decode_action_policy1(action_idx, observation)

    def update_policy(self):
        if len(self.rewards) == 0:
            return
                
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device, requires_grad=True)
        
        values = torch.cat(self.values)
        log_probs = torch.stack(self.log_probs)
        
        advantages = returns - values.detach()
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_loss = []
        entropy = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage.detach())
            probs = torch.exp(log_prob)
            entropy.append(-(probs * log_prob).sum())
        
        policy_loss = torch.stack(policy_loss).sum()
        entropy_loss = -self.entropy_coef * torch.stack(entropy).sum()
        actor_loss = policy_loss + entropy_loss
        
        values = values.view(-1)
        returns = returns.view(-1)
        value_loss = self.value_coef * F.mse_loss(values, returns)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []

    def _decode_action_policy1(self, action_idx, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        
        valid_products = [(i, p) for i, p in enumerate(products) if p["quantity"] > 0]
        if not valid_products:
            return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}
        
        valid_products.sort(key=lambda x: x[1]["size"][0] * x[1]["size"][1], reverse=True)
        
        for prod_idx, prod in valid_products:
            prod_size = prod["size"]
            best_stock = None
            best_position = None
            best_size = None
            min_waste_ratio = float('inf')
            
            prod_area = prod_size[0] * prod_size[1]
            
            stock_options = []
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                stock_area = stock_w * stock_h
                
                if stock_area < prod_area:
                    continue
                    
                stock_options.append((stock_idx, stock, stock_area))
            
            stock_options.sort(key=lambda x: x[2])
            
            for stock_idx, stock, stock_area in stock_options:
                stock_w, stock_h = self._get_stock_size_(stock)
                
                orientations = [
                    prod_size,                    
                    [prod_size[1], prod_size[0]]  
                ]
                
                for current_size in orientations:
                    width, height = current_size
                    if stock_w < width or stock_h < height:
                        continue
                    
                    for x in range(stock_w - width + 1):
                        for y in range(stock_h - height + 1):
                            if self._can_place_(stock, (x, y), (width, height)):
                                waste = self._calculate_waste(stock, (x, y), (width, height))
                                waste_ratio = waste / stock_area
                                
                                if waste_ratio < min_waste_ratio:
                                    min_waste_ratio = waste_ratio
                                    best_stock = stock_idx
                                    best_position = (x, y)
                                    best_size = current_size
                    
                    if min_waste_ratio < 0.05:
                        break
                
                if min_waste_ratio < 0.05:
                    break
            
            if best_position is not None:
                return {
                    "stock_idx": best_stock,
                    "size": best_size,
                    "position": best_position
                }
        
        return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}

    def _calculate_waste(self, stock, position, size):
        x, y = position
        w, h = size
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        
        empty_area = 0
        for i in range(max(0, x-1), min(stock_w, x+w+1)):
            for j in range(max(0, y-1), min(stock_h, y+h+1)):
                if stock[i,j] == -1:
                    empty_area += 1
                    
        return empty_area - w*h

    def _get_stock_size_(self, stock):
        width = np.sum(np.any(stock != -2, axis=1))
        height = np.sum(np.any(stock != -2, axis=0))
        return width, height

    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        return np.all(stock[x:x+w, y:y+h] == -1)        
        
# BEGIN POLICY 2 #

    def solve_cutting_stock_problem(self, stock_rolls, orders):
        self.initial_patterns = self._generate_initial_patterns(stock_rolls, orders)
        
        optimal_solution = False
        iteration = 0
        
        while not optimal_solution and iteration < self.max_iterations:
            lp_solution = self._solve_linear_programming(
                stock_rolls, 
                orders, 
                self.initial_patterns
            )
            
            dual_values = self._extract_dual_values(lp_solution)
            
            new_pattern = self._solve_pattern_generation_knapsack(
                stock_rolls, 
                orders, 
                dual_values
            )
            
            if new_pattern and self._is_pattern_improving(new_pattern):
                self.initial_patterns.append(new_pattern)
            else:
                optimal_solution = True
            
            iteration += 1
        
        integer_solution = self._convert_to_integer_solution(
            lp_solution, 
            orders
        )
        
        final_solution = self._refine_solution(
            integer_solution, 
            stock_rolls, 
            orders
        )
        
        return final_solution

    def _generate_initial_patterns(self, stock_rolls, orders):
        patterns = []
        
        sorted_orders = sorted(orders, key=lambda x: x['size'][0] * x['size'][1], reverse=True)
        
        for stock in stock_rolls:
            for order in sorted_orders:
                max_cuts = stock['length'] // (order['size'][0] * order['size'][1])
                
                trim_loss = (stock['length'] - max_cuts * (order['size'][0] * order['size'][1])) / stock['length']
                
                if trim_loss <= self.trim_loss_threshold:
                    pattern = {
                        'stock_length': stock['length'],
                        'cuts': [{'length': order['size'][0] * order['size'][1], 'quantity': max_cuts}],
                        'trim_loss': trim_loss
                    }
                    patterns.append(pattern)
        
        return patterns

    def _solve_linear_programming(self, stock_rolls, orders, patterns):
        pattern_matrix = np.zeros((len(orders), len(patterns)))
        for i, order in enumerate(orders):
            for j, pattern in enumerate(patterns):
                for cut in pattern['cuts']:
                    if cut['length'] == order['size'][0] * order['size'][1]:
                        pattern_matrix[i, j] = cut['quantity']
        
        c = np.ones(len(patterns))
        
        A_ub = -pattern_matrix
        b_ub = -np.array([order['quantity'] for order in orders])
        
        bounds = [(0, None) for _ in patterns]
        
        try:
            result = optimize.linprog(
                c, 
                A_ub=A_ub, 
                b_ub=b_ub, 
                bounds=bounds, 
                method='highs'
            )
        except Exception as e:
            print(f"Linear programming error: {e}")
            result = optimize.OptimizeResult({
                'x': np.zeros(len(patterns)),
                'success': False
            })
        
        return result

    def _extract_dual_values(self, lp_solution):
        if not lp_solution.success:
            return np.zeros(len(lp_solution.x))
        
        try:
            if hasattr(lp_solution, 'ineq') and hasattr(lp_solution.ineq, 'dual'):
                return lp_solution.ineq.dual
            elif hasattr(lp_solution, 'slack'):
                return lp_solution.slack
        except Exception:
            pass
        
        return np.zeros(len(lp_solution.x))

    def _solve_pattern_generation_knapsack(self, stock_rolls, orders, dual_values):
        best_pattern = None
        max_reduced_cost = 0
        
        for stock in stock_rolls:
            current_pattern = {
                'stock_length': stock['length'],
                'cuts': [],
                'trim_loss': 0
            }
            
            remaining_length = stock['length']
            
            sorted_orders = sorted(
                enumerate(orders), 
                key=lambda x: dual_values[x[0]], 
                reverse=True
            )
            
            for idx, order in sorted_orders:
                product_area = order['size'][0] * order['size'][1]
                
                max_cuts = remaining_length // product_area
                
                if max_cuts > 0:
                    current_pattern['cuts'].append({
                        'length': product_area,
                        'quantity': max_cuts
                    })
                    
                    remaining_length -= max_cuts * product_area
                    
                    reduced_cost = 1 - sum(
                        dual_values[idx] * max_cuts 
                        for idx, _ in sorted_orders
                    )
                    
                    if reduced_cost > max_reduced_cost:
                        max_reduced_cost = reduced_cost
                        best_pattern = current_pattern
        
        return best_pattern

    def _is_pattern_improving(self, pattern):
        trim_loss = (
            pattern['stock_length'] - 
            sum(cut['length'] * cut['quantity'] for cut in pattern['cuts'])
        ) / pattern['stock_length']
        
        return trim_loss <= self.trim_loss_threshold

    def _convert_to_integer_solution(self, continuous_solution, orders):
        integer_solution = []
        
        for order in orders:
            order_size = list(order['size']) if hasattr(order['size'], '__iter__') else order['size']
            order_quantity = order['quantity']
            
            matching_cuts = []
            for cut in self.initial_patterns:
                cut_size = cut.get('size', None)
                if cut_size is not None:
                    cut_size = list(cut_size) if hasattr(cut_size, '__iter__') else cut_size
                
                if (cut_size == order_size or 
                    (isinstance(cut.get('cuts', []), list) and 
                     any(c.get('length', None) == order_size[0] * order_size[1] for c in cut.get('cuts', [])))):
                    matching_cuts.append(cut)
            
            if not matching_cuts:
                raise ValueError(f"No suitable place found {order_size}")
            
            best_cut = matching_cuts[0]
            
            pattern_quantity = order_quantity // best_cut.get('quantity', 1)
            if order_quantity % best_cut.get('quantity', 1) > 0:
                pattern_quantity += 1
            
            integer_solution.append({
                'size': order_size,
                'pattern': best_cut,
                'quantity': pattern_quantity
            })
        
        return integer_solution

    def _refine_solution(self, integer_solution, stock_rolls, orders):
        order_fulfillment = {
            order['size'][0] * order['size'][1]: 0 
            for order in orders
        }
        
        for solution in integer_solution:
            for cut in solution['pattern']['cuts']:
                order_fulfillment[cut['length']] += cut['quantity'] * solution['quantity']
        
        for order in orders:
            order_area = order['size'][0] * order['size'][1]
            if order_fulfillment[order_area] < order['quantity']:
                additional_pattern = self._find_supplementary_pattern(
                    stock_rolls, 
                    order, 
                    order['quantity'] - order_fulfillment[order_area]
                )
                
                if additional_pattern:
                    integer_solution.append(additional_pattern)
        
        return integer_solution

    def _find_supplementary_pattern(self, stock_rolls, order, remaining_quantity):
        product_area = order['size'][0] * order['size'][1]
        
        for stock in stock_rolls:
            max_cuts = stock['length'] // product_area
            
            if max_cuts > 0:
                pattern = {
                    'stock_length': stock['length'],
                    'cuts': [{
                        'length': product_area,
                        'quantity': max_cuts
                    }],
                    'trim_loss': (stock['length'] - max_cuts * product_area) / stock['length']
                }
                
                if pattern['trim_loss'] <= self.trim_loss_threshold:
                    return {
                        'pattern': pattern,
                        'quantity': min(max_cuts, remaining_quantity)
                    }
        
        return None
    def _get_action_policy2(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        
        stock_rolls = [
            {
                'length': np.prod(stock.shape),
                'cost': 1.0
            } for stock in stocks
        ]
        
        orders = [
            {
                'size': product['size'],
                'quantity': product['quantity']
            } for product in products if product['quantity'] > 0
        ]

        if self.optimal_patterns is None:
            self.optimal_patterns = self.solve_cutting_stock_problem(stock_rolls, orders)
        
        sorted_products = sorted(
            [p for p in products if p['quantity'] > 0], 
            key=lambda x: x['size'][0] * x['size'][1], 
            reverse=True
        )
        
        if not sorted_products:
            return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}
        
        current_product = sorted_products[0]
        
        matching_pattern = next(
            (pattern for pattern in self.optimal_patterns 
            if np.array_equal(pattern['size'], current_product['size'])), 
            None
        )

        if matching_pattern:
            stock_idx = matching_pattern.get('stock_idx', 0)
            placement = self._find_best_placement(stocks[stock_idx], current_product['size'])
            
            if placement is not None:
                return {
                    "stock_idx": stock_idx,
                    "size": placement['size'],
                    "position": placement['position']
                }
        
        for stock_idx, stock in enumerate(stocks):
            placement = self._find_best_placement(stock, current_product['size'])
            
            if placement is not None:
                return {
                    "stock_idx": stock_idx,
                    "size": placement['size'],
                    "position": placement['position']
                }
                  
        return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}

    def _find_best_placement(self, stock, product_size):
        stock_width = np.sum(np.any(stock != -2, axis=1))
        stock_height = np.sum(np.any(stock != -2, axis=0))
        
        orientations = [
            product_size,
            [product_size[1], product_size[0]]
        ]
        
        best_placement = None
        min_waste = float('inf')
        
        for size in orientations:
            width, height = size
            
            if width > stock_width or height > stock_height:
                continue
            
            for x in range(stock_width - width + 1):
                for y in range(stock_height - height + 1):
                    if np.all(stock[x:x+width, y:y+height] == -1):
                        waste = self._calculate_waste(stock, (x, y), (width, height))
                        
                        if waste < min_waste:
                            min_waste = waste
                            best_placement = {
                                'size': size,
                                'position': (x, y)
                            }
        
        return best_placement
