from policy import Policy
import numpy as np
from scipy.optimize import linprog


class Policy2352915_2352610_2353350_2352102(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        super().__init__()
        self.policy_id = policy_id
        
        # Parameters for Policy 1
        if policy_id == 1:
            self.max_iterations = 50
            self.max_patterns = 100
            
        # Parameters for Policy 2 (Backtracking)
        elif policy_id == 2:
            self.max_depth = 10
            self.placements = []
            self.total_waste = 0
            self.placement_count = 0
            self._dimension_cache = {}

    def _try_rotations(self, product_size):
        """Helper method to get both normal and rotated dimensions"""
        return [
            product_size,  # Original orientation
            (product_size[1], product_size[0])  # Rotated 90 degrees
        ]

    # Policy 1 Methods
    def calculate_pattern_efficiency(self, pattern, products):
        """Policy 1: Calculate the efficiency of a cutting pattern"""
        if self.policy_id != 1:
            return
        total_used_space = sum(
            pattern[i] * products[i]['size'][0] for i in range(len(products))
        )
        remaining_space = 1.0 - total_used_space
        penalty_factor = 0.5
        return total_used_space - penalty_factor * remaining_space

    def generate_initial_patterns(self, products):
        """Policy 1: Generate initial cutting patterns"""
        if self.policy_id != 1:
            return
        patterns = []
        for i, product in enumerate(products):
            if product['quantity'] > 0:
                for rotated_size in self._try_rotations(product['size']):
                    pattern = [0] * len(products)
                    pattern[i] = int(1.0 // rotated_size[0])
                    patterns.append(pattern)
        return patterns

    def solve_pricing_problem(self, duals, products):
        """Policy 1: Solve the pricing problem using column generation"""
        if self.policy_id != 1:
            return
        n = len(products)
        sizes = [product['size'][0] for product in products]
        profits = [-duals[i] for i in range(n)]

        res = linprog(
            c=profits,
            A_ub=[sizes],
            b_ub=[1.0],
            bounds=[(0, 1) for _ in range(n)],
            method='highs'
        )

        if res.success:
            fractional_pattern = res.x
            new_pattern = [int(np.floor(fractional_pattern[i] * 1.0 // sizes[i])) if sizes[i] > 0 else 0
                           for i in range(n)]
            return new_pattern
        else:
            raise ValueError("Failed to solve pricing problem.")

    def branch_and_price_solve(self, products):
        """Policy 1: Implement Branch and Price solution"""
        if self.policy_id != 1:
            return
        patterns = self.generate_initial_patterns(products)
        n_products = len(products)
        current_demand = [0] * n_products

        for _ in range(self.max_iterations):
            A = np.array(patterns).T
            b = [products[i]['quantity'] for i in range(n_products)]
            c = [1] * len(patterns)

            res = linprog(
                c=c,
                A_eq=A,
                b_eq=b,
                bounds=[(0, None)] * len(patterns),
                method='highs'
            )

            if not res.success:
                raise ValueError("Failed to solve the master problem.")

            duals = res.dual_eq
            new_pattern = self.solve_pricing_problem(duals, products)
            reduced_cost = sum(duals[i] * new_pattern[i] for i in range(n_products)) - 1
            
            if reduced_cost >= 0:
                break

            patterns.append(new_pattern)

            if len(patterns) >= self.max_patterns:
                break

        return patterns

    # Policy 2 Methods (Backtracking)
    def _calculate_stock_dimensions(self, stock):
        """
        Calculate effective stock dimensions efficiently
        """
        if self.policy_id != 2:
            return
            
        stock_hash = hash(stock.tobytes())
        if stock_hash in self._dimension_cache:
            return self._dimension_cache[stock_hash]
            
        width = np.sum(np.any(stock != -2, axis=1))
        height = np.sum(np.any(stock != -2, axis=0))
        
        self._dimension_cache[stock_hash] = (width, height)
        return width, height

    def _is_valid_placement(self, stock, product_size, x, y):
        """
        Check if placement is valid with optimizations
        """
        if self.policy_id != 2:
            return
            
        if (stock[x, y] != -1 or 
            stock[x + product_size[0] - 1, y] != -1 or
            stock[x, y + product_size[1] - 1] != -1 or
            stock[x + product_size[0] - 1, y + product_size[1] - 1] != -1):
            return False
            
        view = stock[x:x+product_size[0], y:y+product_size[1]]
        return np.all(view == -1)

    def _sort_products(self, products):
        """
        Sort products by priority for cutting
        """
        if self.policy_id != 2:
            return
            
        return sorted(enumerate(products), 
                     key=lambda x: x[1]['size'][0] * x[1]['size'][1] * x[1]['quantity'],
                     reverse=True)

    def _backtrack_placement(self, stocks, products, current_depth=0):
        """
        Recursive backtracking with optimizations
        """
        if self.policy_id != 2:
            return
            
        if current_depth >= self.max_depth or not any(p['quantity'] > 0 for p in products):
            return None
        
        sorted_products = self._sort_products(products)
        
        for stock_idx, stock in enumerate(stocks):
            stock_width, stock_height = self._calculate_stock_dimensions(stock)
            
            for product_idx, product in sorted_products:
                if product['quantity'] == 0:
                    continue
                    
                for product_size in self._try_rotations(product['size']):
                    if product_size[0] > stock_width or product_size[1] > stock_height:
                        continue
                    
                    for x in range(0, stock_width - product_size[0] + 1, 1):
                        for y in range(0, stock_height - product_size[1] + 1, 1):
                            if self._is_valid_placement(stock, product_size, x, y):
                                new_stocks = [
                                    np.copy(s) if i == stock_idx else s 
                                    for i, s in enumerate(stocks)
                                ]
                                new_products = [dict(p) for p in products]
                                
                                new_stocks[stock_idx][
                                    x:x+product_size[0], 
                                    y:y+product_size[1]
                                ] = product_idx
                                
                                new_products[product_idx]['quantity'] -= 1
                                
                                return {
                                    'stock_idx': stock_idx,
                                    'size': product_size,
                                    'position': (x, y),
                                    'depth': current_depth
                                }
                                
                    sub_placement = self._backtrack_placement(
                        stocks, products, current_depth + 1
                    )
                    if sub_placement:
                        return sub_placement
        
        return None

    def get_action(self, observation, info):
        """Main method to get cutting action based on selected policy"""
        if self.policy_id == 1:
            list_prods = observation["products"]
            stocks = observation["stocks"]
            
            for prod in list_prods:
                if prod["quantity"] > 0:
                    for prod_size in self._try_rotations(prod["size"]):
                        for stock_idx, stock in enumerate(stocks):
                            stock_w, stock_h = self._get_stock_size_(stock)
                            prod_w, prod_h = prod_size
                            
                            if stock_w < prod_w or stock_h < prod_h:
                                continue
                            
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        return {
                                            "stock_idx": stock_idx, 
                                            "size": prod_size, 
                                            "position": (x, y)
                                        }
            
            return self._get_random_action(observation)
            
        elif self.policy_id == 2:
            # Use backtracking approach for Policy 2
            stocks = [s.copy() for s in observation['stocks']]
            products = [dict(p) for p in observation['products']]
            
            action = self._backtrack_placement(stocks, products)
            
            if action:
                self.placements.append(action)
                self.placement_count += 1
                return action
            
            return {}

    def _get_random_action(self, observation):
        """Common random action method for both policies"""
        list_prods = observation["products"]
        stocks = observation["stocks"]
        
        if self.policy_id == 1:
            for _ in range(100):
                for prod in list_prods:
                    if prod["quantity"] > 0:
                        for prod_size in self._try_rotations(prod["size"]):
                            stock_idx = np.random.randint(0, len(stocks))
                            stock = stocks[stock_idx]
                            stock_w, stock_h = self._get_stock_size_(stock)
                            prod_w, prod_h = prod_size

                            if stock_w < prod_w or stock_h < prod_h:
                                continue

                            pos_x = np.random.randint(0, stock_w - prod_w + 1)
                            pos_y = np.random.randint(0, stock_h - prod_h + 1)

                            if self._can_place_(stock, (pos_x, pos_y), prod_size):
                                return {
                                    "stock_idx": stock_idx, 
                                    "size": prod_size, 
                                    "position": (pos_x, pos_y)
                                }
            raise ValueError("No valid placement found.")
            
        elif self.policy_id == 2:
            # For Policy 2, try random placements as fallback
            for _ in range(100):
                for prod in list_prods:
                    if prod["quantity"] > 0:
                        for prod_size in self._try_rotations(prod["size"]):
                            stock_idx = np.random.randint(0, len(stocks))
                            stock = stocks[stock_idx]
                            stock_w, stock_h = self._get_stock_size_(stock)
                            prod_w, prod_h = prod_size

                            if stock_w < prod_w or stock_h < prod_h:
                                continue

                            pos_x = np.random.randint(0, stock_w - prod_w + 1)
                            pos_y = np.random.randint(0, stock_h - prod_h + 1)

                            if self._can_place_(stock, (pos_x, pos_y), prod_size):
                                return {
                                    "stock_idx": stock_idx, 
                                    "size": prod_size, 
                                    "position": (pos_x, pos_y)
                                }
            return {}

    def get_performance_metrics(self):
        """Return performance metrics for Policy 2"""
        if self.policy_id == 2:
            return {
                'total_placements': self.placement_count,
                'placement_history': self.placements,
                'max_recursion_depth_used': max(
                    [p.get('depth', 0) for p in self.placements] + [0]
                ),
                'cache_hits': len(self._dimension_cache)
            }
        return {}