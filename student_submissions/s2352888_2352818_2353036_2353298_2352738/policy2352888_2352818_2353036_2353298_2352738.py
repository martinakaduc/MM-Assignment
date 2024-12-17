from policy import Policy

class Policy2352888_2352818_2353036_2353298_2352738(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        self.policy_id = policy_id
        if policy_id == 2:
            self.policy = ColumnGenerationPolicy()
        elif policy_id == 1:
            self.policy = BestFitPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)


class ColumnGenerationPolicy(Policy):
    def __init__(self, min_waste_threshold=0.05, max_iterations=100):
        """
        Enhanced Column Generation Policy for Cutting Stock Problem
        
        Args:
            min_waste_threshold: Minimum acceptable waste ratio
            max_iterations: Maximum pattern generation iterations
        """
        self.min_waste_threshold = min_waste_threshold
        self.max_iterations = max_iterations
        self.cutting_patterns = []

    def _calculate_waste(self, stock_size, products):
        """
        Calculate waste for a potential cutting pattern
        
        Args:
            stock_size: Dimensions of stock material
            products: List of products to place
        
        Returns:
            Waste ratio (0-1)
        """
        # Calculate total product area
        total_product_area = sum(
            prod['size'][0] * prod['size'][1] * prod.get('quantity', 1) 
            for prod in products
        )
        
        # Stock area
        stock_area = stock_size[0] * stock_size[1]
        
        # Waste calculation
        waste_ratio = 1 - (total_product_area / stock_area)
        return max(0, waste_ratio)

    def _generate_comprehensive_cutting_pattern(self, stock_size, available_products):
        """
        Generate a comprehensive cutting pattern with multiple products
        
        Args:
            stock_size: Stock material dimensions
            available_products: Available products
        
        Returns:
            Cutting pattern with multiple products and minimal waste
        """
        best_pattern = None
        min_waste = float('inf')
        
        # Sort products by area (largest first) for better packing
        sorted_products = sorted(
            [p for p in available_products if p['quantity'] > 0], 
            key=lambda p: p['size'][0] * p['size'][1], 
            reverse=True
        )
        
        # Try multiple placement strategies
        for placement_strategy in [False, True]:  # With and without rotation
            for num_products in range(1, len(sorted_products) + 1):
                for product_subset in self._generate_product_combinations(
                    sorted_products, num_products
                ):
                    # Try packing the subset of products
                    pattern = self._try_pack_multiple_products(
                        stock_size, 
                        product_subset, 
                        rotate_allowed=placement_strategy
                    )
                    
                    if pattern:
                        waste = self._calculate_waste(stock_size, pattern['products'])
                        
                        # Update best pattern if waste is lower
                        if waste < min_waste:
                            min_waste = waste
                            best_pattern = pattern
                        
                        # Early exit if waste is very low
                        if waste <= self.min_waste_threshold:
                            return best_pattern
        
        return best_pattern

    def _try_pack_multiple_products(self, stock_size, products, rotate_allowed=True):
        """
        Advanced multi-product packing algorithm
        
        Args:
            stock_size: Stock material dimensions
            products: Products to pack
            rotate_allowed: Allow product rotation
        
        Returns:
            Packing configuration or None
        """
        # Track placed products and remaining space
        placed_products = []
        remaining_space = list(stock_size)
        
        # Sort products by area (largest first)
        sorted_products = sorted(
            products, 
            key=lambda p: p['size'][0] * p['size'][1], 
            reverse=True
        )
        
        for product in sorted_products:
            # Try different orientations if rotation is allowed
            orientations = [
                product['size'],
                product['size'][::-1] if rotate_allowed else None
            ]
            orientations = [o for o in orientations if o is not None]
            
            placed = False
            for orientation in orientations:
                # Check if product can fit
                if (orientation[0] <= remaining_space[0] and 
                    orientation[1] <= remaining_space[1]):
                    placed_products.append({
                        **product,
                        'size': orientation,
                        'rotated': orientation != product['size']
                    })
                    
                    # Update remaining space
                    remaining_space[0] -= orientation[0]
                    
                    # Reset width if height is exhausted
                    if remaining_space[0] == 0:
                        remaining_space[0] = stock_size[0]
                        remaining_space[1] -= orientation[1]
                    
                    placed = True
                    break
            
            # Stop if we couldn't place a product
            if not placed:
                break
        
        # Return pattern if we successfully placed multiple products
        return {
            'products': placed_products,
            'total_waste': self._calculate_waste(stock_size, placed_products)
        } if placed_products else None

    def _generate_product_combinations(self, products, num_products):
        """
        Generate combinations of products with unique sizes
        
        Args:
            products: Available products
            num_products: Number of products to combine
        
        Returns:
            Unique product combinations
        """
        # If requesting more products than available, return early
        if num_products > len(products):
            return []
        
        combinations = []
        
        def backtrack(start, current_combo):
            # If we have the desired number of products, add the combination
            if len(current_combo) == num_products:
                combinations.append(current_combo.copy())
                return
            
            # Try adding more products
            for i in range(start, len(products)):
                # Ensure unique product sizes in combination
                if not any(
                    current_combo and 
                    tuple(p['size']) == tuple(products[i]['size']) 
                    for p in current_combo
                ):
                    # Add product to current combination
                    current_combo.append(products[i])
                    
                    # Recursive call to continue building combination
                    backtrack(i + 1, current_combo)
                    
                    # Backtrack by removing last added product
                    current_combo.pop()
        
        # Start backtracking to generate combinations
        backtrack(0, [])
        
        return combinations

    def _compare_product_sizes(self, prod1, prod2):
        """
        Compare product sizes accounting for NumPy array comparison
        
        Args:
            prod1: First product
            prod2: Second product
        
        Returns:
            True if product sizes match, False otherwise
        """
        # Ensure sizes are converted to tuples for consistent comparison
        size1 = tuple(prod1['size'])
        size2 = tuple(prod2['size'])
        
        # Additionally, check for rotated sizes
        return (size1 == size2) or (size1 == size2[::-1])

    def get_action(self, observation, info):
        """
        Select product placement using column generation strategy,
        attempting to fill each stock area completely
        
        Args:
            observation: Environment observation
            info: Additional environment information
        
        Returns:
            Placement action or None if no placement is possible
        """
        # Initialize state if not already set
        if not hasattr(self, '_current_stock_idx'):
            self._current_stock_idx = 0
            self._occupied_positions = {}
        
        list_prods = observation["products"]
        stocks = observation["stocks"]
        
        # Find products with quantity > 0
        valid_products = [prod for prod in list_prods if prod["quantity"] > 0]
        if not valid_products:
            return None
        
        # Iterate through stocks starting from the last used stock index
        for stock_idx in range(self._current_stock_idx, len(stocks)):
            stock = stocks[stock_idx]
            
            # Get stock dimensions
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_size = [stock_w, stock_h]
            
            # Initialize occupied positions for this stock if not exists
            if stock_idx not in self._occupied_positions:
                self._occupied_positions[stock_idx] = set()
            
            # Find the next available position
            def is_position_available(pos, size):
                # Check if the entire product area is unoccupied
                for x in range(pos[0], pos[0] + size[0]):
                    for y in range(pos[1], pos[1] + size[1]):
                        if (x, y) in self._occupied_positions[stock_idx]:
                            return False
                return True
            
            # Generate cutting pattern for this stock
            cutting_pattern = self._generate_cutting_pattern(
                stock_size, 
                valid_products
            )
            
            if cutting_pattern:
                # Select first product from the pattern
                first_product = cutting_pattern['products'][0]
                prod_size = first_product['size']
                
                # Adjust for rotation if needed
                if first_product.get('rotated', False):
                    prod_size = prod_size[::-1]
                
                # Find the next available position
                placement_pos = None
                for y in range(stock_h - prod_size[1] + 1):
                    for x in range(stock_w - prod_size[0] + 1):
                        if is_position_available((x, y), prod_size):
                            placement_pos = (x, y)
                            break
                    if placement_pos:
                        break
                
                # If no position found, continue to next stock
                if placement_pos is None:
                    continue
                
                # Create placement dictionary
                placement = {
                    'stock_idx': stock_idx,
                    'size': prod_size,
                    'position': placement_pos
                }
                
                # Verify placement is valid
                if self._can_place_(stock, placement_pos, prod_size):
                    # Mark the area as occupied
                    for x in range(placement_pos[0], placement_pos[0] + prod_size[0]):
                        for y in range(placement_pos[1], placement_pos[1] + prod_size[1]):
                            self._occupied_positions[stock_idx].add((x, y))
                    
                    # Remove the placed product from valid products
                    for idx, p in enumerate(valid_products):
                        if (len(p['size']) == len(first_product['size']) and
                            all(p['size'][i] == first_product['size'][i] 
                                for i in range(len(p['size'])))):
                            del valid_products[idx]
                            break
                    
                    # Update the current stock index for next call
                    self._current_stock_idx = stock_idx
                    
                    return placement
        
        # Fallback to random placement if no placement is possible
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _generate_cutting_pattern(self, stock_size, available_products):
        """
        Generate an optimized cutting pattern with minimal waste
        
        Args:
            stock_size: Stock material dimensions
            available_products: Available products to place
        
        Returns:
            Optimized cutting pattern or None if no pattern found
        """
        # Sort products by area in descending order to prioritize larger items
        sorted_products = sorted(
            available_products, 
            key=lambda p: p['size'][0] * p['size'][1], 
            reverse=True
        )
        
        # Try different packing strategies
        packing_strategies = [
            (False, False),  # No rotation
            (True, False),   # Rotate products
            (False, True),   # Prioritize height
            (True, True)     # Rotate and prioritize height
        ]
        
        best_pattern = None
        min_waste = float('inf')
        
        # Iterate through packing strategies
        for rotate_products, prioritize_height in packing_strategies:
            # Create a copy of stock size to modify
            current_stock_size = list(stock_size)
            
            # Track placed products and remaining space
            placed_products = []
            remaining_space = list(current_stock_size)
            
            # Create a copy of available products to modify
            current_products = sorted_products.copy()
            
            # Try to pack products
            while current_products:
                packed = False
                
                # Sort products based on current strategy
                if prioritize_height:
                    current_products.sort(key=lambda p: p['size'][1], reverse=True)
                
                for idx, product in enumerate(current_products):
                    # Determine product size (potentially rotated)
                    prod_size = product['size']
                    if rotate_products:
                        prod_size = prod_size[::-1]
                    
                    # Check if product can fit in remaining space
                    if (prod_size[0] <= remaining_space[0] and 
                        prod_size[1] <= remaining_space[1]):
                        
                        # Place the product
                        placed_products.append({
                            **product,
                            'rotated': rotate_products
                        })
                        
                        # Update remaining space
                        remaining_space[0] -= prod_size[0]
                        
                        # Reset width if height is exhausted
                        if remaining_space[0] == 0:
                            remaining_space[0] = current_stock_size[0]
                            remaining_space[1] -= prod_size[1]
                        
                        # Remove the placed product
                        current_products.pop(idx)
                        
                        packed = True
                        break
                
                # If no product could be packed, break the loop
                if not packed:
                    break
            
            # Calculate waste for this configuration
            if placed_products:
                waste = self._calculate_waste(current_stock_size, placed_products)
                
                # Update best pattern if waste is lower
                if waste < min_waste:
                    min_waste = waste
                    best_pattern = {
                        'products': placed_products,
                        'rotated': rotate_products,
                        'waste': waste
                    }
                
                # Early exit if waste is below threshold
                if waste <= self.min_waste_threshold:
                    return best_pattern
        
        return best_pattern

    def __init__(self, min_waste_threshold=0.05, max_iterations=100):
        """
        Enhanced Column Generation Policy for Cutting Stock Problem
        
        Args:
            min_waste_threshold: Minimum acceptable waste ratio
            max_iterations: Maximum pattern generation iterations
        """
        self.min_waste_threshold = min_waste_threshold
        self.max_iterations = max_iterations
        self.cutting_patterns = []

    def _calculate_waste(self, stock_size, products):
        """
        Calculate waste for a potential cutting pattern
        
        Args:
            stock_size: Dimensions of stock material
            products: List of products to place
        
        Returns:
            Waste ratio (0-1)
        """
        # Calculate total product area
        total_product_area = sum(
            prod['size'][0] * prod['size'][1] * prod.get('quantity', 1) 
            for prod in products
        )
        
        # Stock area
        stock_area = stock_size[0] * stock_size[1]
        
        # Waste calculation
        waste_ratio = 1 - (total_product_area / stock_area)
        return max(0, waste_ratio)

    def _generate_comprehensive_cutting_pattern(self, stock_size, available_products):
        """
        Generate a comprehensive cutting pattern with multiple products
        
        Args:
            stock_size: Stock material dimensions
            available_products: Available products
        
        Returns:
            Cutting pattern with multiple products and minimal waste
        """
        best_pattern = None
        min_waste = float('inf')
        
        # Sort products by area (largest first) for better packing
        sorted_products = sorted(
            [p for p in available_products if p['quantity'] > 0], 
            key=lambda p: p['size'][0] * p['size'][1], 
            reverse=True
        )
        
        # Try multiple placement strategies
        for placement_strategy in [False, True]:  # With and without rotation
            for num_products in range(1, len(sorted_products) + 1):
                for product_subset in self._generate_product_combinations(
                    sorted_products, num_products
                ):
                    # Try packing the subset of products
                    pattern = self._try_pack_multiple_products(
                        stock_size, 
                        product_subset, 
                        rotate_allowed=placement_strategy
                    )
                    
                    if pattern:
                        waste = self._calculate_waste(stock_size, pattern['products'])
                        
                        # Update best pattern if waste is lower
                        if waste < min_waste:
                            min_waste = waste
                            best_pattern = pattern
                        
                        # Early exit if waste is very low
                        if waste <= self.min_waste_threshold:
                            return best_pattern
        
        return best_pattern

    def _try_pack_multiple_products(self, stock_size, products, rotate_allowed=True):
        """
        Advanced multi-product packing algorithm
        
        Args:
            stock_size: Stock material dimensions
            products: Products to pack
            rotate_allowed: Allow product rotation
        
        Returns:
            Packing configuration or None
        """
        # Track placed products and remaining space
        placed_products = []
        remaining_space = list(stock_size)
        
        # Sort products by area (largest first)
        sorted_products = sorted(
            products, 
            key=lambda p: p['size'][0] * p['size'][1], 
            reverse=True
        )
        
        for product in sorted_products:
            # Try different orientations if rotation is allowed
            orientations = [
                product['size'],
                product['size'][::-1] if rotate_allowed else None
            ]
            orientations = [o for o in orientations if o is not None]
            
            placed = False
            for orientation in orientations:
                # Check if product can fit
                if (orientation[0] <= remaining_space[0] and 
                    orientation[1] <= remaining_space[1]):
                    placed_products.append({
                        **product,
                        'size': orientation,
                        'rotated': orientation != product['size']
                    })
                    
                    # Update remaining space
                    remaining_space[0] -= orientation[0]
                    
                    # Reset width if height is exhausted
                    if remaining_space[0] == 0:
                        remaining_space[0] = stock_size[0]
                        remaining_space[1] -= orientation[1]
                    
                    placed = True
                    break
            
            # Stop if we couldn't place a product
            if not placed:
                break
        
        # Return pattern if we successfully placed multiple products
        return {
            'products': placed_products,
            'total_waste': self._calculate_waste(stock_size, placed_products)
        } if placed_products else None

    def _generate_product_combinations(self, products, num_products):
        """
        Generate combinations of products with unique sizes
        
        Args:
            products: Available products
            num_products: Number of products to combine
        
        Returns:
            Unique product combinations
        """
        # If requesting more products than available, return early
        if num_products > len(products):
            return []
        
        combinations = []
        
        def backtrack(start, current_combo):
            # If we have the desired number of products, add the combination
            if len(current_combo) == num_products:
                combinations.append(current_combo.copy())
                return
            
            # Try adding more products
            for i in range(start, len(products)):
                # Ensure unique product sizes in combination
                if not any(
                    current_combo and 
                    tuple(p['size']) == tuple(products[i]['size']) 
                    for p in current_combo
                ):
                    # Add product to current combination
                    current_combo.append(products[i])
                    
                    # Recursive call to continue building combination
                    backtrack(i + 1, current_combo)
                    
                    # Backtrack by removing last added product
                    current_combo.pop()
        
        # Start backtracking to generate combinations
        backtrack(0, [])
        
        return combinations

    def _compare_product_sizes(self, prod1, prod2):
        """
        Compare product sizes accounting for NumPy array comparison
        
        Args:
            prod1: First product
            prod2: Second product
        
        Returns:
            True if product sizes match, False otherwise
        """
        # Ensure sizes are converted to tuples for consistent comparison
        size1 = tuple(prod1['size'])
        size2 = tuple(prod2['size'])
        
        # Additionally, check for rotated sizes
        return (size1 == size2) or (size1 == size2[::-1])

    def get_action(self, observation, info):
        """
        Select product placement using column generation strategy,
        attempting to fill each stock area completely
        
        Args:
            observation: Environment observation
            info: Additional environment information
        
        Returns:
            Placement action or None if no placement is possible
        """
        # Initialize state if not already set
        if not hasattr(self, '_current_stock_idx'):
            self._current_stock_idx = 0
            self._occupied_positions = {}
        
        list_prods = observation["products"]
        stocks = observation["stocks"]
        
        # Find products with quantity > 0
        valid_products = [prod for prod in list_prods if prod["quantity"] > 0]
        if not valid_products:
            return None
        
        # Iterate through stocks starting from the last used stock index
        for stock_idx in range(self._current_stock_idx, len(stocks)):
            stock = stocks[stock_idx]
            
            # Get stock dimensions
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_size = [stock_w, stock_h]
            
            # Initialize occupied positions for this stock if not exists
            if stock_idx not in self._occupied_positions:
                self._occupied_positions[stock_idx] = set()
            
            # Find the next available position
            def is_position_available(pos, size):
                # Check if the entire product area is unoccupied
                for x in range(pos[0], pos[0] + size[0]):
                    for y in range(pos[1], pos[1] + size[1]):
                        if (x, y) in self._occupied_positions[stock_idx]:
                            return False
                return True
            
            # Generate cutting pattern for this stock
            cutting_pattern = self._generate_cutting_pattern(
                stock_size, 
                valid_products
            )
            
            if cutting_pattern:
                # Select first product from the pattern
                first_product = cutting_pattern['products'][0]
                prod_size = first_product['size']
                
                # Adjust for rotation if needed
                if first_product.get('rotated', False):
                    prod_size = prod_size[::-1]
                
                # Find the next available position
                placement_pos = None
                for y in range(stock_h - prod_size[1] + 1):
                    for x in range(stock_w - prod_size[0] + 1):
                        if is_position_available((x, y), prod_size):
                            placement_pos = (x, y)
                            break
                    if placement_pos:
                        break
                
                # If no position found, continue to next stock
                if placement_pos is None:
                    continue
                
                # Create placement dictionary
                placement = {
                    'stock_idx': stock_idx,
                    'size': prod_size,
                    'position': placement_pos
                }
                
                # Verify placement is valid
                if self._can_place_(stock, placement_pos, prod_size):
                    # Mark the area as occupied
                    for x in range(placement_pos[0], placement_pos[0] + prod_size[0]):
                        for y in range(placement_pos[1], placement_pos[1] + prod_size[1]):
                            self._occupied_positions[stock_idx].add((x, y))
                    
                    # Remove the placed product from valid products
                    for idx, p in enumerate(valid_products):
                        if (len(p['size']) == len(first_product['size']) and
                            all(p['size'][i] == first_product['size'][i] 
                                for i in range(len(p['size'])))):
                            del valid_products[idx]
                            break
                    
                    # Update the current stock index for next call
                    self._current_stock_idx = stock_idx
                    
                    return placement
        
        # Fallback to random placement if no placement is possible
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _generate_cutting_pattern(self, stock_size, available_products):
        """
        Generate an optimized cutting pattern with minimal waste
        
        Args:
            stock_size: Stock material dimensions
            available_products: Available products to place
        
        Returns:
            Optimized cutting pattern or None if no pattern found
        """
        # Sort products by area in descending order to prioritize larger items
        sorted_products = sorted(
            available_products, 
            key=lambda p: p['size'][0] * p['size'][1], 
            reverse=True
        )
        
        # Try different packing strategies
        packing_strategies = [
            (False, False),  # No rotation
            (True, False),   # Rotate products
            (False, True),   # Prioritize height
            (True, True)     # Rotate and prioritize height
        ]
        
        best_pattern = None
        min_waste = float('inf')
        
        # Iterate through packing strategies
        for rotate_products, prioritize_height in packing_strategies:
            # Create a copy of stock size to modify
            current_stock_size = list(stock_size)
            
            # Track placed products and remaining space
            placed_products = []
            remaining_space = list(current_stock_size)
            
            # Create a copy of available products to modify
            current_products = sorted_products.copy()
            
            # Try to pack products
            while current_products:
                packed = False
                
                # Sort products based on current strategy
                if prioritize_height:
                    current_products.sort(key=lambda p: p['size'][1], reverse=True)
                
                for idx, product in enumerate(current_products):
                    # Determine product size (potentially rotated)
                    prod_size = product['size']
                    if rotate_products:
                        prod_size = prod_size[::-1]
                    
                    # Check if product can fit in remaining space
                    if (prod_size[0] <= remaining_space[0] and 
                        prod_size[1] <= remaining_space[1]):
                        
                        # Place the product
                        placed_products.append({
                            **product,
                            'rotated': rotate_products
                        })
                        
                        # Update remaining space
                        remaining_space[0] -= prod_size[0]
                        
                        # Reset width if height is exhausted
                        if remaining_space[0] == 0:
                            remaining_space[0] = current_stock_size[0]
                            remaining_space[1] -= prod_size[1]
                        
                        # Remove the placed product
                        current_products.pop(idx)
                        
                        packed = True
                        break
                
                # If no product could be packed, break the loop
                if not packed:
                    break
            
            # Calculate waste for this configuration
            if placed_products:
                waste = self._calculate_waste(current_stock_size, placed_products)
                
                # Update best pattern if waste is lower
                if waste < min_waste:
                    min_waste = waste
                    best_pattern = {
                        'products': placed_products,
                        'rotated': rotate_products,
                        'waste': waste
                    }
                
                # Early exit if waste is below threshold
                if waste <= self.min_waste_threshold:
                    return best_pattern
        
        return best_pattern

    
class BestFitPolicy(Policy):
    
    def __init__(self):
        pass

    def get_action(self, observation, info):
        # Lấy danh sách sản phẩm và sắp xếp theo diện tích giảm dần
        sorted_products = sorted(
            observation["products"],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )

        # Duyệt qua từng sản phẩm đã sắp xếp
        for product in sorted_products:
            if product["quantity"] > 0:  # Chỉ xử lý nếu còn sản phẩm
                prod_size = product["size"]
                prod_w, prod_h = prod_size

                best_fit = None
                best_fit_position = None
                min_waste = float('inf')  # Giá trị thừa không gian nhỏ nhất

                # Duyệt qua từng tấm gỗ (stock)
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Kiểm tra không xoay sản phẩm
                    if stock_w >= prod_w and stock_h >= prod_h:
                        position = self._find_first_fit_(stock, (prod_w, prod_h))
                        if position:
                            waste = (stock_w - prod_w) * (stock_h - prod_h)
                            if waste < min_waste:
                                best_fit = (stock_idx, prod_size, position)
                                min_waste = waste

                    # Kiểm tra xoay 90 độ
                    if stock_w >= prod_h and stock_h >= prod_w:
                        position = self._find_first_fit_(stock, (prod_h, prod_w))
                        if position:
                            waste = (stock_w - prod_h) * (stock_h - prod_w)
                            if waste < min_waste:
                                best_fit = (stock_idx, prod_size[::-1], position)
                                min_waste = waste

                # Nếu tìm được tấm gỗ phù hợp nhất, trả về hành động
                if best_fit:
                    stock_idx, prod_size, position = best_fit
                    return {"stock_idx": stock_idx, "size": prod_size, "position": position}

        # Nếu không tìm được vị trí phù hợp
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_first_fit_(self, stock, prod_size):
        """Tìm vị trí đầu tiên có thể đặt sản phẩm trong tấm gỗ."""
        prod_w, prod_h = prod_size
        stock_w, stock_h = stock.shape

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None
