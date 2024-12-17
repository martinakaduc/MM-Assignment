from policy import Policy


class Policy2352330_2352372_2352092_2352296_2352556(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id  # Store the policy ID
        self.current_placements = []
        self.free_rectangles = {}  # Dict[stock_idx, List[tuple]] storing (x, y, w, h)
        self.initial_stocks = []  # Store initial stocks for reset
        self.skyline = {}  # Store skyline state
    
    def _split_rectangle(self, rect, prod_size):
        """
        Guillotine split scheme to subdivide rectangle into F' and F''
        rect: tuple (x, y, width, height)
        prod_size: tuple (width, height)
        Returns: List of new rectangles after split
        """
        x, y, w, h = rect 
        prod_w, prod_h = prod_size
        new_rects = []
        
        # Split horizontally (F')
        if h > prod_h:
            new_rects.append((x, y + prod_h, w, h - prod_h))
        
        # Split vertically (F'')
        if w > prod_w:
            new_rects.append((x + prod_w, y, w - prod_w, prod_h))
        
        return new_rects
    
    def _merge_rectangles(self, stock_idx):
        """
        Merge adjacent rectangles in the stock if possible
        """
        if stock_idx not in self.free_rectangles:
            return
        
        free_rects = self.free_rectangles[stock_idx]
        merged = True
        
        while merged:
            merged = False
            i = 0
            while i < len(free_rects):
                j = i + 1
                while j < len(free_rects):
                    r1 = free_rects[i]
                    r2 = free_rects[j]
                    x1, y1, w1, h1 = r1
                    x2, y2, w2, h2 = r2
                    
                    # Try horizontal merge (same height and y-position)
                    if h1 == h2 and y1 == y2 and x1 + w1 == x2:
                        # Merge r1 and r2 horizontally
                        free_rects[i] = (x1, y1, w1 + w2, h1)
                        free_rects.pop(j)
                        merged = True
                        break
                    
                    # Try vertical merge (same width and x-position)
                    if w1 == w2 and x1 == x2 and y1 + h1 == y2:
                        # Merge r1 and r2 vertically
                        free_rects[i] = (x1, y1, w1, h1 + h2)
                        free_rects.pop(j)
                        merged = True
                        break
                    
                    j += 1
                
                if merged:
                    break
                i += 1
    
    def _find_best_area_fit(self, prod_size, stock_idx, stock):
        """
        Find the rectangle with the smallest area difference compared to the product
        Returns: rect or None if no fit found
        """
        prod_w, prod_h = prod_size
        prod_area = prod_w * prod_h
        best_rect = None
        min_area_diff = float('inf')
        
        for rect in self.free_rectangles[stock_idx]:
            x, y, w, h = rect
            if w >= prod_w and h >= prod_h:
                if self._can_place_(stock, (x, y), prod_size):
                    rect_area = w * h
                    area_diff = rect_area - prod_area
                    if area_diff < min_area_diff:
                        min_area_diff = area_diff
                        best_rect = rect
        
        return best_rect
    
    def reset(self):
        """Reset the policy to its initial state."""
        self.current_placements = []
        self.free_rectangles = {}  # Reset free rectangles

        # Initialize free rectangles for the first stock
        if self.initial_stocks:  # Assuming you have a way to access the initial stocks
            stock_w, stock_h = self._get_stock_size_(self.initial_stocks[0])
            self.free_rectangles[0] = [(0, 0, stock_w, stock_h)]  # Set the first stock
    
    def _place_guillotine(self, products, observation):
        """Guillotine placement strategy"""
        placements = []
        
        for prod in products:
            for _ in range(prod["quantity"]):
                placed = False
                
                # First try stocks that are already in use
                used_stocks = [idx for idx in range(len(observation["stocks"])) 
                             if idx in self.free_rectangles and self.free_rectangles[idx]]
                unused_stocks = [idx for idx in range(len(observation["stocks"])) 
                               if idx not in used_stocks]
                
                # Try all used stocks first
                for stock_idx in used_stocks:
                    stock = observation["stocks"][stock_idx]
                    self._merge_rectangles(stock_idx)
                    
                    # Try both orientations
                    for size in [prod["size"], prod["size"][::-1]]:
                        rect = self._find_best_area_fit(size, stock_idx, stock)
                        if rect:
                            placement = {
                                "stock_idx": stock_idx,
                                "size": size,
                                "position": (rect[0], rect[1])
                            }
                            placements.append(placement)
                            
                            # Update free rectangles
                            free_rects = self.free_rectangles[stock_idx]
                            free_rects.remove(rect)
                            free_rects.extend(self._split_rectangle(rect, size))
                            self._merge_rectangles(stock_idx)
                            
                            placed = True
                            break
                    if placed:
                        break
                
                # If not placed, try unused stocks
                if not placed:
                    for stock_idx in unused_stocks:
                        stock = observation["stocks"][stock_idx]
                        
                        # Initialize free rectangles for new stock
                        stock_w, stock_h = self._get_stock_size_(stock)
                        self.free_rectangles[stock_idx] = [(0, 0, stock_w, stock_h)]
                        
                        # Try both orientations
                        for size in [prod["size"], prod["size"][::-1]]:
                            rect = self._find_best_area_fit(size, stock_idx, stock)
                            if rect:
                                placement = {
                                    "stock_idx": stock_idx,
                                    "size": size,
                                    "position": (rect[0], rect[1])
                                }
                                placements.append(placement)
                                
                                # Update free rectangles
                                free_rects = self.free_rectangles[stock_idx]
                                free_rects.remove(rect)
                                free_rects.extend(self._split_rectangle(rect, size))
                                self._merge_rectangles(stock_idx)
                                
                                placed = True
                                break
                        if placed:
                            break
                
                if not placed:
                    break
        
        return placements

    def _find_best_skyline_position(self, prod_size, stock_idx, stock, try_rotation=True):
        """
        Find the best position using skyline strategy, optionally trying rotation
        Returns: (position, size) or (None, None) if no fit found
        """
        if stock_idx not in self.skyline:
            self._init_skyline(stock_idx, stock)

        best_position = None
        best_size = None 
        lowest_y = float('inf')

        # Try original orientation
        position = self._find_skyline_position(prod_size, stock_idx, stock)
        if position and self._can_place_(stock, position, prod_size):
            best_position = position
            best_size = prod_size
            lowest_y = position[1]

        # Try rotated orientation if allowed
        if try_rotation:
            rotated_size = prod_size[::-1]
            rotated_position = self._find_skyline_position(rotated_size, stock_idx, stock)
            if rotated_position and self._can_place_(stock, rotated_position, rotated_size):
                if rotated_position[1] < lowest_y:
                    best_position = rotated_position
                    best_size = rotated_size

        return best_position, best_size

    def _place_skyline(self, products, observation):
        """Skyline placement strategy with Width-Minimizing approach"""
        placements = []
        
        # Sort products by height (descending) to prioritize taller pieces
        sorted_products = []
        for prod in products:
            for _ in range(prod["quantity"]):
                sorted_products.append({"size": prod["size"], "original": prod["size"]})
        sorted_products.sort(key=lambda x: (x["size"][1], x["size"][0]), reverse=True)
        
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            self._init_skyline(stock_idx, stock)
            
            i = 0
            while i < len(sorted_products):
                best_x = None
                best_y = float('inf')
                best_width = float('inf')
                best_size = None
                
                # Try both orientations
                size = sorted_products[i]["size"]
                for try_size in [size, size[::-1]]:
                    prod_w, prod_h = try_size
                    
                    # Check each possible x position
                    for seg_x, seg_h, seg_w in self.skyline[stock_idx]:
                        if seg_w < prod_w:  # Skip if segment too narrow
                            continue
                            
                        # Calculate maximum height at this position
                        max_height = self._get_max_height(stock_idx, seg_x, prod_w)
                        y = max_height
                        
                        # Check if placement is valid
                        if (y + prod_h <= stock_h and 
                            self._can_place_(stock, (seg_x, y), try_size)):
                            # Calculate total width after this placement
                            total_width = self._calculate_total_width(stock_idx, seg_x, prod_w)
                            
                            # Update best position if this is better
                            if (y < best_y or 
                                (y == best_y and total_width < best_width)):
                                best_x = seg_x
                                best_y = y
                                best_width = total_width
                                best_size = try_size
                
                if best_x is not None:
                    # Place the piece
                    placement = {
                        "stock_idx": stock_idx,
                        "size": best_size,
                        "position": (best_x, best_y)
                    }
                    placements.append(placement)
                    self._update_skyline(stock_idx, (best_x, best_y), best_size)
                    sorted_products.pop(i)  # Remove the placed product
                    
                    if not sorted_products:  # All pieces placed
                        return placements
                else:
                    i += 1  # Move to next product if current one couldn't be placed
        
        return placements

    def _get_max_height(self, stock_idx, x, width):
        """Get maximum height at position x with given width"""
        max_height = 0
        for seg_x, seg_h, seg_w in self.skyline[stock_idx]:
            if seg_x + seg_w <= x:
                continue
            if seg_x >= x + width:
                break
            max_height = max(max_height, seg_h)
        return max_height

    def _calculate_total_width(self, stock_idx, x, width):
        """Calculate total width of skyline after theoretical placement"""
        max_x = 0
        for seg_x, _, seg_w in self.skyline[stock_idx]:
            max_x = max(max_x, seg_x + seg_w)
        return max(max_x, x + width)

    def get_action(self, observation, info):
        # Check if filled_ratio is 0.00 to determine if a reset is needed
        if info.get("filled_ratio", 1.00) == 0.00:
            self.reset()
            self.skyline = {}  # Reset skyline state

        # Store initial stocks on the first call
        if not self.initial_stocks:
            self.initial_stocks = observation["stocks"]

        if not self.current_placements:
            # Get products and sort by area
            products = [prod for prod in observation["products"] if prod["quantity"] > 0]
            products.sort(key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            
            # Use appropriate placement strategy based on policy_id
            if self.policy_id == 1:
                self.current_placements = self._place_guillotine(products, observation)
            else:  # policy_id == 2
                self.current_placements = self._place_skyline(products, observation)

            if not self.current_placements:
                # Return a default action if no placements are available
                return {"stock_idx": 0, "size": [0, 0], "position": (0, 0)}

        # Return the next placement
        return self.current_placements.pop(0)

    def _init_skyline(self, stock_idx, stock):
        """Initialize skyline for a new stock"""
        width, _ = self._get_stock_size_(stock)
        self.skyline[stock_idx] = [(0, 0, width)]  # [(x, height, width), ...]

    def _find_skyline_position(self, prod_size, stock_idx, stock):
        """Find the best position using skyline placement strategy"""
        if stock_idx not in self.skyline:
            self._init_skyline(stock_idx, stock)

        prod_w, prod_h = prod_size
        stock_w, stock_h = self._get_stock_size_(stock)
        best_x = None
        best_y = float('inf')
        skyline = self.skyline[stock_idx]

        for i, (x, h, w) in enumerate(skyline):
            if w >= prod_w:  # Segment is wide enough
                # Check if placing here doesn't exceed stock height
                if h + prod_h <= stock_h:
                    if h < best_y:  # Lower position is better
                        best_x = x
                        best_y = h

        return (best_x, best_y) if best_x is not None else None

    def _update_skyline(self, stock_idx, position, size):
        """Update skyline after placing a product"""
        x, y = position
        w, h = size
        skyline = self.skyline[stock_idx]
        new_skyline = []
        i = 0

        # Add segments before the placement
        while i < len(skyline) and skyline[i][0] < x:
            new_skyline.append(skyline[i])
            i += 1

        # Add the new segment
        new_skyline.append((x, y + h, w))

        # Process remaining segments
        while i < len(skyline):
            seg_x, seg_h, seg_w = skyline[i]
            if seg_x + seg_w <= x + w:
                # Skip segments completely covered by new placement
                i += 1
                continue
            
            if seg_x < x + w:
                # Adjust segment that's partially covered
                new_w = (seg_x + seg_w) - (x + w)
                new_skyline.append((x + w, seg_h, new_w))
            else:
                # Add remaining segments unchanged
                new_skyline.append(skyline[i])
            i += 1

        self.skyline[stock_idx] = new_skyline

    # Student code here
    # You can add more functions if needed
