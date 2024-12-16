from policy import Policy
import numpy as np

class Policy2353132_2152950_2353081_2353136_2352001(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.Art = True
            self.ColGenerate= False
        elif policy_id == 2:
            self.ColGenerate = True
            self.Art = False
            self.new_game = True        
        

    def get_action(self, observation, info):
        # Student code here
        if self.Art:
            return self.ArtAlgo(observation, info)
        elif self.ColGenerate:
            return self.execute_cutting_action(observation, info)
        

    def ArtAlgo(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        # Sort products by largest area first to place bigger items first
        products = sorted(products, key=lambda item: item['size'][0] * item['size'][1], reverse=True)

        # Select the first available product
        selected_product = None
        for product_idx, product in enumerate(products):
            if product["quantity"] > 0:
                selected_product = (product_idx, product)
                break

        if not selected_product:
            return None  # No products available

        product_idx, product = selected_product
        product_size = product["size"]

        best_stock_idx = None
        best_position = None
        best_size = None
        min_waste = float('inf')

        # Iterate over each stock to find free rectangles and place the product
        for stock_idx, stock in enumerate(stocks):
            # Get a list of free rectangles in the stock
            # Here, we'll find only the largest free rectangle for simplicity.
            free_rect = self.get_largest_free_rectangle(stock)
            if free_rect is not None:
                rect_x, rect_y, rect_h, rect_w = free_rect  # top-left corner (x,y) and size (h,w)

                # Try fitting product without rotation
                if product_size[0] <= rect_h and product_size[1] <= rect_w:
                    waste = (rect_h * rect_w) - (product_size[0] * product_size[1])
                    if waste < min_waste:
                        min_waste = waste
                        best_stock_idx = stock_idx
                        best_position = (rect_x, rect_y)
                        best_size = product_size

                # Try fitting product with rotation
                if product_size[1] <= rect_h and product_size[0] <= rect_w:
                    waste = (rect_h * rect_w) - (product_size[0] * product_size[1])
                    if waste < min_waste:
                        min_waste = waste
                        best_stock_idx = stock_idx
                        best_position = (rect_x, rect_y)
                        best_size = (product_size[1], product_size[0])

        if best_stock_idx is not None:
            # Return the chosen action
            return {
                "stock_idx": best_stock_idx,
                "size": best_size,
                "position": best_position,
            }

        return None

    def compute_max_placement(self, available_stock_dim, product_dim, product_quantity):
    # Calculate the maximum number of product pieces that can fit into the given stock dimensions
        max_fit_width = available_stock_dim[0] // product_dim[0]
        max_fit_height = available_stock_dim[1] // product_dim[1]
        return min(product_quantity, max_fit_width * max_fit_height)

    def generate_cut_patterns(self, item_list, raw_stock_size):
        """
        Derive possible cutting patterns based on the provided products and available stock.
        The approach sorts products by their area, then attempts to fit as many as possible,
        and subsequently tries to fill leftover space with other products.
        """
        # Sort products by descending area
        indexed_products = list(enumerate(item_list))
        ordered_products = sorted(
            indexed_products, 
            key=lambda prod: prod[1]['size'][0] * prod[1]['size'][1], 
            reverse=True
        )

        resulting_patterns = []

        # Iterate through sorted products to create patterns
        for prod_idx, product_info in ordered_products:
            # If no more quantity of this product, skip it
            if product_info['quantity'] <= 0:
                continue

            current_stock_w, current_stock_h = raw_stock_size
            p_w, p_h = product_info['size']

            # Skip if the product can't fit at all
            if p_w > current_stock_w or p_h > current_stock_h:
                continue

            # Initialize a pattern placeholder
            provisional_pattern = [0] * len(item_list)

            # Calculate initial fit
            init_pieces = self.compute_max_placement(
                (current_stock_w, current_stock_h),
                (p_w, p_h),
                product_info['quantity']
            )

            if init_pieces > 0:
                provisional_pattern[prod_idx] = init_pieces
                remaining_w = current_stock_w - (init_pieces * p_w)
                remaining_h = current_stock_h - (init_pieces * p_h)

                # Attempt to fill remaining space with other products
                for other_idx, other_prod in ordered_products:
                    if other_idx == prod_idx or other_prod['quantity'] <= 0:
                        continue

                    o_w, o_h = other_prod['size']

                    # Check feasibility in leftover space
                    if o_w <= remaining_w or o_h <= remaining_h:
                        width_fitting = remaining_w // o_w
                        height_fitting = remaining_h // o_h
                        max_additional_fit = width_fitting * height_fitting

                        fitted_units = min(other_prod['quantity'], max_additional_fit)
                        if fitted_units > 0:
                            provisional_pattern[other_idx] = fitted_units
                            remaining_w -= o_w * fitted_units
                            remaining_h -= o_h * fitted_units

                resulting_patterns.append(provisional_pattern)

        return resulting_patterns if resulting_patterns else []

    def execute_cutting_action(self, state_observation, extra_info):
        """
        Determine the next action in placing products onto stocks.
        Generates patterns if needed, tries to place them, and moves on 
        to the next stock when fully occupied or no suitable pattern is found.
        """
        # On a fresh start or a reset condition
        if self.new_game:
            if self.ColGenerate:
                self.cached_blueprints = {}
                self.exploited_patterns = {}
                self.current_stock_pointer = 0
                self.filled_areas = {}
                self.new_game = False

        available_stocks = state_observation["stocks"]
        current_products = state_observation["products"]
        total_left_products = sum([itm["quantity"] for itm in current_products])

        # Reset environment if minimal products left
        if total_left_products == 1:
            self.new_game = True

        # If no products remain that need placing
        if not any(item["quantity"] > 0 for item in current_products):
            return {
                "stock_idx": -1,
                "size": np.array([-1, -1]),
                "position": np.array([-1, -1])
            }

        # Sort stocks by area, largest first
        sorted_stocks = sorted(
            enumerate(available_stocks),
            key=lambda st: self._get_stock_size_(st[1])[0] * self._get_stock_size_(st[1])[1],
            reverse=True
        )

        # Iterate through stocks starting from the current pointer
        for s_idx, _ in sorted_stocks[self.current_stock_pointer:]:
            if s_idx not in self.filled_areas:
                self.filled_areas[s_idx] = set()

            current_dimensions = self._get_stock_size_(available_stocks[s_idx])

            # Generate patterns if not yet generated
            if not self.cached_blueprints:
                self.cached_blueprints = self.generate_cut_patterns(current_products, current_dimensions)

            # Try placing according to generated patterns
            for pattern_id, blueprint in enumerate(self.cached_blueprints):
                if pattern_id not in self.exploited_patterns:
                    for prod_id, qty_to_place in enumerate(blueprint):
                        if qty_to_place > 0 and current_products[prod_id]["quantity"] > 0:
                            # Find all feasible positions for this product
                            candidate_positions = self.discover_valid_positions(
                                available_stocks[s_idx],
                                current_products[prod_id]["size"]
                            )

                            # Choose a valid position not already used
                            for candidate_pos in candidate_positions:
                                position_key = (*candidate_pos, *current_products[prod_id]["size"])
                                if position_key not in self.filled_areas[s_idx]:
                                    self.filled_areas[s_idx].add(position_key)
                                    return {
                                        "stock_idx": s_idx,
                                        "size": np.array(current_products[prod_id]["size"]),
                                        "position": np.array(candidate_pos)
                                    }

            # If the stock is full or no pattern applies, move to the next one
            self.current_stock_pointer += 1
            if self.current_stock_pointer >= len(available_stocks):
                self.current_stock_pointer = 0
                self.cached_blueprints = self.generate_cut_patterns(current_products, current_dimensions)
                self.exploited_patterns = {}

        # If no placements possible
        return {
            "stock_idx": -1,
            "size": np.array([-1, -1]),
            "position": np.array([-1, -1])
        }

    def discover_valid_positions(self, stock_item, product_dims):
        """
        Identify all feasible positions within a given stock
        where the product can be placed without conflicts.
        """
        stk_w, stk_h = self._get_stock_size_(stock_item)
        p_w, p_h = product_dims
        valid_coords = []

        # Brute force check all coordinates for placement feasibility
        for coord_x in range(stk_w - p_w + 1):
            for coord_y in range(stk_h - p_h + 1):
                if self._can_place_(stock_item, (coord_x, coord_y), (p_w, p_h)):
                    valid_coords.append((coord_x, coord_y))

        return valid_coords

    def get_largest_free_rectangle(self, stock):
        """
        Find the largest rectangle of free cells (-1) in the stock.
        Returns (x, y, height, width) of the largest found rectangle.
        If no free rectangle is found, returns None.
        """

        # Convert stock to a binary matrix where 1 = free, 0 = occupied
        # We want 1 for free and 0 for occupied to use the largest rectangle algorithm
        binary_stock = (stock == -1).astype(int)
        rows, cols = binary_stock.shape

        # This uses the "maximal rectangle in a binary matrix" approach
        # We'll build histograms row by row.
        height_hist = [0] * cols
        max_area = 0
        best_rect = None  # (x, y, h, w)

        for x in range(rows):
            for y in range(cols):
                # Update histogram
                if binary_stock[x, y] == 1:
                    height_hist[y] += 1
                else:
                    height_hist[y] = 0

            # Compute largest rectangle for this histogram
            area, rect = self.largest_rectangle_in_histogram(height_hist, x)
            # rect will have (top_x, left_y, height, width)

            if area > max_area:
                max_area = area
                best_rect = rect

        return best_rect

    def largest_rectangle_in_histogram(self, heights, bottom_row):
        """
        Given a list of heights, find the largest rectangle area under the histogram.
        Returns (area, (top_x, left_y, height, width))

        bottom_row: the current row index in the matrix so we can deduce top_x
        """
        stack = []
        max_area = 0
        best_rect = None
        # Append a sentinel height 0 at the end
        for i, h in enumerate(heights + [0]):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                area = height * (i - index)
                if area > max_area:
                    max_area = area
                    # top_x = bottom_row - height + 1
                    # left_y = index
                    # height = height
                    # width = i - index
                    top_x = bottom_row - height + 1
                    left_y = index
                    best_rect = (top_x, left_y, height, i - index)
                start = index
            stack.append((start, h))

        return max_area, best_rect

    # Student code here
    # You can add more functions if needed
