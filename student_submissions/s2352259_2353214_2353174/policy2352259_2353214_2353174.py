import numpy as np
from policy import Policy
from numpy.lib.stride_tricks import as_strided
class Policy2352259_2353214_2353174(Policy):
    def __init__(self, policy_id=1):
        self.policy_id = policy_id
        self.skylines = {}
        self.used_stocks = set()
        self.stock_size_cache = {}
        self.filled_area = 0
        self.last_observation = None
    
    def get_action(self, observation, info):
        if (self.last_observation is None or 
            len(observation["products"]) != len(self.last_observation["products"]) or
            any(p1["quantity"] != p2["quantity"] 
                for p1, p2 in zip(observation["products"], self.last_observation["products"]))):
            self.skylines = {}
            self.used_stocks = set()
            self.stock_size_cache = {}
            self.filled_area = 0
        
        self.last_observation = observation.copy()
        
        if self.policy_id == 1:
            return self.skyline_packing(observation, info)
        else:
            return self.level_packing(observation, info)
            
    def skyline_packing(self, observation, info):
        stocks = observation["stocks"]
        products = [
            (i, p) for i, p in enumerate(observation["products"]) if p["quantity"] > 0
        ]
        
        # Sort by area * quantity, then by max dimension
        products.sort(key=lambda x: (
            -(x[1]["size"][0] * x[1]["size"][1] * x[1]["quantity"]),
            -max(x[1]["size"]),
            -min(x[1]["size"])
        ))

        best_action = None
        best_score = float('-inf')

        for prod_idx, prod in products:
            w, h = prod["size"]
            orientations = [(w, h)]
            if w != h:  # Try both orientations if not square
                orientations.append((h, w))

            # First try used stocks
            used_stocks = [(idx, stock) for idx, stock in enumerate(stocks) 
                          if idx in self.used_stocks]
            unused_stocks = [(idx, stock) for idx, stock in enumerate(stocks) 
                           if idx not in self.used_stocks]
            
            # Try used stocks first, then unused
            for stock_group in [used_stocks, unused_stocks]:
                for stock_idx, stock in stock_group:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    
                    # Initialize skyline if not exists
                    if stock_idx not in self.skylines:
                        self.skylines[stock_idx] = np.zeros(stock_w, dtype=int)

                    for width, height in orientations:
                        if width > stock_w or height > stock_h:
                            continue

                        # Find valid positions along skyline
                        positions = self._find_skyline_positions(
                            stock_idx, stock, width, height
                        )

                        for pos in positions:
                            score = self._calculate_skyline_score(
                                stock, pos, (width, height), stock_idx
                            )
                            
                            if score > best_score:
                                best_score = score
                                best_action = {
                                    "stock_idx": stock_idx,
                                    "size": (width, height),
                                    "position": pos
                                }

                        if best_action and best_action["stock_idx"] == stock_idx:
                            # Update skyline and mark stock as used
                            self._update_skyline(stock_idx, best_action)
                            self.used_stocks.add(stock_idx)
                            return best_action

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def _find_skyline_positions(self, stock_idx, stock, width, height):
        stock_w, stock_h = self._get_stock_size_(stock)
        skyline = self.skylines[stock_idx]
        positions = []

        # Ensure we don't exceed array bounds
        for x in range(min(stock_w - width + 1, len(skyline) - width + 1)):
            # Safely get height for the region
            region = skyline[x:min(x + width, len(skyline))]
            if len(region) < width:
                continue
                
            max_height = max(region)
            y = max_height
            
            if y + height <= stock_h:
                if self._can_place_(stock, (x, y), (width, height)):
                    positions.append((x, y))
                    
        return positions

    def _can_place_(self, stock, position, size):
        x, y = position
        w, h = size
        stock_w, stock_h = self._get_stock_size_(stock)
        # Check if the placement is within bounds
        if x + w > stock_w or y + h > stock_h:
            return False
        # Check for overlap with existing placements
        area = stock[x:x+w, y:y+h]
        return np.all(area == -1)
    
    def _update_skyline(self, stock_idx, action):
        x, y = action["position"]
        w, h = action["size"]
        skyline = self.skylines[stock_idx]
        
        # Ensure we stay within bounds
        x_end = min(x + w, len(skyline))
        skyline[x:x_end] = y + h

    def _calculate_skyline_score(self, stock, pos, size, stock_idx):
        x, y = pos
        w, h = size
        stock_w, stock_h = self._get_stock_size_(stock)
        skyline = self.skylines[stock_idx]

        # Waste calculation with height penalty
        x_end = min(x + w, len(skyline))
        waste_area = sum(y - skyline[i] for i in range(x, x_end))
        height_penalty = (y + h) / stock_h  # Penalize high placements
        
        # Enhanced contact scoring
        contact = 0
        # Edge bonuses with corner rewards
        if x == 0: 
            contact += h * 1.5
            if y == 0: contact += h  # Corner bonus
        if y == 0: 
            contact += w * 1.5
            if x + w == stock_w: contact += w  # Corner bonus
        if x + w == stock_w: contact += h * 1.5
        if y + h == stock_h: contact += w * 1.2

        # Adjacent piece bonuses
        adjacent_count = 0
        for i in range(x, x + w):
            if skyline[i] == y:
                adjacent_count += 1
        contact += adjacent_count * 2.0  # Stronger bonus for adjacent pieces

        # Height balance score
        nearby_range = max(w * 2, 10)  # Look at nearby region
        x_start = max(0, x - nearby_range)
        x_finish = min(len(skyline), x + w + nearby_range)
        nearby_heights = skyline[x_start:x_finish]
        height_variation = np.std(nearby_heights) if len(nearby_heights) > 0 else 0
        balance_score = 1.0 / (1.0 + height_variation)

        # Normalize scores
        waste_score = 1.0 - (waste_area / (stock_w * stock_h))
        contact_score = contact / (4 * (w + h))  # Adjusted normalization
        
        return (
            0.5 * waste_score + 
            0.3 * contact_score +
            0.2 * balance_score -
            0.1 * height_penalty
        )
    
    def level_packing(self, observation, info):
        stocks = observation["stocks"]
        products = [
            (i, p) for i, p in enumerate(observation["products"]) if p["quantity"] > 0
        ]
        # Sort products to prioritize larger products
        products.sort(
            key=lambda x: (
                -(x[1]["size"][0] * x[1]["size"][1] * x[1]["quantity"]),
                -max(x[1]["size"]),
            )
        )

        best_action = None
        best_score = -float('inf')

        for prod_idx, prod in products:
            w, h = prod["size"]
            orientations = [(w, h)]
            if w != h:
                orientations.append((h, w))

            for idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                for width, height in orientations:
                    if width > stock_w or height > stock_h:
                        continue  # Skip orientations that don't fit
                    
                    occupancy_map = (stock != -1).astype(np.uint8)
                    free_space = self._find_positions(occupancy_map, width, height)
                    if free_space is None:
                        continue

                    positions = np.argwhere(free_space)
                    if positions.size == 0:
                        continue

                    x, y = positions[0]  # Take the first available position
                    simulated_stock = stock.copy()
                    self._place(simulated_stock, (x, y), (width, height), prod_idx)

                    # Calculate score
                    score = self._calculate_placement_score(simulated_stock)

                    # Check if this is the best action so far
                    if score > best_score:
                        best_score = score
                        best_action = {
                            "stock_idx": idx,
                            "size": (width, height),
                            "position": (x, y),
                        }

                    # Break after finding a valid placement
                    break  # Remove if you want to consider all positions

                # Early exit if a best action is found
                if best_action and best_action['stock_idx'] == idx:
                    break  # Exit orientation loop

            # Early exit if a best action is found
            if best_action:
                break  # Exit product loop

        if best_action:
            return best_action
        else:
            # No valid action found
            return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
    
    def _get_stock_size_(self, stock):
        key = id(stock)
        if key not in self.stock_size_cache:
            self.stock_size_cache[key] = (
                np.any(stock != -2, axis=1).sum(),
                np.any(stock != -2, axis=0).sum(),
            )
        return self.stock_size_cache[key]
    
    def _find_positions(self, occupancy_map, width, height):
        shape = (occupancy_map.shape[0] - width + 1, occupancy_map.shape[1] - height + 1)
        if shape[0] <= 0 or shape[1] <= 0:
            return None
        # Create a sliding window view
        window_view = as_strided(occupancy_map,
                                 shape=(shape[0], shape[1], width, height),
                                 strides=occupancy_map.strides * 2)
        # Check for all-zero windows (free space)
        free_space = np.all(window_view == 0, axis=(2, 3))
        return free_space
    
    def _place(self, stock, position, size, prod_idx=0):
        x, y = position
        w, h = size
        stock[x : x + w, y : y + h] = prod_idx  # Simulate placement

    def _calculate_placement_score(self, stock):
        # Calculate filled area based on the current stock state
        self.filled_area = np.sum(stock >= 0)

        # Get stock dimensions
        stock_w, stock_h = self._get_stock_size_(stock)
        total_area = stock_w * stock_h if stock_w and stock_h else 1

        # Calculate utilization as the ratio of filled area to total area
        utilization = self.filled_area / total_area

        # Fragmentation Penalty
        # Identify empty cells that are completely enclosed by occupied cells
        empty_cells = (stock == -1).astype(int)
        occupied_cells = (stock >= 0).astype(int)

        # Pad the array to handle edge cases without wrap-around
        padded_empty = np.pad(empty_cells, pad_width=1, mode='constant', constant_values=0)
        neighbor_counts = sum(
            np.roll(np.roll(padded_empty, dx, axis=0), dy, axis=1)
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]
        )[1:-1, 1:-1]

        # Cells surrounded by occupied cells in all four directions
        fragmented_cells = (empty_cells == 1) & (neighbor_counts == 0)
        fragmented_cell_count = np.sum(fragmented_cells)
        fragmentation_penalty = fragmented_cell_count / total_area

        # Each occupied cell can have up to 4 contacts (up, down, left, right)
        shifts = [(-1,0), (1,0), (0,-1), (0,1)]
        contact_total = 0
        for dx, dy in shifts:
            shifted = np.roll(np.roll(occupied_cells, dx, axis=0), dy, axis=1)
            # Zero out the borders to prevent wrap-around effects
            if dx == -1:
                shifted[-1, :] = 0
            if dx == 1:
                shifted[0, :] = 0
            if dy == -1:
                shifted[:, -1] = 0
            if dy == 1:
                shifted[:, 0] = 0
            contact_total += occupied_cells * shifted

        contact_score = np.sum(contact_total)
        max_contacts = 4 * self.filled_area  # Each cell can have up to 4 contacts
        normalized_contact = contact_score / max_contacts if max_contacts > 0 else 0

        # Combined score with weights
        score = (
            0.5 * utilization +
            0.3 * normalized_contact -
            0.2 * fragmentation_penalty
        )

        return score