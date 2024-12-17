from policy import Policy
import numpy as np

class Policy2311987_2313217_2313339_2313648_2311566(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = BrandAndBoundPolicy()
        elif policy_id == 2:
            self.policy = SkylinePlacementPolicy()


    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed
"""BrandAndBoundPolicy
This class implements a placement strategy using the branch-and-bound optimization approach, aiming to minimize waste and maximize the utilization of stock space. 
It focuses on finding the best arrangements for products within the available stocks.

Stock Selection: Chooses the best available stock by considering its area and aspect ratio.
Placement Optimization: Uses a systematic approach to try different placements for products within a stock, aiming to minimize wasted space.
Queue Management: Maintains a queue of planned placements for sequential execution.
Waste Calculation: Evaluates waste in surrounding regions after each placement to optimize future decisions.
Thresholds for Stock Use: Ensures stocks meet a minimum utilization ratio before marking them as used.
Purpose: To handle complex arrangements of products in various stocks, ensuring efficient space utilization through detailed 
calculations and iterative optimization."""
class BrandAndBoundPolicy(Policy):
    def __init__(self):
        self.used_stock_states = None
        self.min_acceptable_ratio = 0.8  # Increase minimum acceptable ratio
        self.placement_queue = []
        self.total_products_count = 0
        self.current_stock_index = -1
        self.min_ratio_threshold = 0.5  # Add minimum threshold

    def get_action(self, observation, info):
        available_stocks = observation["stocks"]
        available_products = observation["products"]

        # Initialize stock states if not done
        if self.used_stock_states is None:
            self.used_stock_states = [0] * len(available_stocks)
            
        # If we have pending placements, return the next one
        if self.placement_queue:
            return self.placement_queue.pop(0)

        # Find best unused stock
        self.current_stock_index = self._find_best_available_stock(available_stocks)

        # If no suitable stock found, mark current stock as used and retry
        if self.current_stock_index == -1:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        current_stock = available_stocks[self.current_stock_index]
        stock_copy = np.copy(current_stock)

        # Store original product quantities
        initial_quantities = [prod["quantity"] for prod in available_products]
        self.total_products_count = sum(initial_quantities)

        # Find optimal placement configuration
        waste_area = self._calculate_placement_waste(stock_copy, available_products)
        
        # Mark current stock as used
        self.used_stock_states[self.current_stock_index] = 1

        # If placement queue is empty, stock is fully utilized
        if not self.placement_queue:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Return first placement from queue
        return self.placement_queue.pop(0)

    def _find_best_available_stock(self, stocks):
        """Find best unused stock based on both area and aspect ratio"""
        best_score = float('-inf')
        best_stock_idx = -1
        
        for idx, stock in enumerate(stocks):
            if self.used_stock_states[idx] != 0:
                continue
                
            stock_width, stock_height = self._get_stock_size_(stock)
            stock_area = stock_width * stock_height
            # Consider aspect ratio to prefer more square-like stocks
            aspect_ratio = min(stock_width/stock_height, stock_height/stock_width)
            score = stock_area * aspect_ratio
            
            if score > best_score:
                best_score = score
                best_stock_idx = idx
                
        return best_stock_idx

    def _calculate_placement_waste(self, stock, products):
        """Optimize placement to minimize waste"""
        stock_width, stock_height = self._get_stock_size_(stock)
        total_area = stock_width * stock_height
        used_area = 0
        
        # Sort products by area to place larger pieces first
        sorted_products = sorted(
            [(i, p) for i, p in enumerate(products) if p["quantity"] > 0],
            key=lambda x: x[1]["size"][0] * x[1]["size"][1],
            reverse=True
        )
        
        for _, prod in sorted_products:
            prod_w, prod_h = prod["size"]
            best_waste = float('inf')
            best_placement = None
            
            # Try both orientations
            for width, height in [(prod_w, prod_h), (prod_h, prod_w)]:
                for x in range(stock_width - width + 1):
                    for y in range(stock_height - height + 1):
                        if self._can_place_(stock, (x, y), (width, height)):
                            # Calculate local waste (surrounding empty space)
                            local_waste = self._calculate_local_waste(
                                stock, (x, y), (width, height)
                            )
                            if local_waste < best_waste:
                                best_waste = local_waste
                                best_placement = (x, y, width, height)
            
            if best_placement:
                x, y, w, h = best_placement
                stock[x:x+w, y:y+h] = 1
                used_area += w * h
                self.placement_queue.append({
                    "stock_idx": self.current_stock_index,
                    "size": [w, h],
                    "position": (x, y)
                })
                prod["quantity"] -= 1
        
        return total_area - used_area

    def _calculate_local_waste(self, stock, position, size):
        """Calculate waste in the immediate surrounding area"""
        x, y = position
        w, h = size
        padding = 1
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(stock.shape[0], x + w + padding)
        y2 = min(stock.shape[1], y + h + padding)
        
        region = stock[x1:x2, y1:y2]
        return np.sum(region == -1)

    def _process_placement_results(self, observation, initial_quantities, info, waste, stock):
        """Process placement results and return appropriate action"""
        stock_area = np.prod(self._get_stock_size_(stock))
        waste_ratio = waste / stock_area

        if waste_ratio > 1 - self.min_acceptable_ratio and self.used_stock_states[self.current_stock_index] == 0:
            self.used_stock_states[self.current_stock_index] = 2
            self.current_stock_index = -1
            self.placement_queue = []
            return self.get_action(observation, info)

        if len(self.placement_queue) == self.total_products_count:
            self.placement_queue = []
            self._reset_quantities(observation["products"], initial_quantities)
            return self._handle_last_stock(observation, initial_quantities, info)

        self.used_stock_states[self.current_stock_index] = 1
        next_action = self.placement_queue.pop(0) if self.placement_queue else {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (0, 0)
        }
        return next_action

    def _handle_no_stock_available(self, observation, info):
        """Handle case when no suitable stock is found"""
        self.min_acceptable_ratio -= 0.1
        if self.min_acceptable_ratio < 0.4:
            self.used_stock_states = [0 if state == 2 else state for state in self.used_stock_states]
        return self.get_action(observation, info)

    def _calculate_used_products(self, products, initial_quantities):
        """Calculate how many products were used"""
        return [init_qty - prod["quantity"] for init_qty, prod in zip(initial_quantities, products)]

    def _reset_quantities(self, products, initial_quantities):
        """Reset product quantities to initial values"""
        for prod, init_qty in zip(products, initial_quantities):
            prod["quantity"] = init_qty

"""SkylinePlacementPolicy
This class uses the skyline-based placement algorithm, which is well-suited for placing products in a stock with minimal computational complexity. 
It models the height profile of the stock and determines the best placement for each product.

Height Profile: Maintains a dynamic height profile of the stock to track the vertical usage and ensure new placements fit properly.
Product Sorting: Prioritizes products by their height and area for more efficient placement.
Stock Iteration: Moves to the next stock when the current one cannot accommodate any more products.
Waste Threshold: Ensures that placements meet a minimum waste threshold to improve space efficiency.
Fast Placement Evaluation: Quickly determines the best position for a product using the skyline profile.
Purpose: To provide a faster, simpler placement strategy for scenarios where computational efficiency is crucial, sacrificing some optimization for speed."""  
class SkylinePlacementPolicy(Policy):
    def __init__(self):
        self.current_stock_idx = 0
        self.height_profile = None
        self.min_waste_threshold = 0.2

    def get_action(self, observation, info):
        products = sorted(
            [p for p in observation["products"] if p["quantity"] > 0],
            key=lambda x: (-x["size"][1], -x["size"][0] * x["size"][1])  # Sort by height then area
        )
        
        if not products:
            self.current_stock_idx += 1
            self.height_profile = None
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            
        stocks = observation["stocks"]
        if self.current_stock_idx >= len(stocks):
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            
        stock = stocks[self.current_stock_idx]
        
        # Initialize height profile for new stock
        if self.height_profile is None:
            self.height_profile = np.zeros(stock.shape[0], dtype=int)
            
        # Find best placement on current stock
        best_placement = self._find_skyline_placement(stock, products)
        
        if best_placement:
            # Update height profile
            x, y = best_placement["position"]
            w, h = best_placement["size"]
            self.height_profile[x:x+w] = y + h
            return best_placement
            
        # Move to next stock if current one is full
        self.current_stock_idx += 1
        self.height_profile = None
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_skyline_placement(self, stock, products):
        stock_w, stock_h = self._get_stock_size_(stock)
        
        for prod in products:
            if prod["quantity"] <= 0:
                continue
                
            w, h = prod["size"]
            
            # Try both orientations
            for width, height in [(w, h), (h, w)]:
                if width > stock_w or height > stock_h:
                    continue
                    
                # Find position with minimum height increase
                min_waste = float('inf')
                best_pos = None
                
                for x in range(stock_w - width + 1):
                    # Get maximum height at this position
                    current_height = max(self.height_profile[x:x+width])
                    if current_height + height > stock_h:
                        continue
                        
                    # Calculate local waste
                    waste = self._calculate_local_waste(x, current_height, width, height)
                    
                    if waste < min_waste and self._can_place_fast(stock, (x, current_height), (width, height)):
                        min_waste = waste
                        best_pos = (x, current_height)
                
                if best_pos and min_waste < self.min_waste_threshold * (width * height):
                    prod["quantity"] -= 1
                    return {
                        "stock_idx": self.current_stock_idx,
                        "size": [width, height],
                        "position": best_pos
                    }
        
        return None

    def _calculate_local_waste(self, x, y, width, height):
        """Calculate waste space below the piece"""
        profile_segment = self.height_profile[x:x+width]
        waste = sum(y - h for h in profile_segment)
        return waste

    def _can_place_fast(self, stock, position, size):
        pos_x, pos_y = position
        w, h = size
        try:
            return np.all(stock[pos_x:pos_x+w, pos_y:pos_y+h] == -1)
        except IndexError:
            return False
