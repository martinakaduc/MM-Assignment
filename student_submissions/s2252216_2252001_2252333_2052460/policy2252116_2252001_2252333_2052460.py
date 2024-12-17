from policy import Policy
import numpy as np

class Policy2252116_2252001_2252333_2052460(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.placement_plan = []
            self.current_step = 0
            self.has_planned = False
            self.used_stock_area = 0
            self.total_used_area = 0
            self.previous_observation = None
            self.overall_utilization = 0
            pass


        if policy_id == 2:
            pass


    def get_action(self, observation, info):
        if self.policy_id == 1:
            # Check if this is a new observation by looking at the stocks
            stocks_changed = (self.previous_observation is None or 
                            not np.array_equal(observation["stocks"], 
                                            self.previous_observation["stocks"]))
            
            if (not self.has_planned or 
                len(self.placement_plan) == 0 or 
                self.current_step >= len(self.placement_plan) or
                stocks_changed):  
                
                # Reset all state variables
                self.placement_plan = []
                self.current_step = 0
                self.has_planned = False
                self.used_stock_area = 0
                self.total_used_area = 0
                self.used_stocks = set()
                self.sorted_stocks = None  
                
                # Create new placement plan
                self.placement_plan = self._plan_placements_(observation)
                self.has_planned = True
                self.current_step = 0
                
                # Store current observation for future comparison
                self.previous_observation = observation  

            if self.current_step < len(self.placement_plan):
                action = self.placement_plan[self.current_step]
                self.current_step += 1
                return action
            
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
        elif self.policy_id == 2:
            list_prods = sorted(
                observation["products"], 
                key=lambda x: x["size"][0] * x["size"][1],
                reverse=True
            )

            stock_idx = -1
            pos_x, pos_y = None, None
            prod_size = [0, 0]

            # Iterate through all products to find one with quantity > 0
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    # Iterate through all stocks
                    for i, stock in enumerate(observation["stocks"]):
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size

                        # Check if the product can fit in the stock in the original orientation
                        if stock_w >= prod_w and stock_h >= prod_h:
                            pos_x, pos_y = self._find_bottom_left_position(stock, prod_size)
                            if pos_x is not None and pos_y is not None:
                                stock_idx = i
                                break

                        # Check if the product can fit in the stock in the rotated orientation
                        if stock_w >= prod_h and stock_h >= prod_w:
                            rotated_pos_x, rotated_pos_y = self._find_bottom_left_position(stock, prod_size[::-1])
                            if rotated_pos_x is not None and rotated_pos_y is not None:
                                pos_x, pos_y = rotated_pos_x, rotated_pos_y
                                prod_size = prod_size[::-1]  # Use the rotated size
                                stock_idx = i
                                break

                    if pos_x is not None and pos_y is not None:
                        break

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    # THIS PART IS FOR GREEDY ALGORITHM

    def _plan_placements_(self, observation):
        """Create a complete placement plan at the start"""
        stocks = observation["stocks"]
        products = observation["products"]
        
        # Reset area calculations
        self.used_stock_area = 0
        self.total_used_area = 0
        
        # Sort stocks only if not sorted yet
        if self.sorted_stocks is None:
            self.sorted_stocks = []
            for i, stock in enumerate(stocks):
                w, h = self._get_stock_size_(stock)
                self.sorted_stocks.append((i, w, h, w * h))
            self.sorted_stocks.sort(key=lambda x: x[3], reverse=True)
            self.used_stocks = set()

        # Create and sort product list
        self.product_list = []
        for prod in products:
            if prod["quantity"] > 0:
                size = prod["size"]
                area = size[0] * size[1]
                for _ in range(prod["quantity"]):
                    self.product_list.append((size[0], size[1], area))
        
        # Sort products by area in descending order
        self.product_list.sort(key=lambda x: x[2], reverse=True)
        
        placement_plan = self._plan_multiple_sizes_(observation)

        return placement_plan

    def _plan_multiple_sizes_(self, observation):
        """Strategy for multiple product sizes"""
        if not self.product_list:  # No products to place
            return []

        selected_utilizations = []
        potential_placements = []
        
        # Try to place products in each unused stock
        for stock_idx, stock_w, stock_h, _ in self.sorted_stocks:
            if stock_idx in self.used_stocks:  # Skip if stock was already used
                continue
                
            stock_placements = self._plan_stock_placement_(
                stock_idx, stock_w, stock_h, observation)
            
            if stock_placements:
                used_area = sum(p["size"][0] * p["size"][1] for p in stock_placements)
                stock_area = stock_w * stock_h
                utilization = (used_area / stock_area) * 100
                
                # print(f"Stock {stock_idx} potential utilization: {utilization:.2f}%")
                
                if utilization >= 90:
                    self.used_stock_area += stock_area
                    self.total_used_area += used_area
                    selected_utilizations.append(utilization)
                    self.used_stocks.add(stock_idx)
                    # print(f"Selected Stock {stock_idx} with high utilization: {utilization:.2f}%")
                    return stock_placements
                
                potential_placements.append({
                    'stock_idx': stock_idx,
                    'placements': stock_placements,
                    'utilization': utilization,
                    'stock_w': stock_w,
                    'stock_h': stock_h,
                    'used_area': used_area,
                    'stock_area': stock_area
                })

        if not potential_placements:
            return []

        best_placement = max(potential_placements, key=lambda x: x['utilization'])
        stock_idx = best_placement['stock_idx']
        self.used_stock_area += best_placement['stock_area']
        self.total_used_area += best_placement['used_area']
        selected_utilizations.append(best_placement['utilization'])
        self.used_stocks.add(stock_idx)
        # print(f"Selected Stock {stock_idx} with best possible utilization: {best_placement['utilization']:.2f}%")

        if selected_utilizations:
            self.overall_utilization = sum(selected_utilizations) / len(selected_utilizations)
        
        return best_placement['placements']
    
    def _plan_stock_placement_(self, stock_idx, stock_w, stock_h, observation):
        """Plan placements for a single stock similar to example policy"""
        placements = []
        stock = observation["stocks"][stock_idx].copy()

        # Try each product from our sorted list
        for prod_w, prod_h, _ in self.product_list:
            # Try to place at each possible position
            pos_x, pos_y = None, None
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), [prod_w, prod_h]):
                        pos_x, pos_y = x, y
                        break
                if pos_x is not None and pos_y is not None:
                    break

            if pos_x is not None and pos_y is not None:
                # Add placement
                placements.append({
                    "stock_idx": stock_idx,
                    "size": [prod_w, prod_h],
                    "position": (pos_x, pos_y)
                })
                
                # Update the stock grid
                for i in range(prod_w):
                    for j in range(prod_h):
                        stock[pos_x+i][pos_y+j] = 0  # Mark as used

        return placements


    # THIS PART IS FOR BOTTOM LEFT ALGORITHM

    def _find_bottom_left_position(self, stock, prod_size):
        """Find the bottom-left position to place the product in the stock."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        for y in range(stock_h - prod_h, -1, -1):
            for x in range(stock_w - prod_w  +1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y  # Return the first found position

        return None, None  # Return None if no position is found