import numpy as np
from policy import Policy


class Policy2353187_2352230_2352181_2352204_2352585(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self._products = None
        self._stocks = None
        self.sorted_products = []
        # Initialize skyline data structures if using policy 2
        if policy_id == 2:
            self.skylines = {}  # Dictionary to store skyline for each stock

    def _initialize_skyline_(self, stock):
        """Initialize skyline for a stock."""
        stock_w = np.sum(np.any(stock != -2, axis=1))
        return np.zeros(stock_w, dtype=int)  # Start with ground level

    def _update_skyline_(self, skyline, x, width, height):
        """Update skyline after placing a rectangle."""
        # Get current height at position x
        current_height = skyline[x]
        # Update the skyline for the width of the piece
        skyline[x:x + width] = current_height + height
        return skyline

    def _find_skyline_position_(self, stock, prod_size, skyline):
        """Find the best position using highly optimized skyline placement strategy."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        
        best_fit = {
            'x': None,
            'y': None,
            'rotated': False,
            'waste': float('inf')
        }
        
        # Pre-calculate empty space count for quick rejection
        empty_space = np.sum(stock == -1)
        min_area = min(prod_w * prod_h, prod_h * prod_w)
        if empty_space < min_area:
            return None, False
        
        # Use vectorized operations for skyline analysis
        skyline_diffs = np.abs(np.diff(skyline))
        max_skyline = np.max(skyline)
        
        # Try both orientations with optimized checking
        for width, height in [(prod_w, prod_h), (prod_h, prod_w)]:
            if width > stock_w or height > stock_h:
                continue
            
            # Use sliding window approach for efficiency
            for x in range(stock_w - width + 1):
                region_skyline = skyline[x:x + width]
                region_height = np.max(region_skyline)
                
                if region_height + height > stock_h:
                    continue
                
                # Vectorized region check
                region = stock[x:x + width, region_height:region_height + height]
                if not np.all(region == -1):
                    continue
                
                # Efficient waste calculation using pre-calculated diffs
                waste = np.sum(skyline_diffs[x:x + width - 1])
                
                if waste < best_fit['waste']:
                    best_fit.update({
                        'x': x,
                        'y': region_height,
                        'rotated': (width == prod_h),
                        'waste': waste
                    })
                    
                    # Early termination for perfect fit
                    if waste == 0:
                        break
        
        if best_fit['x'] is not None:
            return (best_fit['x'], best_fit['y']), best_fit['rotated']
        return None, False

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]
        
        # Sort products if needed or if products have changed
        if (self._products is None or 
            len(self._products) != len(products) or 
            self._products != products):
            self._products = products
            self.sorted_products = self._sort_products_(products)
            if self.policy_id == 2:
                # Reset skylines for new products
                self.skylines = {}
        
        if self.policy_id == 1:
            return self._get_action_policy1_(products, stocks)
        else:
            return self._get_action_policy2_(products, stocks)

    def _get_action_policy1_(self, products, stocks):
        """Optimized first-fit decreasing strategy with faster position checking."""
        best_fit = {
            'stock_idx': None,
            'position': None,
            'size': None,
            'remaining_space': float('inf')
        }
        
        for prod_idx, _, prod_size in self.sorted_products:
            if products[prod_idx]["quantity"] <= 0:
                continue
            
            # Early termination if we found a perfect fit
            if best_fit['remaining_space'] == 0:
                break
            
            for stock_idx, stock in enumerate(stocks):
                # Quick check if stock has enough empty space
                if np.sum(stock == -1) < prod_size[0] * prod_size[1]:
                    continue
                
                position, rotated = self._find_first_fit_(stock, prod_size)
                if position is not None:
                    # Calculate remaining space more efficiently
                    width, height = prod_size[::-1] if rotated else prod_size
                    remaining_space = np.sum(stock == -1) - (width * height)
                    
                    if remaining_space < best_fit['remaining_space']:
                        best_fit.update({
                            'stock_idx': stock_idx,
                            'position': position,
                            'size': (width, height),
                            'remaining_space': remaining_space
                        })
                        
                        # Early termination if perfect fit
                        if remaining_space == 0:
                            break
            
            if best_fit['stock_idx'] is not None:
                return {
                    "stock_idx": best_fit['stock_idx'],
                    "size": best_fit['size'],
                    "position": best_fit['position']
                }
        
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _get_action_policy2_(self, products, stocks):
        """Highly optimized skyline-based placement strategy."""
        best_fit = {
            'stock_idx': None,
            'position': None,
            'rotated': False,
            'waste_score': float('inf'),
            'size': None
        }
        
        # Pre-calculate stock metrics using vectorized operations
        stock_metrics = []
        unused_stocks = []
        
        # Vectorized stock analysis
        for idx, stock in enumerate(stocks):
            stock_mask = stock >= 0
            stock_usage = np.sum(stock_mask)
            
            if stock_usage > 0:
                valid_mask = stock != -2
                total_area = np.sum(valid_mask)
                usage_ratio = stock_usage / total_area
                
                # Vectorized row analysis
                used_rows = np.any(stock_mask, axis=1)
                if np.any(used_rows):
                    max_height = np.max(np.nonzero(used_rows)[0]) + 1
                    width_sum = np.sum(np.any(valid_mask, axis=1))
                    compactness = stock_usage / (max_height * width_sum)
                    
                    stock_metrics.append({
                        'idx': idx,
                        'score': usage_ratio * 0.7 + compactness * 0.3,
                        'usage': stock_usage  # Cache for later use
                    })
            else:
                unused_stocks.append(idx)
        
        # Efficient sorting
        ordered_stocks = [s['idx'] for s in sorted(stock_metrics, key=lambda x: x['score'], reverse=True)]
        ordered_stocks.extend(unused_stocks)
        
        # Cache frequently accessed values
        stock_metric_count = len(stock_metrics)
        used_stock_indices = set(ordered_stocks[:stock_metric_count])
        
        for prod_idx, _, prod_size in self.sorted_products:
            if products[prod_idx]["quantity"] <= 0:
                continue
            
            prod_area = prod_size[0] * prod_size[1]
            
            for stock_idx in ordered_stocks:
                stock = stocks[stock_idx]
                
                # Quick rejection based on available space
                if np.sum(stock == -1) < prod_area:
                    continue
                
                # Initialize or get existing skyline
                if stock_idx not in self.skylines:
                    self.skylines[stock_idx] = self._initialize_skyline_(stock)
                
                position, rotated = self._find_skyline_position_(
                    stock, prod_size, self.skylines[stock_idx])
                
                if position is not None:
                    current_size = prod_size[::-1] if rotated else prod_size
                    
                    # More efficient waste score calculation
                    waste_score = self._calculate_enhanced_waste_score_(
                        stock,
                        self.skylines[stock_idx],
                        position,
                        current_size,
                        stock_idx in used_stock_indices
                    )
                    
                    if waste_score < best_fit['waste_score']:
                        best_fit.update({
                            'stock_idx': stock_idx,
                            'position': position,
                            'rotated': rotated,
                            'waste_score': waste_score,
                            'size': current_size
                        })
                        
                        # Early termination for good solutions
                        if stock_idx in used_stock_indices and waste_score < 30:  # Lowered threshold
                            break
            
            if best_fit['stock_idx'] is not None:
                # Efficient skyline update
                self._update_skyline_efficient_(
                    self.skylines[best_fit['stock_idx']],
                    best_fit['position'][0],
                    best_fit['size'][0],
                    best_fit['size'][1]
                )
                return {
                    "stock_idx": best_fit['stock_idx'],
                    "size": best_fit['size'],
                    "position": best_fit['position']
                }
        
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _calculate_remaining_space_(self, stock, prod_size, position):
        """Calculate remaining space after placing the product."""
        width, height = prod_size
        x, y = position
        placed_area = width * height
        total_empty = np.sum(stock == -1)
        return total_empty - placed_area

    def _sort_products_(self, products):
        """Sort products by area in decreasing order."""
        product_list = []
        for idx, prod in enumerate(products):
            if prod["quantity"] > 0:
                area = prod["size"][0] * prod["size"][1]
                product_list.append((idx, area, prod["size"]))
        
        return sorted(product_list, key=lambda x: x[1], reverse=True)

    def _get_stock_size_(self, stock):
        """Get the actual size of the stock (excluding -2 values)."""
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _calculate_enhanced_waste_score_(self, stock, skyline, position, size, is_used_stock):
        """Optimized waste score calculation using vectorized operations."""
        x, y = position
        w, h = size
        stock_w, stock_h = self._get_stock_size_(stock)
        
        # Vectorized calculations
        height_variation = np.sum(np.abs(np.diff(skyline)))
        new_max_height = max(np.max(skyline), y + h)
        
        # Efficient area calculations
        total_used_area = np.sum(stock >= 0) + (w * h)
        total_area = stock_w * stock_h
        area_utilization = total_used_area / total_area
        
        # Optimized region analysis
        region_start = max(0, x-1)
        region_end = min(len(skyline), x+w+1)
        region_skyline = skyline[region_start:region_end]
        horizontal_fragmentation = np.sum(np.abs(np.diff(region_skyline)))
        
        # Combined score calculation
        return (
            height_variation * 0.2 +
            new_max_height * 0.2 +
            (1 - area_utilization) * 0.3 +
            (new_max_height / stock_h) * 0.15 +
            horizontal_fragmentation * 0.15 +
            (0 if is_used_stock else 1000)
        )

    def _find_first_fit_(self, stock, prod_size):
        """Find the first position where the product fits in the stock using optimized checking."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        
        # Try both orientations (normal and rotated)
        orientations = [(prod_w, prod_h), (prod_h, prod_w)]
        
        for width, height in orientations:
            if width > stock_w or height > stock_h:
                continue
            
            # Quick check if stock has enough empty spaces before detailed scan
            if np.sum(stock == -1) < width * height:
                continue
            
            # Pre-calculate valid ranges
            y_range = stock_h - height + 1
            x_range = stock_w - width + 1
            
            for y in range(y_range):
                # Quick row check - skip rows without enough empty spaces
                if np.sum(stock[:, y:y+height] == -1) < width * height:
                    continue
                
                for x in range(x_range):
                    # Direct numpy check for empty space
                    if np.all(stock[x:x+width, y:y+height] == -1):
                        return (x, y), (width == prod_h)  # Returns position and whether rotated
        
        return None, False

    def _update_skyline_efficient_(self, skyline, x, width, height):
        """More efficient skyline update using vectorized operations."""
        skyline[x:x + width] += height
        return skyline
