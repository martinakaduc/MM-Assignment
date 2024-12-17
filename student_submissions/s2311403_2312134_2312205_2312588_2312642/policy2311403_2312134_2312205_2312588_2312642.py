from policy import Policy
import numpy as np

class MaximalRectangle:
    def __init__(self, width, height):
        self.rectangles = [(0, 0, width, height)]
        self.width = width
        self.height = height
        
    def find_best_fit(self, w, h):
        best_pos = (-1, -1)
        min_waste = float('inf')
        for rx, ry, rw, rh in self.rectangles:
            if rw >= w and rh >= h:
                waste = (rw * rh - w * h) + (rx + ry) * 0.1
                if waste < min_waste:
                    min_waste = waste
                    best_pos = (rx, ry)
        return best_pos

    def split(self, x, y, w, h):
        new_rectangles = []
        for rx, ry, rw, rh in self.rectangles:
            if not (x + w <= rx or x >= rx + rw or y + h <= ry or y >= ry + rh):
                if y > ry:
                    new_rectangles.append((rx, ry, rw, y - ry))
                if y + h < ry + rh:
                    new_rectangles.append((rx, y + h, rw, ry + rh - (y + h)))
                if x > rx:
                    new_rectangles.append((rx, y, x - rx, h))
                if x + w < rx + rw:
                    new_rectangles.append((x + w, y, rx + rw - (x + w), h))
            else:
                new_rectangles.append((rx, ry, rw, rh))
        self.rectangles = sorted(new_rectangles, key=lambda r: (r[1], r[0]))
        
class Policy2311403_2312134_2312205_2312588_2312642(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # Student code here
        if policy_id == 1:
            self.highestusedstock = -1
            self.sorted_prods = None
            self.sorted_stocks = None
            # self.sorted_prods = observation["products"]
            # self.sorted_stocks = observation["stocks"]
            # self.sortLists(observation)
            self.isSorted = False
            pass
        elif policy_id == 2:
            self.pattern_cache = {}
            pass
        else:
            pass   

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            if not self.isSorted:
                self.sortLists(observation)
                self.isSorted = True
    # Iterate through sorted stocks to place products
            for stock_idx, stock in enumerate(self.sorted_stocks):
                if stock_idx <= self.highestusedstock:
                    continue
                stock_w, stock_h = self._get_stock_size_(stock)

                # Attempt to place products in descending order of size
                for prod in self.sorted_prods:
                    if prod["quantity"] <= 0:
                        continue

                    # Define both orientations: original and rotated
                    orientations = [
                        prod["size"],  # Original orientation
                        (prod["size"][1], prod["size"][0])  # Rotated orientation
                    ]

                    for orientation in orientations:
                        prod_w, prod_h = orientation

                        # Check if the product fits in the current orientation
                        if stock_w < prod_w or stock_h < prod_h:
                            continue

                        # Determine the range for x and y based on orientation
                        max_x = stock_w - prod_w
                        max_y = stock_h - prod_h

                        # Find a position for the current product
                        pos_x, pos_y = None, None
                        for x in range(max_x + 1):
                            for y in range(max_y + 1):
                                if self._can_place_(stock, (x, y), orientation):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                        # If position is found, return the action with the current orientation
                        if pos_x is not None and pos_y is not None:
                            # Get the original stock index without ambiguity
                            original_stock_idx = next(
                                (i for i, original_stock in enumerate(observation["stocks"]) if original_stock is stock),
                                -1
                            )
                            return {
                                "stock_idx": original_stock_idx,
                                "size": list(orientation),
                                "position": (pos_x, pos_y)
                            }

                # If the largest products don't fit, try placing smaller products
                for prod in self.sorted_prods[::-1]:  # Now consider smallest products
                    if prod["quantity"] <= 0:
                        continue

                    # Define both orientations: original and rotated
                    orientations = [
                        prod["size"],  # Original orientation
                        (prod["size"][1], prod["size"][0])  # Rotated orientation
                    ]

                    for orientation in orientations:
                        prod_w, prod_h = orientation

                        # Check if the product fits in the current orientation
                        if stock_w < prod_w or stock_h < prod_h:
                            continue

                        # Determine the range for x and y based on orientation
                        max_x = stock_w - prod_w
                        max_y = stock_h - prod_h

                        # Find a position for the current product
                        pos_x, pos_y = None, None
                        for x in range(max_x + 1):
                            for y in range(max_y + 1):
                                if self._can_place_(stock, (x, y), orientation):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                        # If position is found, return the action with the current orientation
                        if pos_x is not None and pos_y is not None:
                            return {
                                "stock_idx": observation["stocks"].index(stock),  # Map to original stock index
                                "size": list(orientation),
                                "position": (pos_x, pos_y)
                            }

                # Update the highest used stock index
                self.highestusedstock = stock_idx
            self.isSorted = False
            # If no valid placement is found, return a default action
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        elif self.policy_id == 2:
            products = [p for p in observation["products"] if p["quantity"] > 0]
            if not products:
                return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

            best_result = {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            best_score = 0

            for stock_idx, stock in enumerate(observation["stocks"]):
                patterns = []
                # Generate initial patterns
                for i, prod in enumerate(products):
                    pattern = self._generate_base_pattern(products, stock, i)
                    if pattern and sum(pattern["counts"]) > 0:
                        patterns.append(pattern)
                
                # Column generation phase
                for _ in range(3):  # Limited iterations
                    new_pattern = self._generate_pattern(products, stock)
                    if new_pattern and new_pattern not in patterns:
                        patterns.append(new_pattern)
                
                # Try patterns
                for pattern in patterns:
                    score = self._evaluate_pattern(pattern, stock)
                    if score > best_score:
                        pos = self._find_position(stock, pattern["size"])
                        if pos[0] >= 0 and self._can_place_(stock, pos, pattern["size"]):
                            best_score = score
                            best_result = {
                                "stock_idx": stock_idx,
                                "size": pattern["size"],
                                "position": pos
                            }
                            if score > 0.9:  # Good enough
                                return best_result

            return best_result
    # Student code here
    #Policy 1
    def sortLists(self, observation):
        self.highestusedstock = -1
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # Sort products by area (descending order)
        self.sorted_prods = sorted(
            [prod for prod in list_prods if prod["quantity"] > 0],
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True
        )

        # Sort stocks by area (descending order)
        self.sorted_stocks = sorted(
            stocks,
            key=lambda stock: self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1],
            reverse=True
        )
    #Policy 2
    def _generate_base_pattern(self, products, stock, start_idx):
        stock_w, stock_h = self._get_stock_size_(stock)
        w, h = products[start_idx]["size"]
        max_rect = MaximalRectangle(stock_w, stock_h)
        counts = [0] * len(products)
        
        # Try both orientations
        pos1 = max_rect.find_best_fit(w, h)
        pos2 = max_rect.find_best_fit(h, w)
        
        if pos1[0] >= 0:
            counts[start_idx] = 1
            return {"counts": counts, "size": [w, h]}
        elif pos2[0] >= 0:
            counts[start_idx] = 1
            return {"counts": counts, "size": [h, w]}
            
        return None

    def _generate_pattern(self, products, stock):
        stock_w, stock_h = self._get_stock_size_(stock)
        max_rect = MaximalRectangle(stock_w, stock_h)
        counts = [0] * len(products)
        best_area = 0
        best_size = None
        
        for i, prod in enumerate(products):
            w, h = prod["size"]
            qty = prod["quantity"]
            
            while qty > 0:
                pos1 = max_rect.find_best_fit(w, h)
                pos2 = max_rect.find_best_fit(h, w)
                
                if pos1[0] >= 0:
                    max_rect.split(pos1[0], pos1[1], w, h)
                    counts[i] += 1
                    qty -= 1
                    area = w * h
                    if area > best_area:
                        best_area = area
                        best_size = [w, h]
                elif pos2[0] >= 0:
                    max_rect.split(pos2[0], pos2[1], h, w)
                    counts[i] += 1
                    qty -= 1
                    area = w * h
                    if area > best_area:
                        best_area = area
                        best_size = [h, w]
                else:
                    break
                    
        return {"counts": counts, "size": best_size} if sum(counts) > 0 else None

    def _evaluate_pattern(self, pattern, stock):
        if not pattern or not pattern["size"]:
            return 0
            
        stock_w, stock_h = self._get_stock_size_(stock)
        w, h = pattern["size"]
        
        area_util = (w * h) / (stock_w * stock_h)
        aspect_ratio = min(w/h, h/w)
        
        return 0.8 * area_util + 0.2 * aspect_ratio

    def _find_position(self, stock, size):
        stock_w, stock_h = self._get_stock_size_(stock)
        w, h = size
        best_pos = (-1, -1)
        min_waste = float('inf')
        
        for x in range(stock_w - w + 1):
            for y in range(stock_h - h + 1):
                if self._can_place_(stock, (x, y), (w, h)):
                    waste = x + y  # Prefer positions closer to origin
                    if waste < min_waste:
                        min_waste = waste
                        best_pos = (x, y)
                        
        return best_pos
    # You can add more functions if needed
