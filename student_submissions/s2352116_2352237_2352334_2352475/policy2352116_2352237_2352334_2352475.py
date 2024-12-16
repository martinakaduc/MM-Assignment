# import numpy as np
# import time
from policy import Policy
class Policy2352116_2352237_2352334_2352475(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        self.isFirstSolved= False
        if self.policy_id == 1:
            self.solution_1 = Policy2352116_2352237_2352334_2352475.Policy2352475()
        elif self.policy_id == 2:
            self.solution_2 = Policy2352116_2352237_2352334_2352475.Policy2352237()

    def get_action(self, observation, info):
        if self.policy_id == 1:
            action = self.solution_1.get_action(observation, info)
            self.isFirstSolved = self.solution_1.isSolved
            return action
        elif self.policy_id == 2:
            return self.solution_2.get_action(observation, info)
    
    class Policy2352475(Policy):
        def __init__(self):
            self.isSolved = False
            self.result_1 = []
            self.it = 0
            self.performance = []
            pass

        def get_action(self, observation, info):
            if self.it >= len(self.result_1):
                self.isSolved = False
                self.it = 0
                self.result_1 = []

            if not self.isSolved:
                input_prods = []
                input_stocks = []
                products = observation["products"]
                for prod in products:
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]
                        prod_w, prod_h = prod_size
                        new_product = Policy2352116_2352237_2352334_2352475.Policy2352475.Product(prod_w, prod_h)
                        for _ in range(prod["quantity"]):
                            input_prods.append(new_product)

                stocks = observation["stocks"]

                stock_id = 0
                for stock in stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    new_stock = Policy2352116_2352237_2352334_2352475.Policy2352475.Stock(stock_id, stock_w, stock_h)
                    stock_id += 1
                    input_stocks.append(new_stock)

                # start_time = time.time()
                Policy2352116_2352237_2352334_2352475.Policy2352475.place_products_across_stocks(input_stocks, input_prods)
                # end_time = time.time()
                # time_to_solve = end_time - start_time
                # Call evaluate_performance and capture the returned metrics
                # performance_metrics = Policy2352116_2352237_2352334_2352475.Policy2352475.evaluate_performance(input_stocks)
                # if performance_metrics:
                #     performance_metrics['time_to_solve'] = time_to_solve
                #     self.performance.append(performance_metrics)


                self.isSolved = True 
                for stock_idx, stock in enumerate(input_stocks):
                    # Iterate over the placed products in the current stock
                    for product in stock.placed_products:
                        pos_x, pos_y, prod_width, prod_height = product
                        self.result_1.append({
                            "stock_idx": stock_idx,  # Index of the current stock
                            "size": (prod_width, prod_height), 
                            "position": (pos_x, pos_y)
                        })
        
            # print(self.it)
            # print(len(self.result_1))
            action = {
                "stock_idx": self.result_1[self.it]["stock_idx"],
                "size": self.result_1[self.it]["size"],
                "position": self.result_1[self.it]["position"]
            }
            self.it += 1
            return action
        
        class Product:
            def __init__(self, width, height):
                self.width = width
                self.height = height
                self.area = width * height  # Calculate the area of the product
                
        class Area:
            def __init__(self, x, y, width, height):
                self.x = x
                self.y = y
                self.width = width
                self.height = height
                self.area = width * height

            def can_fit(self, product):
                return self.width >= product.width and self.height >= product.height

            def split_area(self, product):
                """Split the remaining area after placing a product using the guillotine cutting approach.
                Returns a list of new areas sorted by area size."""
                new_areas = []
                
                # Calculate the two possible cuts
                if self.width > product.width and self.height > product.height:
                    # Horizontal cut
                    horizontal_right = Policy2352116_2352237_2352334_2352475.Policy2352475.Area(self.x + product.width, self.y, 
                                                    self.width - product.width, self.height)
                    horizontal_top = Policy2352116_2352237_2352334_2352475.Policy2352475.Area(self.x, self.y + product.height,
                                                product.width, self.height - product.height)
                    
                    # Vertical cut
                    vertical_right = Policy2352116_2352237_2352334_2352475.Policy2352475.Area(self.x + product.width, self.y,
                                                self.width - product.width, product.height)
                    vertical_top = Policy2352116_2352237_2352334_2352475.Policy2352475.Area(self.x, self.y + product.height,
                                                self.width, self.height - product.height)
                    
                    # Choose the cut that minimizes waste
                    horizontal_waste = (horizontal_right.area if horizontal_right.area > 0 else float('inf')) + \
                                    (horizontal_top.area if horizontal_top.area > 0 else float('inf'))
                    vertical_waste = (vertical_right.area if vertical_right.area > 0 else float('inf')) + \
                                    (vertical_top.area if vertical_top.area > 0 else float('inf'))
                    
                    if horizontal_waste <= vertical_waste:
                        if horizontal_right.area > 0:
                            new_areas.append(horizontal_right)
                        if horizontal_top.area > 0:
                            new_areas.append(horizontal_top)
                    else:
                        if vertical_right.area > 0:
                            new_areas.append(vertical_right)
                        if vertical_top.area > 0:
                            new_areas.append(vertical_top)
                elif self.width > product.width:
                    # Only horizontal cut possible
                    new_areas.append(Policy2352116_2352237_2352334_2352475.Policy2352475.Area(self.x + product.width, self.y,
                                                self.width - product.width, self.height))
                elif self.height > product.height:
                    # Only vertical cut possible
                    new_areas.append(Policy2352116_2352237_2352334_2352475.Policy2352475.Area(self.x, self.y + product.height,
                                                self.width, self.height - product.height))
                
                # Sort areas by size (largest first) to promote better space utilization
                new_areas.sort(key=lambda a: a.area, reverse=True)
                return new_areas
        class Stock:
            def __init__(self, stock_id, width, height):
                self.stock_id = stock_id
                self.width = width
                self.height = height
                self.area = width * height
                self.remaining_areas = [Policy2352116_2352237_2352334_2352475.Policy2352475.Area(0, 0, width, height)]
                self.placed_products = []

            def place_product(self, product):
                """Try to place the product using best-fit strategy."""
                best_area = None
                best_waste = float('inf')
                best_orientation = None
                
                # Try both orientations
                orientations = [(product.width, product.height)]
                if product.width != product.height:  # Only add rotation if dimensions differ
                    orientations.append((product.height, product.width))
                    
                for width, height in orientations:
                    for area in self.remaining_areas:
                        if area.width >= width and area.height >= height:
                            waste = (area.width - width) * (area.height - height)
                            if waste < best_waste:
                                best_waste = waste
                                best_area = area
                                best_orientation = (width, height)
                
                if best_area:
                    # Place the product in the best-fit area
                    product.width, product.height = best_orientation
                    self.placed_products.append((best_area.x, best_area.y, product.width, product.height))

                    # Update remaining areas
                    self.remaining_areas.remove(best_area)
                    new_areas = best_area.split_area(product)
                    self.remaining_areas.extend(new_areas)
                    
                    # Merge compatible areas
                    self._merge_areas()
                    return True
                
                return False

            def _merge_areas(self):
                """Merge compatible areas to reduce fragmentation."""
                i = 0
                while i < len(self.remaining_areas):
                    j = i + 1
                    while j < len(self.remaining_areas):
                        area1 = self.remaining_areas[i]
                        area2 = self.remaining_areas[j]
                        
                        # Check if areas can be merged horizontally
                        if (area1.y == area2.y and area1.height == area2.height and 
                            area1.x + area1.width == area2.x):
                            new_area = Policy2352116_2352237_2352334_2352475.Policy2352475.Area(area1.x, area1.y, 
                                                    area1.width + area2.width, area1.height)
                            self.remaining_areas.pop(j)
                            self.remaining_areas[i] = new_area
                            continue
                            
                        # Check if areas can be merged vertically
                        if (area1.x == area2.x and area1.width == area2.width and 
                            area1.y + area1.height == area2.y):
                            new_area = Policy2352116_2352237_2352334_2352475.Policy2352475.Area(area1.x, area1.y,
                                                    area1.width, area1.height + area2.height)
                            self.remaining_areas.pop(j)
                            self.remaining_areas[i] = new_area
                            continue
                            
                        j += 1
                    i += 1
        
        def place_products_across_stocks(stocks, products):
            """Improved placement strategy to maximize utilization across all stocks"""
            # Sort products by area in descending order
            products = sorted(products, key=lambda p: p.area, reverse=True)
            stocks = sorted(stocks, key=lambda p: p.area, reverse=True)
            unplaced_products = products.copy()
            
            while unplaced_products:
                # Find the most empty stock (lowest utilization rate)
                stock_utilizations = []
                for stock in stocks:
                    filled_area = sum(p[2] * p[3] for p in stock.placed_products)
                    utilization = filled_area / (stock.width * stock.height)
                    stock_utilizations.append((stock, utilization))
                
                # Sort stocks by utilization (least utilized first)
                stock_utilizations.sort(key=lambda x: x[1])
                
                # Try to place products in the most empty stock
                current_stock = stock_utilizations[0][0]
                products_to_remove = []
                # Try to fill current stock with remaining products
                for product in unplaced_products:
                    if current_stock.place_product(product):
                        products_to_remove.append(product)
                        
                # Remove placed products from unplaced list
                for product in products_to_remove:
                    unplaced_products.remove(product)
                
                # If we couldn't place any products in this iteration, break to avoid infinite loop
                if not products_to_remove:
                    print(f"Warning: Could not place {len(unplaced_products)} remaining products")
                    break
        
        def evaluate_performance(stocks):
            """
            Evaluate and print aggregate performance metrics for all stocks.

            Metrics:
            - Maximum filled area (%)
            - Minimum filled area (%)
            - Trim loss: Unused area across all filled stocks (%)
            - Unused stocks: Number of stocks with 0% filled area
            """
            utilized_stocks = [stock for stock in stocks if sum(product[2] * product[3] for product in stock.placed_products) > 0]
            
            total_area = sum(stock.width * stock.height for stock in utilized_stocks)  # Total area of utilized stocks
            
            # Calculate the filled areas for each utilized stock
            filled_areas = [
                sum(product[2] * product[3] for product in stock.placed_products)
                for stock in utilized_stocks
            ]
            
            # Calculate trim loss as the sum of unused areas in utilized stocks
            trim_loss = sum((stock.width * stock.height - filled_area) for stock, filled_area in zip(utilized_stocks, filled_areas))
            
            # Calculate the filled percentages for utilized stocks
            filled_percentages = [
                (filled_area / (stock.width * stock.height)) * 100
                for filled_area, stock in zip(filled_areas, utilized_stocks)
            ]
            
            # Calculate the number of unused stocks (those with zero filled area)
            unused_stock_count = len([stock for stock in stocks if sum(product[2] * product[3] for product in stock.placed_products) == 0])
            
            # Calculate max, min, and average filled areas
            # max_filled_area = np.amax(filled_percentages) if filled_percentages else 0
            # min_filled_area = np.amin(filled_percentages) if filled_percentages else 0
            # avg_filled_area = np.mean(filled_percentages) if filled_percentages else 0
        
            
            # Calculate trim loss as a percentage
            trim_loss_percentage = (trim_loss / total_area) * 100
            # Output the performance metrics
            performance_metrics = {
            # "max_filled_area": max_filled_area,
            # "min_filled_area": min_filled_area,
            # "avg_filled_area": avg_filled_area,
            "trim_loss_percentage": trim_loss_percentage,
            "unused_stock_count": unused_stock_count,
            "trim_loss": trim_loss,
            "time_to_solve": None,
            }
            return performance_metrics  
        
        def print_performance(self):
            """
            Print the performance metrics from all test cases and format the output.

            Args:
            - self.performance: List of performance metrics dictionaries from all test cases.

            Returns:
            - Dictionary of aggregated and formatted metrics.
            """
            # Aggregate metrics across all test cases
            # aggregated = {
            #     "max_filled_area": np.max([metrics["max_filled_area"] for metrics in self.performance]),
            #     "min_filled_area": np.min([metrics["min_filled_area"] for metrics in self.performance]),
            #     "avg_filled_area": np.mean([metrics["avg_filled_area"] for metrics in self.performance]),
            #     "max_time_to_solve": np.max([metrics["time_to_solve"] for metrics in self.performance]),  # Fix generator issue
            #     "min_time_to_solve": np.min([metrics["time_to_solve"] for metrics in self.performance]),  # Fix generator issue
            #     "avg_time_to_solve": np.mean([metrics["time_to_solve"] for metrics in self.performance]),  # Fix generator issue
            #     "avg_trim_loss_percentage": np.mean([metrics["trim_loss_percentage"] for metrics in self.performance]),
            #     "avg_unused_stock_count": np.mean([metrics["unused_stock_count"] for metrics in self.performance]),
            #     "total_trim_loss": np.sum([metrics["trim_loss"] for metrics in self.performance]),
            #     "total_unused_stocks": np.sum([metrics["unused_stock_count"] for metrics in self.performance]),
            #     "total_use_stock": len(self.performance)
            # }

            # Format the values to improve readability
            # print(f"Max Filled Area: {aggregated['max_filled_area']:.2f}%")
            # print(f"Min Filled Area: {aggregated['min_filled_area']:.2f}%")
            # print(f"Avg Filled Area: {aggregated['avg_filled_area']:.2f}%")
            # print(f"Avg Trim Loss Percentage: {aggregated['avg_trim_loss_percentage']:.2f}%")
            # print(f"Max Time To Solve: {aggregated['max_time_to_solve']:.4f}")
            # print(f"Min Time To Solve: {aggregated['min_time_to_solve']:.4f}")
            # print(f"Avg Time To Solve: {aggregated['avg_time_to_solve']:.4f}")
            # print(f"Avg Unused Stock Count: {aggregated['avg_unused_stock_count']:.2f}%")
            # print(f"Total Trim Loss: {aggregated['total_trim_loss']:,}")
            # print(f"Total Unused Stocks: {aggregated['total_unused_stocks']:,}")
            # print(f"Total Used Stock: {aggregated['total_use_stock']:,}")

    class Policy2352237(Policy):
        def __init__(self):
            self.result_2 = {}
            self.height_areas = []
            self.current_stock = 0
            self.stock_inventory = []
            self.total_prod_area = 0
            self.total_stock_area = 0
            pass

        def get_action(self, observation, info):
            if info['filled_ratio'] == 0.0:
                self.initialize_height_areas(observation["stocks"])
                # self.initialize_height_areas(observation["stocks"])
            prods = observation["products"]
            prods = sorted(prods, key=lambda p: (p["size"][0] * p["size"][1]), reverse=True)

            for prod in prods:
                if prod["quantity"] == 0:
                    continue

                
                min_waste, best_position, selected_stock, rotate = float('inf'), (-1, -1), -1, False

                for i,(stock, stock_idx) in enumerate(self.stock_inventory[:self.current_stock]):
                    waste, position, should_rotate = self.best_placement(prod, stock, stock_idx)
                    if waste < min_waste:
                        min_waste, best_position, selected_stock, rotate = waste, position, stock_idx, should_rotate

                while min_waste == float('inf'):
                    self.current_stock += 1
                    new_stock = self.stock_inventory[self.current_stock - 1]
                    stock_width, stock_height = self._get_stock_size_(new_stock[0])
                    self.total_stock_area += stock_width * stock_height

                    waste, position, should_rotate = self.best_placement(prod, new_stock[0], new_stock[1])
                    if waste < float('inf'):
                        min_waste, best_position, selected_stock, rotate = waste, position, new_stock[1], should_rotate

                if rotate:
                    prod["size"] = prod["size"][::-1]

                self.update_height_area(prod, best_position, observation["stocks"][selected_stock], selected_stock)
                self.total_prod_area += prod["size"][0] * prod["size"][1]
                action= {
                    "stock_idx": selected_stock,
                    "size": prod["size"],
                    "position": best_position
                }
                # if (action["stock_idx"], self._get_stock_size_(observation["stocks"][selected_stock])) not in self.result:
                #     self.result[(action["stock_idx"], self._get_stock_size_(observation["stocks"][selected_stock]))] = 0
                # self.result[(action["stock_idx"], self._get_stock_size_(observation["stocks"][selected_stock]))] += (action["size"][0] * action["size"][1])
                # print(str(self.result))
                return action

        def print_performance(self):
            filled_rat =[]
            for key in self.result_2:
                filled=self.result_2[key]*100 / (key[1][0]*key[1][1])
                if filled > 100:
                    filled = 100
                filled_rat.append(filled)
                    
            aggregated = {
                # "max_filled_area": np.max(filled_rat),
                # "min_filled_area": np.min(filled_rat),
                # "avg_filled_area": np.mean(filled_rat),
                "total_use_stock": len(filled_rat),
                "total_unused_stock": len(self.stock_inventory) - len(filled_rat),
                # "avg_filled_area": np.mean([metrics["avg_filled_area"] for metrics in self.performance]),
                # "max_time_to_solve": np.max([metrics["time_to_solve"] for metrics in self.performance]),  # Fix generator issue
                # "min_time_to_solve": np.min([metrics["time_to_solve"] for metrics in self.performance]),  # Fix generator issue
                # "avg_time_to_solve": np.mean([metrics["time_to_solve"] for metrics in self.performance]),  # Fix generator issue
                # "avg_trim_loss_percentage": np.mean(),
                # "avg_unused_stock_count": np.mean([metrics["unused_stock_count"] for metrics in self.performance]),
                # "total_trim_loss": np.sum([metrics["trim_loss"] for metrics in self.performance]),
                # "total_unused_stocks": np.sum([metrics["unused_stock_count"] for metrics in self.performance]),
            }
            print(f"Max Filled Area: {aggregated['max_filled_area']:.2f}%")
            print(f"Min Filled Area: {aggregated['min_filled_area']:.2f}%")
            print(f"Avg Filled Area: {aggregated['avg_filled_area']:.2f}%")
            # print(f"Avg Trim Loss Percentage: {aggregated['avg_trim_loss_percentage']:.2f}%")
            print(f"Total Used Stock: {aggregated['total_use_stock']:.2f}")
            # print(f"Total Unused Stock: {aggregated['total_unused_stock']:.2f}")
                
        class HeightSegment:
            def __init__(self, start_x, end_x, level):
                self.start_x = start_x
                self.end_x = end_x
                self.level = level

            def has_overlap(self, prod_size, position):
                prod_start_x, prod_y = position
                prod_width, prod_height = prod_size
                prod_end_x = prod_start_x + prod_width

                if prod_end_x <= self.start_x or prod_start_x >= self.end_x:
                    return False

                if prod_y >= self.level:
                    return False

                return True

            def compute_gap_area(self, prod_size, position):
                prod_start_x, prod_y = position
                prod_width, prod_height = prod_size
                prod_end_x = prod_start_x + prod_width

                if prod_end_x <= self.start_x or prod_start_x >= self.end_x:
                    return 0

                return (prod_y - self.level) * (min(prod_end_x, self.end_x) - max(prod_start_x, self.start_x))

        def initialize_height_areas(self, stocks):
            self.height_areas = []
            self.stock_inventory = sorted(
                [(stock, idx) for idx, stock in enumerate(stocks)],
                key=lambda s: -(self._get_stock_size_(s[0])[0] * self._get_stock_size_(s[0])[1])
            )

            for stock, idx in self.stock_inventory:
                stock_width = self._get_stock_size_(stock)[0]
                self.height_areas.append([self.HeightSegment(0, stock_width, 0)])

        def best_placement(self, prod, stock, stock_idx):
            stock_width, stock_height = self._get_stock_size_(stock)
            prod_width, prod_height = prod["size"]

            min_waste = float('inf')
            best_position = (-1, -1)
            should_rotate = False

            for segment in self.height_areas[stock_idx]:
                for x_offset in [segment.start_x, segment.end_x - prod_width]:
                    y_offset = segment.level
                    if x_offset < 0 or x_offset + prod_width > stock_width or y_offset + prod_height > stock_height:
                        continue

                    if not self._can_place_(stock, (x_offset, y_offset), prod["size"]):
                        continue

                    if any(segment.has_overlap(prod["size"], (x_offset, y_offset)) for segment in self.height_areas[stock_idx]):
                        continue

                    waste = sum(segment.compute_gap_area(prod["size"], (x_offset, y_offset)) for segment in self.height_areas[stock_idx])
                    if waste < min_waste:
                        min_waste, best_position, should_rotate = waste, (x_offset, y_offset), False

            prod["size"] = prod["size"][::-1]  # Rotate prod

            for segment in self.height_areas[stock_idx]:
                for x_offset in [segment.start_x, segment.end_x - prod_width]:
                    y_offset = segment.level
                    if x_offset < 0 or x_offset + prod_width > stock_width or y_offset + prod_height > stock_height:
                        continue

                    if not self._can_place_(stock, (x_offset, y_offset), prod["size"]):
                        continue

                    if any(segment.has_overlap(prod["size"], (x_offset, y_offset)) for segment in self.height_areas[stock_idx]):
                        continue

                    waste = sum(segment.compute_gap_area(prod["size"], (x_offset, y_offset)) for segment in self.height_areas[stock_idx])
                    if waste < min_waste:
                        min_waste, best_position, should_rotate = waste, (x_offset, y_offset), True

            prod["size"] = prod["size"][::-1]  # Restore original rotation
            return min_waste, best_position, should_rotate

        def update_height_area(self, prod, position, stock, stock_idx):
            updated_area = []
            prod_width, prod_height = prod["size"]
            position_x, position_y = position
            start = position_x
            end = start + prod_width
            new_height = position_y + prod_height

            updated_area.append(self.HeightSegment(start, end, new_height))

            for segment in self.height_areas[stock_idx]:
                if segment.end_x <= start or segment.start_x >= end:
                    updated_area.append(segment)
                    continue

                if segment.start_x < start:
                    updated_area.append(self.HeightSegment(segment.start_x, start, segment.level))

                if segment.end_x > end:
                    updated_area.append(self.HeightSegment(end, segment.end_x, segment.level))

            self.height_areas[stock_idx] = sorted(updated_area, key=lambda p: p.start_x)