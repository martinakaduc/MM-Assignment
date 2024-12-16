import numpy as np
from scipy.optimize import linprog
from policy import Policy

from copy import deepcopy
    
class Policy2352197_2352542_2352176_2352586_2352374(Policy):
    def __init__(self, policy_id=2):
        # Student code here
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self.guillotine_gr(observation["stocks"], observation["products"])
        else:
            return self.column_gen(observation["stocks"], observation["products"])

    # Student code here
    # You can add more functions if needed

    def guillotine_gr(self, stocks, products):
        
        # def sim_cut_product(stock, position, product_size):
        #     col, row = position
        #     product_width, product_height = product_size
            
        #     for r in range(row, row + product_height):
        #         for c in range(col, col + product_width):
        #             if stock[c, r] == -1:  # Only mark the available spaces
        #                 stock[c, r] = 1
        #             else:
        #                 print(position)
        #                 print(product_size)
        #                 print(stock)
        #                 raise ValueError("Attempted to cut a product into an already occupied space.")         

        # Helper function to calculate remaining space after a cut
        def remaining_space(stock, product_size, position):
            stock_array = np.array(stock)

            cut_area = np.zeros_like(stock_array)
            cut_area[position[0]:position[0] + product_size[0], position[1]:position[1] + product_size[1]] = 1

            remaining_area = np.sum(stock_array == -1) - np.sum(cut_area)
            return remaining_area


        def rectangularity(stock):
            binary_stock = (stock == -1).astype(int)  # Available space is 1, others are 0

            rows, cols = np.where(binary_stock == 1)

            if len(rows) == 0 or len(cols) == 0:
                return 0 

            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            bounding_box_area = (max_row - min_row + 1) * (max_col - min_col + 1)

            shape_area = binary_stock.sum()

            return shape_area / bounding_box_area if bounding_box_area > 0 else 0
        
        def calculate_score(stock, product_size, position):
            remaining_area = remaining_space(stock, product_size, position)
            score = remaining_area
            
            rectangularity_score = rectangularity(stock)
            score = score/rectangularity_score if rectangularity_score > 0 else 1
            
            # print("Remaining area:", remaining_area)
            # print("Rectangularity:", rectangularity_score)
            # print("Score:", score)
            return score

        def find_best_cut(stock, products, direction):
            best_score = float('inf')
            best_product_idx = None
            best_product_size = None
            best_position = None
            
            if direction == "top-left":
                rowrange = range(max_height)
                colrange = range(max_width)
            elif direction == "bottom-right":
                rowrange = range(max_height - 1, -1, -1)
                colrange = range(max_width - 1, -1, -1)
            elif direction == "bottom-left":
                rowrange = range(max_height - 1, -1, -1)
                colrange = range(max_width)
            elif direction == "top-right":
                rowrange = range(max_height)
                colrange = range(max_width - 1, -1, -1)
            
            for row in rowrange:
                for col in colrange:
                    if stock[col, row] != -1:
                        continue

                    for product_idx, product in sorted_products:
                        if product["quantity"] > 0:
                            product_width, product_height = product["size"]
                            
                            if self._can_place_(stock, (col, row), (product_width, product_height)):
                                score = calculate_score(stock, (product_width, product_height), (col, row))

                                if score < best_score:
                                    best_score = score
                                    best_product_idx = product_idx
                                    best_product_size = (product_width, product_height)
                                    best_position = (col, row)

                            # Check rotating
                            if self._can_place_(stock, (col, row), (product_height, product_width)):
                                score = calculate_score(stock, (product_height, product_width), (col, row))

                                if score < best_score:
                                    best_score = score
                                    best_product_idx = product_idx
                                    best_product_size = (product_height, product_width)
                                    best_position = (col, row)
                                    
                    # Found a valid product
                    if best_product_idx is not None:
                        product = products[best_product_idx]
                        product_width, product_height = best_product_size

                        result = {
                            "stock_idx": stock_idx,
                            "size": best_product_size,
                            "position": best_position
                        }
                        print(result, best_score)
                        return result, best_score
                    
            return None, best_score
                        
        ### Main
        sorted_products = sorted(
            enumerate(products),
            key=lambda p: p[1]["size"][0] * p[1]["size"][1],
            reverse=True
        )

        for stock_idx, stock in enumerate(stocks):
            max_width, max_height = self._get_stock_size_(stock)
            result = {}

            best_score = float('inf')
            
            for direction in ["top-left", "bottom-right", "bottom-left", "top-right"]:
                # print("Direction:", direction)
                new_result, new_score = find_best_cut(stock, products, direction)
                if new_result is not None and new_score < best_score:
                    result = new_result
                    best_score = new_score
            # result, best_score = find_best_cut(stock, products, "top-right", best_score)
            
            if result:
                return result
        return None


    def column_gen(self, stocks, _products):
        def solve_column_gen(stock):
            b_new_pattern = True
            next_pattern = None
            current_pattern = initial_pattern

            while b_new_pattern:
                if next_pattern is not None:
                    current_pattern = np.column_stack((current_pattern, next_pattern))

                dual_prices = solve_lp(current_pattern)
                b_new_pattern, next_pattern = gen_new_pattern(dual_prices, stock)

            return {
                "pattern": current_pattern,
                "integers": solve_ilp(current_pattern)
            }
        
        def solve_lp(current_pattern):
            A_ub = -current_pattern
            b_ub = -products_quantities
            res = linprog(c, A_ub, b_ub)

            return res.ineqlin.marginals if res.success else None

        def solve_ilp(current_pattern):
            A_ub = -current_pattern
            b_ub = -products_quantities
            res = linprog(c, A_ub, b_ub)

            return res.x.astype(int) if res.success else None

        def gen_new_pattern(dual_prices, stock):
            A_ub = [
                np.minimum(products_size[:, 0], products_size[:, 1]),
                np.maximum(products_size[:, 0], products_size[:, 1])
            ]
            stock_size = self._get_stock_size_(stock)
            b_ub = [stock_size[0], stock_size[1]]
            res = linprog(dual_prices, A_ub, b_ub)
            
            # print(res.x)
            
            if res.success:
                new_pattern = np.round(res.x).astype(int)
                if 1 - (dual_prices @ new_pattern) < 0:
                    return True, new_pattern
                else:
                    return False, None
            
            return None
        
        def find_cut_in_size(stock, stock_width, stock_height, p_size):
            for col in range(stock_width - p_size[0] + 1):
                for row in range(stock_height - p_size[1] + 1):
                    if self._can_place_(stock, (col, row), p_size):
                        return {"size": p_size, "position": (col, row)}
            return None
        
        def find_cut_in_pattern(stock, stock_width, stock_height, pattern, products_size):
            for p_idx, p_count in enumerate(pattern):
                if p_count > 0:
                    for p_size in (products_size[p_idx], products_size[p_idx][::-1]):
                        if stock_width >= p_size[0] and stock_height >= p_size[1]:
                            result = find_cut_in_size(stock, stock_width, stock_height, p_size)
                            if result:
                                return result
            return None

        
        ### Main
        products = sorted(
            _products,
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )
        
        c = np.ones(len(products))
        initial_pattern = np.eye(len(products), dtype=int)
        products_quantities = np.array([p["quantity"] for p in products])
        products_size = np.array([p["size"] for p in products])


        # Find the valid cut
        for stock_idx, stock in enumerate(stocks):
            patterns = solve_column_gen(stock)

            stock_width, stock_height = self._get_stock_size_(stock)
            for pattern, qty in zip(patterns["pattern"].T, patterns["integers"]):
                if qty > 0:
                    result = find_cut_in_pattern(stock, stock_width, stock_height, pattern, products_size)
                    if result:
                        return {
                            "stock_idx": stock_idx,
                            "size": result["size"],
                            "position": result["position"]
                        }
                        
        return None
