from policy import Policy
import numpy as np

class Policy2353338_2352166_2353264_1952266_2352690(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        class GreedyAreaPackingPolicy(Policy):
            def __init__(self):
                self.decent_waste = 500
                self.stock_counter = 0
                self.sorted_stocks = []
                self.stock_indices = []
                self.sorted_products = []
                self.actions = []
                self.action_index = 0
                self.min_wasted_stock_actions = []
                self.min_wasted_area = float("inf")

                self.total_products = 0

            def _process_stock(self, stock, products, stock_idx):
                stock_w, stock_h = self._get_stock_size_(stock)
                temp_stock = np.copy(stock) # Copy the stock to not modify the original stock
                area_wasted = stock_w * stock_h
                for prod in products:
                    if prod["quantity"] < 1:
                        continue
                    temp_quantity = prod["quantity"]
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    while temp_quantity > 0:
                        placed = False
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(temp_stock, (x, y), prod_size):
                                    temp_stock[x : x + prod_w, y : y + prod_h] = 0
                                    self.actions.append({
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)
                                    })
                                    temp_quantity -= 1
                                    placed = True
                                    area_wasted -= prod_w * prod_h
                                    break
                            if placed:
                                break
                        if not placed:
                            # Try to rotate the product
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(temp_stock, (x, y), prod_size[::-1]):
                                        temp_stock[x : x + prod_h, y : y + prod_w] = 0
                                        prod_size = prod_size[::-1]
                                        self.actions.append({
                                            "stock_idx": stock_idx,
                                            "size": prod_size,
                                            "position": (x, y)
                                        })
                                        temp_quantity -= 1
                                        placed = True
                                        area_wasted -= prod_w * prod_h
                                        break
                                if placed:
                                    break
                        if not placed:
                            break # Try the next product
                if area_wasted < self.min_wasted_area:
                    self.min_wasted_area = area_wasted
                    self.min_wasted_stock_actions = self.actions
                    # print("min_wasted_area:", self.min_wasted_area)
                    # print("min_wasted_stock_actions:", self.min_wasted_stock_actions)
                return area_wasted


            def get_action(self, observation, info):
                if self.actions:
                    action = self.actions.pop(0)
                    # print("action:", action)
                    self.total_products -= 1
                    return action
                # Step 0: Sort the products and stocks by area in descending order
                if self.stock_counter == 0 or self.total_products == 0:
                    stocks = observation["stocks"]
                    products = observation["products"]
                    indexed_stocks = [(i, stock) for i, stock in enumerate(stocks)]
                    indexed_stocks = sorted(
                        indexed_stocks,
                        key=lambda s: np.sum(s[1] != -2),
                        reverse=True
                    )

                    self.sorted_stocks = [item[1] for item in indexed_stocks]
                    self.stock_indices = [item[0] for item in indexed_stocks]
                    # Sort products by area
                    self.sorted_products = sorted(
                        products,
                        key=lambda p: p["size"][0] * p["size"][1],
                        reverse=True
                    )
                    self.total_products = sum([prod["quantity"] for prod in self.sorted_products])

                # Step 1: Initialize the stock and product
                stock_idx = self.stock_indices[self.stock_counter]
                stock = self.sorted_stocks[self.stock_counter]
                products = self.sorted_products

                # Step 2: Process the stock
                if not self.actions:
                    area_wasted = self._process_stock(stock, products, stock_idx)
                    self.stock_counter += 1
                    temp_counter = 0
                    # print("Stock area wasted:", area_wasted)
                    self.min_wasted_stock_actions = self.actions
                    self.min_wasted_area = area_wasted
                    while area_wasted > self.decent_waste:
                        temp_counter += 1
                        if self.stock_counter + temp_counter >= len(self.sorted_stocks):
                            # Find the stock with the least wasted area
                            # print("No more stock to process")
                            self.actions = self.min_wasted_stock_actions
                            break
                        stock_idx = self.stock_indices[self.stock_counter + temp_counter]
                        stock = self.sorted_stocks[self.stock_counter + temp_counter]
                        area_wasted = self._process_stock(stock, products, stock_idx)
                        if area_wasted > self.decent_waste:
                            self.actions = []
                        # print("Stock area wasted(repeat):", area_wasted)
                        
                
                # Double check if self.actions is empty (no solution)
                if not self.actions:
                    # print("Double check")
                    i = -1
                    while not self.actions:
                        stock_idx = self.stock_indices[i]
                        stock = self.sorted_stocks[i]
                        area_wasted = self._process_stock(stock, products, stock_idx)
                        i -= 1
                        if i < -len(self.sorted_stocks):
                            # print("Out of stocks")
                            break
                
                # Triple check if self.actions is still empty (no solution), use GreedyPolicy
                if not self.actions:
                    # print("Triple check")
                    for prod in products:
                        # While there are still products to place, find the available stock
                        while prod["quantity"] > 0:
                            placed = False
                            for i, stock in enumerate(observation["stocks"]):
                                stock_w, stock_h = self._get_stock_size_(stock)
                                prod_w, prod_h = prod["size"]
                                for x in range(stock_w - prod_w + 1):
                                    for y in range(stock_h - prod_h + 1):
                                        if self._can_place_(stock, (x, y), prod["size"]):
                                            self.actions.append({
                                                "stock_idx": i,
                                                "size": prod["size"],
                                                "position": (x, y)
                                            })
                                            prod["quantity"] -= 1
                                            placed = True
                                            break
                                    if placed:
                                        break
                                if placed:
                                    break
                

                # Step 3: Return the action
                action = self.actions.pop(0)
                # print("action:", action)
                self.total_products -= 1
                return action
        
        class BFDPolicy(Policy):
            def __init__(self):
                pass

            def get_action(self, observation, info):
                list_prods = observation["products"]

                list_prods = sorted(list_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

                prod_size = [0, 0]
                stock_idx = -1
                pos_x, pos_y = 0, 0
                best_fit_area = float("inf")  # Start with a very large value for the best fit

                # Pick a product that has quantity > 0
                for prod in list_prods:
                    if prod["quantity"] > 0:
                        prod_size = prod["size"]

                        # Loop through all stocks to find the best fit
                        for i, stock in enumerate(observation["stocks"]):
                            stock_w, stock_h = self._get_stock_size_(stock)
                            prod_w, prod_h = prod_size

                            # Skip if the stock is too small for the product
                            if stock_w < prod_w or stock_h < prod_h:
                                continue

                            # Check for all possible positions in the stock
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        # Calculate the area of unused space in this stock after placing the product
                                        leftover_space = (stock_w * stock_h) - (prod_w * prod_h)
                                        if leftover_space < best_fit_area:
                                            # If this fit is better, update the best fit
                                            best_fit_area = leftover_space
                                            pos_x, pos_y = x, y
                                            stock_idx = i

                        # If we found a valid position, break out of the loop
                        if stock_idx != -1 and pos_x is not None and pos_y is not None:
                            break

                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
            
        if policy_id == 1:
            self.policy = GreedyAreaPackingPolicy()
        elif policy_id == 2:
            self.policy = BFDPolicy()

    def get_action(self, observation, info):
        # Student code here
        return self.policy.get_action(observation, info)
        

    # Student code here
    # You can add more functions if needed
