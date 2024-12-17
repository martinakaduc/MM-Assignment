from numpy import empty, stack
from policy import Policy
import numpy as np


class Policy2312613_2311388_2312271_2313145(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.my_policy = FirstFitDecreasing() 
        elif policy_id == 2:
            self.my_policy =BestFitDecreasing()

    def get_action(self, observation, info):
        return self.my_policy.get_action(observation, info)

class FirstFitDecreasing(Policy): 

    def __init__(self):
        # Student code here
        self.stack = []
        self.stock_index = 0
        self.passed_list = []

    def get_action(self, observation, info):
        # Student code here
        list_stock = observation["stocks"]
        list_prods = observation["products"]
        while True:
            if not self.stack:
                self.stock_index = self.max_stock(list_stock)
                self.passed_list.append(self.stock_index)
                stock = list_stock[self.stock_index]
                stock_w, stock_h = self._get_stock_size_(stock)

                self.stack.append([0, 0, stock_w, stock_h])

            while self.stack:
                pos_x, pos_y, prod_size = self.max_prod(list_prods)
                if pos_x != -1 and pos_y != -1:
                    if(self.empty(list_prods) == 1):
                        self.clear()
                    if self._can_place_(list_stock[self.stock_index], (pos_x, pos_y), prod_size):
                        return {"stock_idx": self.stock_index, "size": prod_size, "position": (pos_x, pos_y)}






    # Student code here
    def max_prod(self, list_prods, mode = 2):
        if mode == 0:
            list_prods = sorted(list_prods, reverse = True, key= lambda x:x["size"][0] )

        if mode == 1:
            list_prods = sorted(list_prods, reverse = True, key= lambda x:x["size"][1] )

        if mode == 2:
            list_prods = sorted(list_prods, reverse = True, key= lambda x:x["size"][0] * x["size"][1] )



        pos_x, pos_y, width, height = self.stack.pop()
        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue
            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            if(prod_w < prod_h):
                prod_size = prod_h, prod_w
                prod_w, prod_h = prod_size

            if prod_w > width or prod_h > height:
                if prod_w > height or prod_h > width:
                    continue
                prod_size = prod_h, prod_w
                prod_w, prod_h = prod_size

            self.stack.append([pos_x + prod_w , pos_y, width - prod_w, height])
            self.stack.append([pos_x, pos_y + prod_h , prod_w, height - prod_h])
            return pos_x, pos_y, prod_size
        return -1, -1, (-1,-1)




    def empty(self, list_prods): #Check if product has only one product (Will reset eviroment next time) ==> To call clear() if teacher not call
        one = 0
        for prod in list_prods:
            if prod["quantity"] > 1:
                return 0
            if prod["quantity"] == 0:
                continue
            one = one + 1
        if one == 1:
            return 1
        return 0


    def clear(self):            # MUST USE AFTER EACH EPISODE TO CLEAR STACK
        self.stack.clear()      # MUST USE AFTER EACH EPISODE TO CLEAR STACK
        self.passed_list.clear() # MUST USE AFTER EACH EPISODE TO CLEAR STACK
        #WHAT IMPORTANT MUST BE REPEATED 3 TIMES

    def max_stock(self, list_stock):
        max_size = 0
        max_idx = -1
        for i in range(0, len(list_stock) - 1):
            if i in self.passed_list:
                continue
            stock_i = list_stock[i]
            stock_w, stock_h = self._get_stock_size_(stock_i)
            if(stock_w * stock_h > max_size):
                max_size = stock_h * stock_w
                max_idx = i

        return max_idx



    # You can add more functions if needed

class BestFitDecreasing(Policy):
    def __init__(self):
        super().__init__()
        self.stocks_list = []

    def get_action(self, observation, info):
        list_prods = list(observation["products"])  # Convert tuple to list
        num_products = len(list_prods)  # Number of products in the input

        # Sort products by area from largest to smallest
        list_prods.sort(key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        # Check and initialize leftovers if not exist
        if "leftovers" not in observation:
            observation["leftovers"] = []

        # Update current stock list
        self.stocks_list = list(observation["stocks"])
        used_stocks = set()
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # If the number of products is small, choose the smallest stock
                if num_products < 4:
                    while prod["quantity"] > 0:
                        smallest_stock_idx = self._find_smallest_stock_(used_stocks)
                        while smallest_stock_idx is not None:
                            pos_x, pos_y = self._dp_cutting_(self.stocks_list[smallest_stock_idx], prod_size)
                            if pos_x is None and pos_y is None:
                                # Try rotating the product 90 degrees
                                rotated_size = (prod_size[1], prod_size[0])
                                pos_x, pos_y = self._dp_cutting_(self.stocks_list[smallest_stock_idx], rotated_size)
                                if pos_x is not None and pos_y is not None:
                                    prod_size = rotated_size

                            if pos_x is not None and pos_y is not None:
                                self._update_stock_(observation, smallest_stock_idx, prod_size, pos_x, pos_y)
                                prod["quantity"] -= 1
                                used_stocks.add(smallest_stock_idx)
                                return {
                                    "stock_idx": smallest_stock_idx,
                                    "size": prod_size,
                                    "position": (pos_x, pos_y)
                                }
                            else:
                                # If unable to place the product in the current stock, find the next smallest stock
                                smallest_stock_idx = self._find_next_smallest_stock_(smallest_stock_idx, used_stocks)
                        # If unable to find a suitable position in all stocks, exit the loop
                        break
                    continue  # Move to the next product

                # Find the best position between leftovers and stocks
                best_stock_idx = -1
                best_score = -float('inf')
                best_pos_x, best_pos_y = None, None

                # Iterate through the leftovers list first
                for i, leftover in enumerate(observation["leftovers"]):
                    score = self._evaluate_stock_(leftover, prod_size)
                    if score > best_score:
                        pos_x, pos_y = self._dp_cutting_(leftover, prod_size)
                        if pos_x is None and pos_y is None:
                            # Try rotating the product 90 degrees
                            rotated_size = (prod_size[1], prod_size[0])
                            pos_x, pos_y = self._dp_cutting_(leftover, rotated_size)
                            if pos_x is not None and pos_y is not None:
                                prod_size = rotated_size

                        if pos_x is not None and pos_y is not None:
                            best_score = score
                            best_stock_idx = i
                            best_pos_x, best_pos_y = pos_x, pos_y

                # If a suitable leftover is found, use it
                if best_stock_idx != -1:
                    self._update_stock_(observation, best_stock_idx, prod_size, best_pos_x, best_pos_y)
                    prod["quantity"] -= 1
                    return {
                        "stock_idx": best_stock_idx,
                        "size": prod_size,
                        "position": (best_pos_x, best_pos_y)
                    }

                # If no suitable leftover is found, iterate through new stocks
                for i, stock in enumerate(self.stocks_list):
                    if i in used_stocks:
                        continue
                    score = self._evaluate_stock_(stock, prod_size)
                    if score > best_score:
                        pos_x, pos_y = self._dp_cutting_(stock, prod_size)
                        if pos_x is None and pos_y is None:
                            # Try rotating the product 90 degrees
                            rotated_size = (prod_size[1], prod_size[0])
                            pos_x, pos_y = self._dp_cutting_(stock, rotated_size)
                            if pos_x is not None and pos_y is not None:
                                prod_size = rotated_size

                        if pos_x is not None and pos_y is not None:
                            best_score = score
                            best_stock_idx = i
                            best_pos_x, best_pos_y = pos_x, pos_y

                # If a suitable stock is found, place the product in it
                if best_stock_idx != -1 and best_pos_x is not None and best_pos_y is not None:
                    self._update_stock_(observation, best_stock_idx, prod_size, best_pos_x, best_pos_y)
                    prod["quantity"] -= 1
                    used_stocks.add(best_stock_idx)
                    return {
                        "stock_idx": best_stock_idx,
                        "size": prod_size,
                        "position": (best_pos_x, best_pos_y)
                    }

        # If unable to place the product in any stock
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _evaluate_stock_(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w < prod_w or stock_h < prod_h:
            return -float('inf')  # Cannot place

        # Evaluate by the optimal area
        stock_area = stock_w * stock_h
        prod_area = prod_w * prod_h

        # Optimize with the leftover factor after cutting
        shape_factor = (stock_w - prod_w) * (stock_h - prod_h)

        waste_factor = (stock_w - prod_w) + (stock_h - prod_h)  # Prioritize the smallest part

        return stock_area - prod_area - waste_factor

    def _calculate_waste_(self, stock, prod_size, position):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        x, y = position

        remaining_w = stock_w - (x + prod_w)
        remaining_h = stock_h - (y + prod_h)

        return remaining_w + remaining_h

    def _dp_cutting_(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Only check areas that can be placed
        possible_positions = self._get_possible_positions(stock, prod_size)

        min_waste = float('inf')
        best_position = (None, None)

        for x, y in possible_positions:
            waste = self._calculate_waste_(stock, prod_size, (x, y))
            if waste < min_waste:
                min_waste = waste
                best_position = (x, y)

        return best_position

    def _get_possible_positions(self, stock, prod_size):
        # Only return feasible positions based on the remaining empty area
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        positions = []

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    positions.append((x, y))

        return positions

    def _calculate_leftover_scores(self, leftovers):
        scores = []
        for leftover in leftovers:
            score = leftover["size"][0] * leftover["size"][1]  # Calculate the area of the leftover
            scores.append(score)
        return scores

    def _update_stock_(self, observation, stock_idx, prod_size, pos_x, pos_y):
        stock = self.stocks_list[stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Update the empty area after cutting the product
        new_leftover = {
            "size": [stock_w - prod_w, stock_h - prod_h],
            "pos_x": pos_x + prod_w,
            "pos_y": pos_y + prod_h
        }
        observation["leftovers"].append(new_leftover)

        # Update the stock
        self.stocks_list[stock_idx] = np.copy(stock)
        self.stocks_list[stock_idx][pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] = -2

    def _find_smallest_stock_(self, used_stocks):
        smallest_stock_idx = None
        smallest_stock_size = float('inf')
        for i, stock in enumerate(self.stocks_list):
            if i in used_stocks:
                continue
            stock_size = self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]
            if stock_size < smallest_stock_size:
                smallest_stock_size = stock_size
                smallest_stock_idx = i
        return smallest_stock_idx

    def _find_next_smallest_stock_(self, current_idx, used_stocks):
        next_smallest_stock_idx = None
        next_smallest_stock_size = float('inf')
        current_stock_size = self._get_stock_size_(self.stocks_list[current_idx])[0] * self._get_stock_size_(self.stocks_list[current_idx])[1]
        for i, stock in enumerate(self.stocks_list):
            if i not in used_stocks and i != current_idx:
                stock_size = self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1]
                if current_stock_size < stock_size < next_smallest_stock_size:
                    next_smallest_stock_size = stock_size
                    next_smallest_stock_idx = i
        return next_smallest_stock_idx