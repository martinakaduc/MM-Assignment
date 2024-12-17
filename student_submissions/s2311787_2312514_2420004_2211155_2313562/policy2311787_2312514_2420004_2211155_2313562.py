from tkinter import SE
from policy import Policy
from scipy.optimize import linprog
from pulp import LpProblem, LpVariable, LpMinimize, lpSum
import random
import numpy as np

class Policy2311787_2312514_2420004_2211155_2313562(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy_id = 1
            self.visited = []
            pass
        elif policy_id == 2:
            self.policy_id = 2
            self.visited = []
            pass

    def get_max_uncut_stock(self,observation,info):
        list_stocks = observation["stocks"]
        max_w = -1
        max_h = -1
        
        for sidx, stock in enumerate(list_stocks):
            stock_w, stock_h = self._get_stock_size_(stock)

            if sidx in self.visited:
                continue
            if stock_w > max_w and stock_h > max_h:
                max_w = stock_w
                max_h = stock_h
                max_idx = sidx
                max_stock = stock
            
        return max_idx, max_stock if max_w > 0 else (-1, None)

    def GreedyFFDH(self, observation, info):
        prod_size = [0, 0]
        prod_size = self.get_max_product(observation, info)
        stock_idx = -1
        pos_x, pos_y = None, None

        for i in self.visited:

            stock = observation["stocks"][i]
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_rot = False

            if stock_h > stock_w:
                stock_rot = True

            prod_w, prod_h = prod_size
            if (stock_rot is False):
                if stock_w >= prod_w and stock_h >= prod_h:
                    pos_x, pos_y = None, None
                    for y in range(stock_h - prod_h + 1):
                        for x in range(stock_w - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
            else:
                if stock_h >= prod_w and stock_w >= prod_h:
                     pos_x, pos_y = None, None
                        
                     for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                prod_size = prod_size[::-1]
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                     if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

        stock_idx, next_stock = self.get_max_uncut_stock(observation,info)
        stock_w, stock_h = self._get_stock_size_(next_stock)

        if (stock_h > stock_w) : # rotate stock
            if self._can_place_(next_stock, (0, 0), prod_size[::-1]):
                self.visited.append(stock_idx)
                return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": (0, 0)}

        else:
            if self._can_place_(next_stock, (0, 0), prod_size):
                self.visited.append(stock_idx)
                return {"stock_idx": stock_idx, "size": prod_size, "position": (0, 0)}

    def get_max_product(self,observation, info):
        list_prods = observation["products"]
        max_w = -1
        max_h = -1
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_w, prod_h = prod["size"]
                if prod_h > prod_w:
                    prod_w, prod_h = prod_h, prod_w

                if prod_w > max_w and prod_h > max_h:
                    max_w = prod_w
                    max_h = prod_h

        return max_w, max_h 


    """"""""""""""""""""""""""""""""""""""""""""""""
# implement the column generation
    def column_generation(self, observation, info):
        list_prods = observation["products"]
        best_column = None
        best_weight = float('inf')

        # Iterate over each stock
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)  # Get stock dimensions

            # Solve Knapsack for width
            width_items = [(prod["quantity"], prod["size"][1]) for prod in list_prods if prod["quantity"] > 0]
            width_capacity = stock_w
            width_value, width_selected = self.solve_knapsack(width_items, width_capacity)

            # Generate patterns from width results
            for item_idx in width_selected:
                prod = list_prods[item_idx]
                prod_size = prod["size"]

                # Solve Knapsack for length based on selected width pattern
                length_items = [(1, prod_size[0])]
                length_capacity = stock_h
                length_value, length_selected = self.solve_knapsack(length_items, length_capacity)

                # Check placement of the resulting pattern
                for pos_x in range(stock_w - prod_size[0] + 1):
                    for pos_y in range(stock_h - prod_size[1] + 1):
                        if self._can_place_(stock, (pos_x, pos_y), prod_size):
                            waste = self._calculate_waste(stock, (pos_x, pos_y), prod_size)

                            if waste < best_weight:
                                best_weight = waste
                                best_column = {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (pos_x, pos_y),
                                }

                                # Early stop if no waste
                                if best_weight == 0:
                                    return best_column

        return best_column

    def solve_knapsack(self, items, capacity):
        n = len(items)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            value, weight = items[i - 1]
            for w in range(capacity + 1):
                if weight > w:
                    dp[i][w] = dp[i - 1][w]
                else:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight] + value)

        # Trace back to find selected items
        w = capacity
        selected_items = []
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                selected_items.append(i - 1)
                w -= items[i - 1][1]

        return dp[n][capacity], selected_items

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))  # Usable width
        stock_h = np.sum(np.any(stock != -2, axis=0))  # Usable height
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)

    def _calculate_waste(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        # Total usable area in the stock
        stock_area = np.sum(stock != -2)  # Non-waste area in stock

        # Area to be occupied by the product
        cut_area = prod_w * prod_h

        # Waste area calculation
        return stock_area - cut_area
    """"""""""""""""""""""""""""""""""""""""""""""""""
    def get_action(self, observation, info):
        # Student code here
        if(self.policy_id == 1):
            if info["filled_ratio"] == 0:
                self.visited = []
            return self.GreedyFFDH(observation,info)
        elif(self.policy_id == 2):
            return self.column_generation(observation, info)
    # Student code here
    # You can add more functions if needed
