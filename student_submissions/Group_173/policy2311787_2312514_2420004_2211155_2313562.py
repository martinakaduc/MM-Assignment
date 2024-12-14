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
        list_prods = sorted(observation["products"], key=lambda p: -p["quantity"])
        best_column = None
        best_weight = float('inf')

        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)

            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]

                    valid_positions = self._find_valid_positions(stock, prod_size)
                    for pos_x, pos_y in valid_positions:
                        waste = self._calculate_waste(stock, (pos_x, pos_y), prod_size)

                        if waste < best_weight:
                            best_weight = waste
                            best_column = {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (pos_x, pos_y),
                            }
                            if best_weight == 0:
                                return best_column

        return best_column


    def _get_stock_size_(self, stock):
        """
        Calculate the usable dimensions of the stock.

        Args:
            stock: 2D numpy array representing the stock.

        Returns:
            Tuple of (usable width, usable height).
        """
        stock_w = np.sum(np.any(stock != -2, axis=1))  # Usable width
        stock_h = np.sum(np.any(stock != -2, axis=0))  # Usable height
        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        """
        Check if a product can be placed in the stock at the given position.

        Args:
            stock: 2D numpy array representing the stock.
            position: Tuple (x, y) for the top-left corner of placement.
            prod_size: Tuple (width, height) of the product.

        Returns:
            True if the product can be placed, otherwise False.
        """
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        return np.all(stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] == -1)

    def _find_valid_positions(self, stock, prod_size):
        """
        Find all valid positions in the stock where the product can be placed.

        Args:
            stock: 2D numpy array representing the stock.
            prod_size: Tuple (width, height) of the product.

        Returns:
            List of valid (x, y) positions.
        """
        prod_w, prod_h = prod_size
        stock_w, stock_h = stock.shape

        valid_positions = []
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    valid_positions.append((x, y))
        return valid_positions

    def _calculate_waste(self, stock, position, prod_size):
        """
        Calculate the waste area for placing a product in the stock.

        Args:
            stock: 2D numpy array representing the stock.
            position: Tuple (x, y) for the top-left corner of placement.
            prod_size: Tuple (width, height) of the product.

        Returns:
            Waste area (integer).
        """
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
            if info["filled_ratio"] == 0:  
                self.visited = []
            return self.column_generation(observation, info)
    # Student code here
    # You can add more functions if needed
