from policy import Policy
import numpy as np
from typing import Dict, List, Tuple

class Policy2311795_2311805_2311806_2311875_2311883(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        if self.policy_id == 1:
            self.current_stock_index = 0
        elif self.policy_id == 2:
            pass

    def get_action(self, observation: Dict, info: Dict) -> Dict:
        if self.policy_id == 1:
            return self.policy_1_logic(observation, info)
        elif self.policy_id == 2:
            return self.policy_2_logic(observation, info)

    def policy_1_logic(self, observation: Dict, info: Dict) -> Dict:
        #Best Fit

        if self.current_stock_index >= len(observation["stocks"]):
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            
        current_stock = observation["stocks"][self.current_stock_index]
        
        placement = self.find_best_placement(current_stock, observation)  
        if placement:
            return {
                "stock_idx": self.current_stock_index,
                "size": placement["size"],
                "position": placement["position"]
            }
        
        self.current_stock_index += 1
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def find_best_placement(self, stock: np.ndarray, observation: Dict) -> Dict:

        stock_h, stock_w = stock.shape
        best_placement = None
        min_waste = float('inf')

        for product in observation["products"]:
            if product["quantity"] == 0:
                continue
                
            size = product["size"]
            prod_w, prod_h = size

            if stock_w >= prod_w and stock_h >= prod_h:
                for x in range(stock_h - prod_h + 1):
                    for y in range(stock_w - prod_w + 1):
                        if self._can_place(stock, (x, y), size):
                            temp_stock = stock.copy()
                            temp_stock[x:x+prod_h, y:y+prod_w] = 1
                            
                            waste_score = self._calculate_waste_score(temp_stock)
                            if waste_score < min_waste:
                                min_waste = waste_score
                                best_placement = {
                                    "size": size,
                                    "position": (x, y)
                                }

            if prod_w != prod_h and stock_w >= prod_h and stock_h >= prod_w:
                rotated_size = (prod_h, prod_w)
                for x in range(stock_h - prod_w + 1):
                    for y in range(stock_w - prod_h + 1):
                        if self._can_place(stock, (x, y), rotated_size):
                            temp_stock = stock.copy()
                            temp_stock[x:x+prod_w, y:y+prod_h] = 1
                            
                            waste_score = self._calculate_waste_score(temp_stock)
                            if waste_score < min_waste:
                                min_waste = waste_score
                                best_placement = {
                                    "size": rotated_size,
                                    "position": (x, y)
                                }

        return best_placement

    def _can_place(self, stock: np.ndarray, position: Tuple[int, int], size: Tuple[int, int]) -> bool:
        """Check if piece can be placed at position"""
        x, y = position
        width, height = size
        if x < 0 or y < 0 or x + width > stock.shape[0] or y + height > stock.shape[1]:
            return False
        return np.all(stock[x:x+width, y:y+height] == -1)

    def _calculate_waste_score(self, stock: np.ndarray) -> float:
        """Calculate waste score based on empty spaces"""
        empty_spaces = np.sum(stock == -1)
        total_spaces = np.sum(stock != -2)
        return empty_spaces / total_spaces if total_spaces > 0 else float('inf')


    def policy_2_logic(self, observation, info):
        #Column Generation
        list_prods = observation["products"]
        stock_idx = -1
        pos_x, pos_y = None, None
        prod_size = None

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = self._find_position(stock, prod_size)
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            break

                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = self._find_position(stock, prod_size[::-1])
                        if pos_x is not None and pos_y is not None:
                            stock_idx = i
                            prod_size = prod_size[::-1]
                            break

                if stock_idx != -1:
                    break
        if stock_idx == -1:
            return {"stock_idx": stock_idx, "size": None, "position": None}

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        

    def _find_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return x, y
        return None, None