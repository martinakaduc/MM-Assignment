import numpy as np
from policy import Policy


class Policy2213367_2213370_2213323_2213240(Policy):
    def __init__(self, policy_id=None):
        self.policy_id = policy_id

    def get_action(self, observation, info):
        """
        Determine the action based on the policy_id.
        Args:
            observation (dict): Current state of the environment.
            info (dict): Additional environment info.

        Returns:
            dict: Action containing stock index, size, and position.
        """
        if self.policy_id == 1:
            return self._bfd_action(observation, info)
        elif self.policy_id == 2:
            return self._bottom_left_action(observation, info)
        else:
            raise ValueError("Invalid policy_id. Supported values: 1 (BFD), 2 (Bottom-Left)")
    # BFD algorithm
    def _bfd_action(self, observation, info):
        """
        Improved BFD policy to reduce trim loss by considering product rotation and optimizing space usage.

        Args:
            observation (dict): Current state of the environment.
            info (dict): Additional environment info.

        Returns:
            dict: Action containing stock index, size, and position.
        """
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # Sort products by area in descending order
        sorted_prods = sorted(
            list_prods,
            key=lambda prod: prod["size"][0] * prod["size"][1],
            reverse=True,
        )

        # Ensure that at least one product is placed in the first action to avoid empty stock
        product_placed = False

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                best_fit = None
                best_stock_idx = -1
                best_position = None
                best_orientation = prod_size  # Default orientation

                # Try placing in each stock and find the best fit
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w < prod_w and stock_h < prod_h:
                        continue

                    # Try both orientations (original and rotated)
                    for orientation in [(prod_w, prod_h), (prod_h, prod_w)]:
                        ori_w, ori_h = orientation
                        if stock_w < ori_w or stock_h < ori_h:
                            continue

                        # Search for the best position within the stock
                        for x in range(stock_w - ori_w + 1):
                            for y in range(stock_h - ori_h + 1):
                                if self._can_place_(stock, (x, y), orientation):
                                    # Evaluate fit quality (minimize wasted space)
                                    waste = (stock_w * stock_h) - (ori_w * ori_h)
                                    if best_fit is None or waste < best_fit:
                                        best_fit = waste
                                        best_stock_idx = stock_idx
                                        best_position = (x, y)
                                        best_orientation = orientation

                # If a position is found in an existing stock
                if best_position is not None:
                    product_placed = True
                    return {
                        "stock_idx": best_stock_idx,
                        "size": best_orientation,
                        "position": best_position,
                    }

                # If no stock can fit, create a new stock and place the product
                if not product_placed:
                    return {
                        "stock_idx": len(stocks),
                        "size": prod_size,
                        "position": (0, 0),
                    }

        # If no valid product is found, return a default action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _find_bottom_left_position(self, stock, prod_size, stock_w, stock_h):
        """
        Tìm vị trí thấp nhất và gần bên trái nhất để đặt sản phẩm.
        """
        prod_w, prod_h = prod_size
        best_position = None
        best_y = stock_h  

        for pos_x in range(stock_w - prod_w + 1):  
            for pos_y in range(stock_h - prod_h + 1):  
                
                if self._can_place_(stock, (pos_x, pos_y), prod_size):
                    if pos_y < best_y or (pos_y == best_y and pos_x < best_position[0]):
                        best_position = (pos_x, pos_y)
                        best_y = pos_y
        return best_position  
    
    #Bottom Left algorithm
    def _find_bottom_left_position(self, stock, prod_size, stock_w, stock_h):
        """
        Tìm vị trí thấp nhất và gần bên trái nhất để đặt sản phẩm.
        """
        prod_w, prod_h = prod_size
        best_position = None
        best_y = stock_h  

        for pos_x in range(stock_w - prod_w + 1):  
            for pos_y in range(stock_h - prod_h + 1):  
                
                if self._can_place_(stock, (pos_x, pos_y), prod_size):
                    if pos_y < best_y or (pos_y == best_y and pos_x < best_position[0]):
                        best_position = (pos_x, pos_y)
                        best_y = pos_y
        return best_position  


    def _bottom_left_action(self, observation, info):
        """
        Thực hiện thuật toán Bottom-Left để đặt sản phẩm lên các kho.
        """
        list_prods = observation["products"]  
        stocks = observation["stocks"]       

        list_prods = sorted(list_prods, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)

        for prod_idx, prod in enumerate(list_prods):
            if prod["quantity"] > 0:  
                prod_size = prod["size"]  
                for stock_idx, stock in enumerate(stocks):
                    
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    position = self._find_bottom_left_position(stock, prod_size, stock_w, stock_h)
                    if position is not None:
                        return {"stock_idx": stock_idx, "size": prod_size, "position": position}

                    position = self._find_bottom_left_position(stock, prod_size[::-1], stock_w, stock_h)
                    if position is not None:
                        return {"stock_idx": stock_idx, "size": prod_size[::-1], "position": position}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}