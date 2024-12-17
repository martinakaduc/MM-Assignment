import numpy as np
from policy import Policy


class Policy2352097_2352436_2352435_2053367_2353336(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        
    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.algorithm_one(observation, info)
        elif self.policy_id == 2:
            return self.algorithm_two(observation, info)


    #Wang Algorithm
    def algorithm_one(self, observation, info):
        list_prods = observation["products"]
        stock_idx = -1
        best_fit_area = None
        best_fit_position = None

        #Prioritize products based on their area (greater area first)
        sorted_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        #Loop through all products
        for prod in sorted_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                best_fit_area = None
                best_fit_position = None

                #Try to fit the product in stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Check for the original orientation first
                    fit_found = False
                    for orientation in [(prod_size[0], prod_size[1]), (prod_size[1], prod_size[0])]:
                        prod_w, prod_h = orientation

                        if stock_w < prod_w or stock_h < prod_h:
                            continue  # Skip if the product does not fit in stock

                        # Step 4: Find the position that minimizes fragmentation
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), orientation):
                                    free_space = self._calculate_free_space(stock, (x, y), orientation)
                                    if best_fit_area is None or free_space > best_fit_area:
                                        best_fit_area = free_space
                                        best_fit_position = (x, y)
                                        best_orientation = orientation
                                        stock_idx = i
                                        fit_found = True
                                        break
                            if fit_found:
                                break
                    if fit_found:
                        break  # Exit once we find a valid position

                #If a valid position was found, return the action
                if best_fit_position is not None:
                    return {"stock_idx": stock_idx, "size": best_orientation, "position": best_fit_position}


    def _calculate_free_space(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        stock_w, stock_h = self._get_stock_size_(stock)

        # Create a mask for the current stock's used space
        stock_copy = stock.copy()

        for x in range(pos_x, pos_x + prod_w):
            for y in range(pos_y, pos_y + prod_h):
                stock_copy[x, y] = 1  # Mark as occupied

        # Calculate the remaining free space (unoccupied areas)
        free_space = np.sum(stock_copy == -1)
        return free_space




    #Art Algorithm
    def algorithm_two(self, observation, info):
        list_prods = observation["products"]

        # Sort products based on demand (higher demand first)
        list_prods = sorted(list_prods, key=lambda x: -x["quantity"])

        # Default values if no valid placement is found
        stock_idx, prod_size, pos_x, pos_y = -1, [0, 0], 0, 0

        # Try placing each product based on demand
        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue  # Skip if there is no demand for the product

            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            # Check for placement without rotation
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                if stock_w >= prod_w and stock_h >= prod_h:
                    pos_x, pos_y = self._find_best_position(stock, prod_size)
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

            # Check for placement with rotation
            rotated_size = [prod_h, prod_w]
            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                if stock_w >= rotated_size[0] and stock_h >= rotated_size[1]:
                    pos_x, pos_y = self._find_best_position(stock, rotated_size)
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        return {"stock_idx": stock_idx, "size": rotated_size, "position": (pos_x, pos_y)}


    def _find_best_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Try to find the best position within the stock
        best_pos_x, best_pos_y = None, None
        min_waste = float('inf')  # To track the minimum trim loss

        # Iterate over all positions within stock bounds
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    waste = self._calculate_waste(stock, prod_size, (x, y))
                    if waste < min_waste:
                        min_waste = waste
                        best_pos_x, best_pos_y = x, y

        return best_pos_x, best_pos_y

    def _calculate_waste(self, stock, prod_size, position):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        # Waste is the unutilized space in the stock
        total_area = stock_w * stock_h
        used_area = prod_w * prod_h
        waste_area = total_area - used_area

        return waste_area


    # Student code here
    # You can add more functions if needed
