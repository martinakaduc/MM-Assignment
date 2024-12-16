from policy import Policy
import numpy as np


class Policy2353084_2352765_2353044(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # Student code here
        if policy_id == 1: #Hybrid Algorithm
            self.quantity = 0
            self.proceed = 0
            self.get_action = self.get_action1

        elif policy_id == 2:#First - Fit - Decreasing algorithm
            self.get_action = self.get_action2
            pass

    def get_action(self, observation, info):
        # Student code here
        pass

    def get_action1(self, observation, info):
        # Student code here
        list_prods = observation["products"]
        list_prods_sorted = sorted(list_prods, key=lambda prod: prod['size'][0] * prod['size'][1], reverse=True)
        for prod in list_prods_sorted:
            if prod['size'][0] > prod['size'][1]:
                prod['size'] = [prod['size'][1], prod['size'][0]]

        if self.quantity == 0:
            self.quantity = sum(prod['quantity'] for prod in list_prods_sorted)

        list_stocks = observation["stocks"]

        stock_areas = []
        for index, stock in enumerate(list_stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            area = stock_w * stock_h
            stock_areas.append((area, index))

        if self.quantity > 100:
            sorted_stock_areas_descending = sorted(stock_areas, key=lambda x: x[0], reverse=True)
        else:
            sorted_stock_areas_descending = sorted(stock_areas, key=lambda x: x[0])

        # Process large group first

        for prod in list_prods_sorted:
            if self.proceed < self.quantity // 2 and prod["quantity"] > 0:
                # print("under 1/3.")
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                for area, original_index in sorted_stock_areas_descending:
                    stock = observation["stocks"][original_index]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    pos_x, pos_y = None, None
                    if stock_w <= stock_h:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):  # spread width
                                if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                        if pos_x is not None and pos_y is not None:
                            stock_idx = original_index
                            self.proceed += 1
                            if self.proceed >= self.quantity:
                                self.proceed = self.quantity = 0
                            return {
                                "stock_idx": stock_idx,
                                "size": (prod_h, prod_w),  # Return the rotated orientation
                                "position": (pos_x, pos_y),
                            }

                    else:
                        for x in range(stock_h - prod_h + 1):
                            for y in range(stock_w - prod_w + 1):  # Spread width
                                if self._can_place_(stock, (y, x), (prod_w, prod_h)):
                                    pos_x, pos_y = y, x
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                        if pos_x is not None and pos_y is not None:
                            stock_idx = original_index
                            self.proceed += 1
                            if self.proceed >= self.quantity:
                                self.proceed = self.quantity = 0
                            return {
                                "stock_idx": stock_idx,
                                "size": (prod_w, prod_h),  # Return the rotated orientation
                                "position": (pos_x, pos_y),
                            }

        # Process small group next

        for prod in list_prods_sorted:
            if prod["quantity"] > 0:
                # print("over 1/3.")
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                for area, original_index in sorted_stock_areas_descending:
                    stock = observation["stocks"][original_index]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    pos_x, pos_y = None, None
                    # Try both orientations: original and rotated
                    for attempt in range(2):  # First attempt: original, second attempt: rotated
                        if attempt == 1:  # On second attempt, rotate the product
                            prod_w, prod_h = prod_h, prod_w

                        for y in range(stock_h - prod_h, -1, -1):  # Iterate over rows (bottom to top)
                            for x in range(stock_w - prod_w, -1, -1):  # Iterate over columns (right to left)
                                if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                    pos_x, pos_y = x, y
                                    break
                            if pos_x is not None and pos_y is not None:
                                break

                        if pos_x is not None and pos_y is not None:
                            break

                    if pos_x is not None and pos_y is not None:
                        stock_idx = original_index
                        self.proceed += 1
                        if self.proceed >= self.quantity:
                            self.proceed = self.quantity = 0
                        return {
                            "stock_idx": stock_idx,
                            "size": (prod_w, prod_h),  # Return the final orientation
                            "position": (pos_x, pos_y),
                        }

    def get_action2(self, observation, info):
        # Student code here

        # Sort products based on their total area in descending order
        list_prods = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda x: x["size"][0] * x["size"][1],
            reverse=True
        )

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Sort stocks based on available area in descending order
        list_of_sorted_stocks = sorted(
            enumerate(observation["stocks"]),
            key=lambda x: self._get_stock_size_(np.array(x[1]))[0] * self._get_stock_size_(np.array(x[1]))[1] if len(
                x[1]) > 0 else 0,
            reverse=True
        )

        # Pick a product that has quantity > 0
        for prod in list_prods:
            prod_size = prod["size"]
            rotated = False  # Flag to check if the product has been rotated

            # If rotated, we rotate the product back to its original size
            if rotated:
                prod_size = prod_size[::-1]

            # Loop through sorted stocks
            for i, stock in list_of_sorted_stocks:
                stock = np.array(stock)  # Ensure stock is a NumPy array
                stock_w, stock_h = self._get_stock_size_(stock)

                prod_w, prod_h = prod_size

                # First check with the original placement
                if stock_w >= prod_w and stock_h >= prod_h:
                    pos_x, pos_y = None, None

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                        # If the original placement is not possible, try the rotated placement
                if not rotated and stock_w >= prod_h and stock_h >= prod_w:
                    pos_x, pos_y = None, None

                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                prod_size = prod_size[::-1]
                                pos_x, pos_y = x, y
                                rotated = True  # Mark as rotated
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break  # Break once we find a valid placement

            if pos_x is not None and pos_y is not None:
                break  # Break the product loop once a valid placement is found

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}