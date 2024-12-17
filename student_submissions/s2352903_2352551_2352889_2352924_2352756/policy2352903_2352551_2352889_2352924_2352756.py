from policy import Policy

class Policy2352903_2352551_2352889_2352924_2352756(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.__greedy_best_fit(observation, info)
        elif self.policy_id == 2:
            return self.__greedy_largest_fit_fragment_correction(observation, info)

    def __greedy_best_fit(self, observation, info):
        list_prods = observation["products"]

        best_fit_diff = float('inf')
        best_action = {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (None, None)
        }

        # Loop through all products
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Check if product fits in its original orientation
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    remaining_space = (stock_w * stock_h) - (prod_w * prod_h)
                                    if remaining_space < best_fit_diff:
                                        best_fit_diff = remaining_space
                                        best_action = {
                                            "stock_idx": i,
                                            "size": prod_size,
                                            "position": (x, y)
                                        }
                                    break

                    # Check if product fits in rotated orientation
                    if stock_w >= prod_h and stock_h >= prod_w:
                        for x in range(stock_w - prod_h + 1):
                            for y in range(stock_h - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    remaining_space = (stock_w * stock_h) - (prod_h * prod_w)
                                    if remaining_space < best_fit_diff:
                                        best_fit_diff = remaining_space
                                        best_action = {
                                            "stock_idx": i,
                                            "size": prod_size[::-1],
                                            "position": (x, y)
                                        }
                                    break

        # Return the best action found
        return best_action

    def __greedy_largest_fit_fragment_correction(self, observation, info):
        # Step 0: Sort products by area (largest first)
        products = observation["products"]
        products = sorted(
            products, 
            key=lambda p: (p["size"][0] * p["size"][1], p["size"][0] / p["size"][1]), 
            reverse=True
        )

        selected_product_size = [0, 0]
        selected_stock_index = -1
        placement_x, placement_y = 0, 0

        # Track used stocks for fragmentation correction
        used_stocks = set()

        # Step 1: Largest Fit
        for stock_index, stock in enumerate(observation["stocks"]):
            stock_width, stock_height = self._get_stock_size_(stock)

            for product in products:
                if product["quantity"] > 0:
                    product_size = product["size"]

                    # Check normal orientation
                    for x in range(stock_width - product_size[0] + 1):
                        for y in range(stock_height - product_size[1] + 1):
                            if self._can_place_(stock, (x, y), product_size):
                                selected_stock_index = stock_index
                                selected_product_size = product_size
                                placement_x, placement_y = x, y
                                used_stocks.add(stock_index)
                                break
                        if selected_stock_index != -1:
                            break

                    # Check rotated orientation
                    if selected_stock_index == -1:
                        for x in range(stock_width - product_size[1] + 1):
                            for y in range(stock_height - product_size[0] + 1):
                                if self._can_place_(stock, (x, y), product_size[::-1]):
                                    selected_stock_index = stock_index
                                    selected_product_size = product_size[::-1]
                                    placement_x, placement_y = x, y
                                    used_stocks.add(stock_index)
                                    break
                        if selected_stock_index != -1:
                            break

                # Exit loop if a product is placed
                if selected_stock_index != -1:
                    break

            # Exit loop if a product is placed
            if selected_stock_index != -1:
                break
        # Step 2: Fragmentation Correction
        if selected_stock_index == -1:
            # Iterate through products again, trying to fill gaps
            for product in products:
                if product["quantity"] > 0:
                    product_size = product["size"]

                    # Check all used stocks for possible placement
                    for stock_index in used_stocks:
                        stock = observation["stocks"][stock_index]
                        stock_width, stock_height = self._get_stock_size_(stock)

                        # Normal orientation
                        for x in range(stock_width - product_size[0] + 1):
                            for y in range(stock_height - product_size[1] + 1):
                                if self._can_place_(stock, (x, y), product_size):
                                    selected_stock_index = stock_index
                                    selected_product_size = product_size
                                    placement_x, placement_y = x, y
                                    break
                            if selected_stock_index != -1:
                                break

                        # Rotated orientation
                        if selected_stock_index == -1:
                            for x in range(stock_width - product_size[1] + 1):
                                for y in range(stock_height - product_size[0] + 1):
                                    if self._can_place_(stock, (x, y), product_size[::-1]):
                                        selected_stock_index = stock_index
                                        selected_product_size = product_size[::-1]
                                        placement_x, placement_y = x, y
                                        break
                                if selected_stock_index != -1:
                                    break

                    # Exit loop if the product is placed
                    if selected_stock_index != -1:
                        break
        return {
            "stock_idx": selected_stock_index,
            "size": selected_product_size,
            "position": (placement_x, placement_y),
        }