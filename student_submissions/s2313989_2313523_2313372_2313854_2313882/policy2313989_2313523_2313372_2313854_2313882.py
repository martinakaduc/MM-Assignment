from policy import Policy

class Policy2313989_2313523_2313372_2313854_2313882(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            return self.ffdh_policy(observation)
        elif self.policy_id == 2:
            return self.branch_policy(observation)

    def ffdh_policy(self, observation):
        # Sort products by height in descending order
        list_prods = sorted(
            observation["products"],
            key=lambda x: x["size"][0],  # Sort by height
            reverse=True
        )

        # Iterate over sorted products
        for prod in list_prods:
            if prod["quantity"] > 0:  # Only process if quantity > 0
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                # Iterate through available stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Check if stock dimensions can accommodate the product
                    if stock_w >= prod_w and stock_h >= prod_h:
                        # Place the product at the first valid position
                        for y in range(stock_h - prod_h + 1):
                            for x in range(stock_w - prod_w + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    # Mark product as placed
                                    for row in range(y, y + prod_h):
                                        for col in range(x, x + prod_w):
                                            observation["stocks"][i][row][col] = 0

                                    prod["quantity"] -= 1  # Decrease quantity

                                    return {
                                        "stock_idx": i,
                                        "size": prod_size,
                                        "position": (x, y)
                                    }

        # Return default if no valid placement is found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def branch_policy(self, observation):
        # Sort products by height in descending order
        list_prods = sorted(
            observation["products"],
            key=lambda x: x["size"][0],  # Sort by height
            reverse=True
        )

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                best_stock_idx = -1
                best_position = (None, None)
                min_waste = float('inf')

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                # Calculate waste and distance penalty
                                waste_area = (stock[x:x+prod_w, y:y+prod_h] == -1).sum()
                                distance_penalty = x + y

                                waste = 0.7 * waste_area + 0.3 * distance_penalty

                                # Update best position
                                if waste < min_waste:
                                    min_waste = waste
                                    best_stock_idx = i
                                    best_position = (x, y)

                # Return best position if found
                if best_stock_idx != -1 and best_position is not None:
                    return {
                        "stock_idx": best_stock_idx,
                        "size": prod_size,
                        "position": best_position
                    }

        # Return default if no valid placement is found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    # Student code here
    # You can add more functions if needed
