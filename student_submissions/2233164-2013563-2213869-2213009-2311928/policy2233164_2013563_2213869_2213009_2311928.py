from policy import Policy
import numpy as np

class Policy2210xxx(Policy):
    def __init__(self, policy_id=1, stocks=None, products=None):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def _get_state_key(self, observation, info):
        return str(observation)

    def get_action(self, observation, info):
        if self.policy_id == 1:
            # Use the Adaptive Fit Policy
            action = self.AdaptiveFitPolicy(observation, info)
        else:
            
            # Use the Weighted Area Policy
            action = self.WeightedAreaPolicy(observation, info)
        return action

    def AdaptiveFitPolicy(self, observation, info):
        list_prods = observation["products"]

        # Dynamically evaluate stocks and categorize them as "tight" or "spacious"
        stock_status = []
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            total_space = stock_w * stock_h
            occupied_space = np.sum(stock != -1)
            free_space = total_space - occupied_space
            stock_status.append({
                "index": i,
                "stock": stock,
                "free_space": free_space,
                "tight": free_space < (total_space * 0.3)  # Stock is "tight" if less than 30% free
            })

        # Sort stocks by free space
        tight_stocks = [s for s in stock_status if s["tight"]]
        spacious_stocks = [s for s in stock_status if not s["tight"]]

        # Sort products by area
        sorted_prods_tight = sorted(
            list_prods,
            key=lambda prod: prod["size"][0] * prod["size"][1] if prod["quantity"] > 0 else float("inf"),
        )
        sorted_prods_spacious = sorted(
            list_prods,
            key=lambda prod: -(prod["size"][0] * prod["size"][1]) if prod["quantity"] > 0 else float("-inf"),
        )

        # Try placing smaller items in tight stocks
        for prod in sorted_prods_tight:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                for stock_info in tight_stocks:
                    stock = stock_info["stock"]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {
                                        "stock_idx": stock_info["index"],
                                        "size": prod_size,
                                        "position": (x, y)
                                    }

        # Try placing larger items in spacious stocks
        for prod in sorted_prods_spacious:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size

                for stock_info in spacious_stocks:
                    stock = stock_info["stock"]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {
                                        "stock_idx": stock_info["index"],
                                        "size": prod_size,
                                        "position": (x, y)
                                    }

        # No valid placement found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def WeightedAreaPolicy(self, observation, info):
        list_prods = observation["products"]

        # Calculate a weighted priority for each product
        prioritized_prods = []
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                best_fit_ratio = float("inf")  # Lower is better

                # Check how well the product fits into each stock
                for stock in observation["stocks"]:
                    stock_w, stock_h = self._get_stock_size_(stock)

                    if stock_w >= prod_w and stock_h >= prod_h:
                        wasted_space = (stock_w * stock_h) - (prod_w * prod_h)
                        fit_ratio = wasted_space / (prod_w * prod_h)
                        best_fit_ratio = min(best_fit_ratio, fit_ratio)

                # Add the product to the prioritized list if it can fit somewhere
                if best_fit_ratio != float("inf"):
                    prioritized_prods.append({
                        "product": prod,
                        "fit_ratio": best_fit_ratio
                    })

        # Sort products by fit ratio
        prioritized_prods.sort(key=lambda x: x["fit_ratio"])

        # Attempt to place the highest priority product
        for item in prioritized_prods:
            prod = item["product"]
            prod_size = prod["size"]
            prod_w, prod_h = prod_size

            for i, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                if stock_w >= prod_w and stock_h >= prod_h:
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                return {
                                    "stock_idx": i,
                                    "size": prod_size,
                                    "position": (x, y)
                                }

        # No valid placement found
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

