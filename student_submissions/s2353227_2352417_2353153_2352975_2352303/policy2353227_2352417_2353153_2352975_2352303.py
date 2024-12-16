from policy import Policy
import numpy as np
class Policy2353227_2352417_2353153_2352975_2352303(Policy):
    def __init__(self, policy_id=1):
        # assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy_id = 1
            
        elif policy_id == 2:
            self.policy_id = 2
    
    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1:
            prods = observation["products"]
            sorted_prods = sorted(prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
            sorted_stocks = sorted(enumerate(observation["stocks"]), key=lambda x: self._get_stock_size_(x[1])[0] * self._get_stock_size_(x[1])[1],)

            for prod in sorted_prods:
                if prod["quantity"] == 0:
                    continue
                
                prod_w, prod_h = prod["size"]

                # for i, stock in enumerate(observation["stocks"]):
                for i, stock in sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Fast placement: If the stock is empty, place in the top-left corner
                    if np.all(stock == -1):
                        return {
                            "stock_idx": i,
                            "size": (prod_w, prod_h),
                            "position": (0, 0),
                        }

                    # Evaluate placements and choose the best
                    best_fit = None
                    min_waste = float('inf')

                    for orientation in [(prod_w, prod_h), (prod_h, prod_w)]:
                        max_x = stock_w - orientation[0]
                        max_y = stock_h - orientation[1]

                        for x in range(max_x + 1):
                            for y in range(max_y + 1):
                                if self._can_place_(stock, (x, y), orientation):
                                    # Calculate waste for this placement
                                    used_area = orientation[0] * orientation[1]
                                    total_free_area = np.sum(stock == -1)
                                    waste = total_free_area - used_area

                                    # Update the best fit based on waste
                                    if waste < min_waste:
                                        min_waste = waste
                                        best_fit = {
                                            "stock_idx": i,
                                            "size": orientation,
                                            "position": (x, y),
                                            "waste": waste,
                                        }

                    # If a valid fit is found, use the best one
                    if best_fit:
                        return {
                            "stock_idx": best_fit["stock_idx"],
                            "size": best_fit["size"],
                            "position": best_fit["position"],
                        }

            # If no placement is found
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}


        elif self.policy_id==2:
            products = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True)

            for prod in products:
                    if prod["quantity"] > 0:
                        # Try to place the product into a stock
                        stock_idx, pos_x, pos_y = self._find_fit(observation, prod)

                        if stock_idx is not None:  # If a valid placement was found
                            return {"stock_idx": stock_idx, "size": prod["size"], "position": (pos_x, pos_y)}

        # If no valid placement is found, return None (i.e., no action)
            return None

    def _find_fit(self, observation, prod):
        
        prod_w, prod_h = prod["size"]
        
        # Try placing the product in each stock
        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)

            if stock_w < prod_w or stock_h < prod_h:
                continue  # Skip stocks that are too small to fit the product

            # Find the first available position to place the product
            for pos_x in range(stock_w - prod_w + 1):
                for pos_y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (pos_x, pos_y), prod["size"]):
                        # Apply placement and return the placement details
                        return stock_idx, pos_x, pos_y

        # No fit found
        return None, None, None