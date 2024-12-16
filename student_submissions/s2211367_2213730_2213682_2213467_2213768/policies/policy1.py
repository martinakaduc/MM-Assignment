from policy import Policy

class CornerPoint(Policy):
    def __init__(self):
        self.stock_idx = 0
        self.corner_points = {0:[(0,0)]}
    def get_action(self, observation, info):
        # Reset all attributes in new episode
        if not info["filled_ratio"]:
            self.stock_idx = 0
            self.corner_points.clear()
            # Sort the list of products 
            self.list_prods = sorted(observation["products"], key=lambda prod: prod["size"][0], reverse=True)

        # Initialize first corner point if current stock is not placed
        if self.stock_idx not in self.corner_points:
            self.corner_points[self.stock_idx] = [(0,0)]
        stock_w,stock_h = self._get_stock_size_(observation["stocks"][self.stock_idx])
        # Loop through all products, place all possible to current stock
        for prod in self.list_prods:
            # Pick a product
            if prod["quantity"] <=0:
                continue

            prod_w, prod_h = prod["size"]
            # Find a corner point to place this product
            for x,y in self.corner_points[self.stock_idx]:
                if stock_w < x + prod_w or stock_h <y + prod_h: # Product can't fit current stock
                    continue
                # Check if this product doesnt overlap the others
                if self._can_place_(observation["stocks"][self.stock_idx], (x, y), (prod_w, prod_h)):
                    action ={
                        "stock_idx": self.stock_idx,
                        "size": (prod_w,prod_h),
                        "position":(x,y),
                    }
                    # Update corner points list
                    self.corner_points[self.stock_idx].append((x + prod_w, y))
                    self.corner_points[self.stock_idx].append((x, y + prod_h))
                    self.corner_points[self.stock_idx].remove((x, y))
                    return action
        # Move to next stock if cant place any product
        self.stock_idx += 1
        return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}