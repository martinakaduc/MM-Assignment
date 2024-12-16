
from policy import Policy


class Policy2352333_2352467_2352930_2352553_2352339(Policy):
    def __init__(self, policy_id=2):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.dp = {}  # Dynamic programming memoization table
        if policy_id == 1:
            self.policy_id = 1
        elif policy_id == 2:
            self.policy_id = 2
        self.dp = {}  # Dynamic programming memoization table
    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.dp_get_action(observation, info)
        elif self.policy_id == 2:
            return self.ffd_get_action(observation, info)

    def dp_get_action(self, observation, info):
        stock_idx = 0
        stock = observation['stocks'][stock_idx]
        products = observation['products']
        
        _, prod_idx, position, size = self.dp_solve(stock, products)
        
        if prod_idx is None:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            
        return {
            "stock_idx": stock_idx,
            "size": size,
            "position": position
        }
    def dp_solve(self, stock, products):
        # Convert state to hashable form
        state = (tuple(map(tuple, stock)), 
                 tuple((p['size'][0], p['size'][1], p['quantity']) for p in products))
        
        if state in self.dp:
            return self.dp[state]
            
        if all(p['quantity'] == 0 for p in products):
            return 0, None, None, None
            
        stock_w, stock_h = self.get_stock_size(stock)
        best_value = float('-inf')
        best_move = None
        
        for i, prod in enumerate(products):
            if prod['quantity'] <= 0:
                continue
                
            w, h = prod['size']
            if stock_w >= w and stock_h >= h:
                for x in range(stock_w - w + 1):
                    for y in range(stock_h - h + 1):
                        if self._can_place_(stock, (x, y), (w, h)):
                            new_stock = stock.copy()
                            new_stock[x:x+w, y:y+h] = i
                            
                            new_products = list(products)
                            new_products[i] = {
                                'size': prod['size'],
                                'quantity': prod['quantity'] - 1
                            }
                            
                            value, _, _, _ = self.dp_solve(new_stock, new_products)
                            value += w * h
                            
                            if value > best_value:
                                best_value = value
                                best_move = (i, (x, y), (w, h))
                                
        self.dp[state] = (best_value, *best_move) if best_move else (0, None, None, None)
        return self.dp[state]

    def ffd_get_action(self, observation, info):
        def sort_by_area(items):
            return sorted(items, key=lambda item: item["size"][0] * item["size"][1], reverse=True)

        def find_position_for_product(stock, product_size):
            stock_w, stock_h = self.get_stock_size(stock)
            prod_w, prod_h = product_size

            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self.can_place(stock, (x, y), product_size):
                        return (x, y)
            return None

        sorted_products = sort_by_area(observation["products"])
        stocks_with_sizes = [(i, stock, *self.get_stock_size(stock)) for i, stock in enumerate(observation["stocks"])]
        stocks_with_sizes.sort(key=lambda s: s[2] * s[3], reverse=True)

        for product in sorted_products:
            if product["quantity"] > 0:  # if quantity is greater than 0, we can place the product
                continue

            prod_size = product["size"]
            best_stock_idx = -1
            best_position = None

            for stock_idx, stock, stock_w, stock_h in stocks_with_sizes:
                for orientation in [prod_size, prod_size[::-1]]:
                    position = find_position_for_product(stock, orientation)
                    if position:
                        best_stock_idx = stock_idx
                        best_position = position
                        best_size = orientation
                        break
                if best_position:
                    break

            if best_position:
                return {"stock_idx": best_stock_idx, "size": best_size, "position": best_position}

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    