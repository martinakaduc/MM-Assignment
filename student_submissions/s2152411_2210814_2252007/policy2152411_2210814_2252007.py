from policy import Policy


class policy2152411_2210814_2252007(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        if(self.policy_id == 1):
            return self.blf(observation, info)
        elif(self.policy_id == 2):
            return self.nfp(observation, info)
        # Student code here
        
    def blf(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Pick a product that has quantity > 0
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        best_y, best_x = None, None

                        for y in range(stock_h - prod_h, -1, -1):
                            for x in range(stock_w - prod_w + 1): 
                                if self._can_place_(stock, (x, y), prod_size):
                                    best_y, best_x = y, x
                                    break
                            if best_y is not None and best_x is not None:
                                break

                        if best_y is not None and best_x is not None:
                            pos_x, pos_y = best_x, best_y
                            break
                        
                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = None, None
                        best_y, best_x = None, None

                        for y in range(stock_h - prod_w, -1, -1):
                            for x in range(stock_w - prod_h + 1): 
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    prod_size = prod_size[::-1]
                                    best_y, best_x = y, x
                                    break
                            if best_y is not None and best_x is not None:
                                break

                        if best_y is not None and best_x is not None:
                            pos_x, pos_y = best_x, best_y
                            break
            if pos_x is not None and pos_y is not None:
                stock_idx = i
                break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    def nfp(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        list_prods = list(list_prods)
        list_prods = sorted(list_prods, key=lambda temp: temp["size"][0] * temp["size"][1], reverse=True)
        
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)

            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        pos_x, pos_y = None, None
                        position = self._find_position(stock_w, stock_h, stock, prod_size)
                        if position is not None:
                            pos_x, pos_y = position
                            break
                            
                    if stock_w >= prod_h and stock_h >= prod_w:
                        pos_x, pos_y = None, None
                        position = self._find_position_R(stock_w, stock_h, stock, prod_size[::-1])
                        if position is not None:
                            prod_size = prod_size[::-1]
                            pos_x, pos_y = position
                            break

            if pos_x is not None and pos_y is not None:
                stock_idx = i
                break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def _find_position(self, stock_w, stock_h, stock, shape_size):
        best_position = None
        best_x, best_y = None, None
        prod_w, prod_h = shape_size

        # Duyệt qua tất cả các vị trí khả thi
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), shape_size):
                    if best_y is None or (y < best_y) or (y == best_y and x < best_x):
                        best_y, best_x = y, x
                        best_position = (x, y)

        return best_position
    
    def _find_position_R(self, stock_w, stock_h, stock, shape_size):
        best_position = None
        best_x, best_y = None, None
        prod_w, prod_h = shape_size

        # Duyệt qua tất cả các vị trí khả thi
        for x in range(stock_w - prod_h + 1):
            for y in range(stock_h - prod_w + 1):
                if self._can_place_(stock, (x, y), shape_size):
                    if best_y is None or (y < best_y) or (y == best_y and x < best_x):
                        best_y, best_x = y, x
                        best_position = (x, y)

        return best_position

    # Student code here
    # You can add more functions if needed

