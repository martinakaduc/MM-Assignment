from policy import Policy
import numpy as np

class OptimalPlacementStrategyPolicy(Policy):
    def __init__(self):
        self.set_action = []
        self.set_pre_action = []
        self.indexed_stocks = []

    def get_action(self, observation, info):
        if self.set_action:
            action = {"stock_idx": self.set_action[0][0],
                      "size": self.set_action[0][1],
                      "position": self.set_action[0][2]}
            self.set_action.pop(0)
            return action
        if self.indexed_stocks == [] or self._is_not_placed_(observation["stocks"]):
            self.indexed_stocks = [(idx, stock) for idx, stock in enumerate(observation["stocks"])]
            self.indexed_stocks = sorted(
                self.indexed_stocks,
                key=lambda s: self._get_stock_size_(s[1])[0] * self._get_stock_size_(s[1])[1],
                reverse=True
                )
        best_trim_loss = float("inf")
        best_stock_idx = -1
        list_prods = observation["products"]
        list_prods = sorted(list_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        for stock_idx, stock in self.indexed_stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_grid = np.full((stock_w, stock_h), -1, dtype=int)
            used_area = 0

            for prod in list_prods:
                if prod["quantity"] > 0:
                    orientations = [(prod["size"][0], prod["size"][1]), (prod["size"][1], prod["size"][0])]
                    for _ in range(prod["quantity"]):
                        for cur_width, cur_height in orientations:
                            placed = False
                            if cur_width <= stock_w and cur_height <= stock_h:
                                for x in range(stock_w - cur_width + 1):
                                    for y in range(stock_h - cur_height + 1):
                                        if self._can_place_(stock_grid, (x, y), (cur_width, cur_height)) and self._can_place_(stock, (x, y), (cur_width, cur_height)):
                                            stock_grid[x:x + cur_width, y:y + cur_height] = 0
                                            self.set_action.append([stock_idx, (cur_width, cur_height), (x, y)])
                                            used_area += cur_width * cur_height
                                            placed = True
                                            break
                                    if placed:
                                        break
                                if placed:
                                    break
        
            total_area = stock_w * stock_h
            trim_loss = (total_area - used_area) / total_area
            
            if trim_loss <= 0.15:
                action = {"stock_idx": self.set_action[0][0],
                          "size": self.set_action[0][1],
                          "position": self.set_action[0][2]}
                self.set_action.pop(0)
                self.indexed_stocks.remove((stock_idx, stock))
                return action
            
            if best_stock_idx == stock_idx:
                if self.set_pre_action == []: 
                    break
                self.set_action = self.set_pre_action
                self.set_pre_action = []
                action = {"stock_idx": self.set_action[0][0],
                            "size": self.set_action[0][1],
                            "position": self.set_action[0][2]}
                self.set_action.pop(0)
                return action
            
            if trim_loss < best_trim_loss:
                best_stock_idx = stock_idx
                best_trim_loss = trim_loss
                self.set_pre_action = self.set_action
            self.indexed_stocks.append((stock_idx, stock))
            self.indexed_stocks.remove((stock_idx, stock))
            self.set_action.clear()

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    def _is_not_placed_(seft, stocks):
        for stock in stocks:
            if stock[0][0] != -1:
                return False
        return True
        
class BestFitPolicy(Policy):
    def __init__(self, policy_id=1):
        self.pre_prod_size = [-1, -1]
        self.set_pre_stock = []
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            pass
        elif policy_id == 2:
            pass

    def get_action(self, observation, info):
        list_prods = sorted(
            observation["products"], 
            key=lambda p: p["size"][0] * p["size"][1], 
            reverse=True
        )

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = None, None
        min_remaining_area = float("inf")
        isPlace = False

        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_size_true = prod["size"]
                prod_w, prod_h = prod_size

                if tuple(prod_size) != tuple(self.pre_prod_size):
                    self.set_pre_stock = list(range(len(observation["stocks"])))
                    self.pre_prod_size = prod_size

                for i in range(len(observation["stocks"])):
                    if i in self.set_pre_stock:
                        stock = observation["stocks"][i]
                        stock_w, stock_h = self._get_stock_size_(stock)
                        if stock_w >= prod_w and stock_h >= prod_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        remaining_area = (
                                            (stock_w * stock_h) 
                                            - (prod_w * prod_h)
                                        )
                                        isPlace = True
                                        if remaining_area < min_remaining_area:
                                            prod_size_true = prod_size
                                            min_remaining_area = remaining_area
                                            pos_x, pos_y = x, y
                                            stock_idx = i
                                        if i not in self.set_pre_stock:
                                            self.set_pre_stock.append(i)
                                        break
                                       
                        if stock_w >= prod_h and stock_h >= prod_w:
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), prod_size[::-1]):
                                        remaining_area = (
                                            (stock_w * stock_h) 
                                            - (prod_h * prod_w) 
                                        )
                                        isPlace = True
                                        if remaining_area < min_remaining_area:
                                            prod_size_true = prod_size[::-1]
                                            min_remaining_area = remaining_area
                                            pos_x, pos_y = x, y
                                            stock_idx = i
                                        if i not in self.set_pre_stock:
                                            self.set_pre_stock.append(i)
                                        break
                    if isPlace == False and i in self.set_pre_stock:
                        self.set_pre_stock.remove(i)
                if pos_x is not None and pos_y is not None:
                    break

        if stock_idx == -1 or pos_x is None or pos_y is None:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        return {"stock_idx": stock_idx, "size": prod_size_true, "position": (pos_x, pos_y)}    
    
class Policy2311892_2311449_2311459_2310836_2310979(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = OptimalPlacementStrategyPolicy()
        elif policy_id == 2:
            self.policy = BestFitPolicy()
    
    
    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

    # Student code here
    # You can add more functions if needed