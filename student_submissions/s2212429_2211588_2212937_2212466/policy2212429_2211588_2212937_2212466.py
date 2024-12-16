import numpy as np
from policy import Policy


class Policy2212429_2211588_2212937_2212466(Policy):
    policy = 0
    stock_sort = None
    observation_old = None
    item_sort = None
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = 1
        elif policy_id == 2:
            self.policy = 2
    def _can_place_(self, stock, position, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        if prod_size[0] + position[0] > stock_w:
            return 0
        if prod_size[1] + position[1] > stock_h:
            return 0
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)
    def _sort_stocks(self, stocks):
        stock_info = []
        for idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            area = stock_w * stock_h
            perimeter = 2 * (stock_w + stock_h)
            stock_info.append((idx, stock, area, perimeter))

        sorted_stock_info = sorted(stock_info, key=lambda x: (-x[2], x[3]))
        return [(info[0], info[1]) for info in sorted_stock_info]
    def _sort_item1(self, items):
        stock_info = []
        for item in (items):
            if item["quantity"] <= 0:
                continue
            stock_w, stock_h = item["size"]
            area = stock_w * stock_h
            perimeter = 2 * (stock_w + stock_h)
            stock_info.append((item, area, perimeter))

        sorted_stock_info = sorted(stock_info, key=lambda x: (-x[1], x[2]))
        return [(info[0]) for info in sorted_stock_info]
    def _sort_item2(self, items):
        stock_info = []
        for item in (items):
            if item["quantity"] <= 0:
                continue
            stock_w, stock_h = item["size"]
            if stock_w < stock_h:
                item["size"] = item["size"][::-1]
            stock_w, stock_h = item["size"]
            area = stock_w * stock_h
            perimeter = 2 * (stock_w + stock_h)
            stock_info.append((item, stock_w, area))

        # Sort by decreasing area and increasing perimeter
        sorted_stock_info = sorted(stock_info, key=lambda x: (-x[1], -x[2]))
        return [(info[0]) for info in sorted_stock_info]
    def get_action(self, observation, info):
        
        # Student code here
        match self.policy:
            case 1:
                if not np.array_equal(self.observation_old, observation):
                    self.stock_sort = self._sort_stocks(observation["stocks"])
                    self.item_sort = self._sort_item1(observation["products"])
                    # self.observation_old = observation

                list_prods = self.item_sort
                pos_x, pos_y = None, None
                stock_idx = -1
                best_pos = None

                for i, stock in self.stock_sort:
                    for prod in list_prods:
                        original_size = prod["size"]
                        rotated_size = [original_size[1], original_size[0]]

                        for prod_size in [original_size, rotated_size]:
                            stock_w, stock_h = self._get_stock_size_(stock)
                            prod_w, prod_h = prod_size

                            if prod_w > stock_w or prod_h > stock_h:
                                continue

                            x, y = 0, 0
                            while y < stock_h - prod_h + 1:
                                if self._can_place_(stock, (x, y), prod_size):
                                    best_pos = (x, y)
                                    while y < stock_h - prod_h and self._can_place_(stock, (x, y + 1), prod_size):
                                        y += 1
                                        best_pos = (x, y)
                                    break

                                x += 1
                                if x > stock_w - prod_w:
                                    break
                            if best_pos is not None:
                                break
                        if best_pos is not None:
                            pos_x, pos_y = best_pos
                            stock_idx = i
                            break
                    if best_pos is not None:
                        break

                # print(f"Final Placement: Stock {stock_idx}, Product Size {prod_size}, Position ({pos_x}, {pos_y})")
                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

            case 2:
                if self.observation_old != observation:
                    self.stock_sort = self._sort_stocks(observation["stocks"])
                    self.item_sort = self._sort_item2(observation["products"])
                    # self.observation_old = observation
                list_prods = self.item_sort
                prod_size = [0, 0]
                stock_idx = -1
                pos_x, pos_y = None, None
                best_pos = None
                x, y = 0, 0
                for i, stock in self.stock_sort:
                    for prod in list_prods:
                        if prod["quantity"] > 0:
                            prod_size = prod["size"]
                            stock_w, stock_h = self._get_stock_size_(stock)
                            prod_w, prod_h = prod_size
                            if not self._can_place_(stock, (x, y), prod_size):
                                continue
                            best_pos = (x, y)
                            while y < stock_h - prod_h + 1:
                                if self._can_place_(stock, (x, y + 1), prod_size):
                                    best_pos = (x, y + 1)
                                    y = y + 1
                                    continue
                                x = x + 1
                                if x > stock_w - prod_w: break
                                if self._can_place_(stock, (x, y), prod_size):
                                    best_pos = (x, y)
                                else: break
                                
                            if best_pos is not None:
                                break
                    if best_pos is not None:
                        pos_x, pos_y = best_pos
                        stock_idx = i
                        break
                # print(f"Final Placement: Stock {stock_idx}, Product Size {prod_size}, Position ({pos_x}, {pos_y})")
                return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    # Student code here
    # You can add more functions if needed
