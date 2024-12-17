from policy import Policy
import numpy as np
import random
import math

class Policy_2352065_2352256_2352504_2252600_2350034:
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = brandandbound()
        elif policy_id == 2:
            self.policy = greedy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)


class brandandbound(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        products = observation["products"]
        total_qty = sum(p["quantity"] for p in products)
        if total_qty == 0:
            return {"stock_idx":0,"size":[1,1],"position":(0,0)}

        current_products_key = tuple((p["size"][0], p["size"][1], p["quantity"]) for p in products)
        if current_products_key != getattr(self, 'last_products_info', None):
            self.last_products_info = current_products_key
            self.actions_queue = []
            self.solved = False
            self.run_branch_and_bound(observation)

        if not getattr(self, 'solved', False) or len(getattr(self, 'actions_queue', [])) == 0:
            return {"stock_idx":0,"size":[1,1],"position":(0,0)}

        return self.actions_queue.pop(0)


    def run_branch_and_bound(self, observation):
        stocks = observation["stocks"]
        products = list(observation["products"])
        stock_info = []
        for i, s in enumerate(stocks):
            w = np.sum(np.any(s != -2, axis=1))
            h = np.sum(np.any(s != -2, axis=0))
            stock_info.append((i, w, h))
        stock_info.sort(key=lambda x: x[1]*x[2])
        products = [{"size":p["size"].copy(),"quantity":p["quantity"]} for p in products]
        solution = []

        def total_free_area(stocks):
            area = 0
            for s in stocks:
                stock_w = np.sum(np.any(s != -2, axis=1))
                stock_h = np.sum(np.any(s != -2, axis=0))
                free_cells = np.sum(s[:stock_w,:stock_h] == -1)
                area += free_cells
            return area

        def try_place_product(stock, w, h):
            stock_w = np.sum(np.any(stock != -2, axis=1))
            stock_h = np.sum(np.any(stock != -2, axis=0))
            if w<=stock_w and h<=stock_h:
                for x in range(stock_w - w + 1):
                    for y in range(stock_h - h + 1):
                        if self._can_place_(stock,(x,y),(w,h)):
                            return (x,y,(w,h))
            if h<=stock_w and w<=stock_h:
                for x in range(stock_w - h + 1):
                    for y in range(stock_h - w + 1):
                        if self._can_place_(stock,(x,y),(h,w)):
                            return (x,y,(h,w))
            return None

        def backtrack(prod_idx):
            if prod_idx == len(products):
                return True
            if sum(p["quantity"]*p["size"][0]*p["size"][1] for p in products[prod_idx:]) > total_free_area(current_stocks):
                return False
            p = products[prod_idx]
            if p["quantity"] == 0:
                return backtrack(prod_idx+1)
            needed = p["quantity"]
            for n in range(needed):
                placed = False
                for (original_idx, w_s, h_s) in stock_info:
                    res = try_place_product(current_stocks[original_idx], p["size"][0], p["size"][1])
                    if res is not None:
                        x, y, chosen_size = res
                        cw, ch = chosen_size
                        stock = current_stocks[original_idx]
                        stock[x:x+cw,y:y+ch] = prod_idx
                        solution.append({"stock_idx":original_idx,"size":[cw,ch],"position":(x,y)})
                        p["quantity"] -= 1
                        if backtrack(prod_idx if p["quantity"]>0 else prod_idx+1):
                            placed = True
                            break
                        else:
                            stock[x:x+cw,y:y+ch] = -1
                            solution.pop()
                            p["quantity"] += 1
                if not placed:
                    return False
            return True

        current_stocks = [s.copy() for s in stocks]
        current_stocks = tuple(current_stocks)
        if backtrack(0):
            self.actions_queue = solution
            self.solved = True
        else:
            self.solved = False


class greedy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        products = observation["products"]
        total_qty = sum(p["quantity"] for p in products)
        if total_qty == 0:
            return {"stock_idx":0,"size":[1,1],"position":(0,0)}

        current_products_key = tuple((p["size"][0], p["size"][1], p["quantity"]) for p in products)
        if current_products_key != getattr(self, 'last_products_info', None):
            self.last_products_info = current_products_key
            self.actions_queue = []
            self.solved = False
            self.run_greedy(observation)

        if not getattr(self, 'solved', False) or len(getattr(self, 'actions_queue', [])) == 0:
            return {"stock_idx":0,"size":[1,1],"position":(0,0)}

        return self.actions_queue.pop(0)

    def run_greedy(self, observation):
        stocks = observation["stocks"]
        products = [{"size":p["size"].copy(),"quantity":p["quantity"]} for p in observation["products"]]
        stock_info = []
        for i, s in enumerate(stocks):
            w, h = self._extract_usable_dimensions(s)
            stock_info.append((i, w, h))
        stock_info.sort(key=lambda x: x[1]*x[2])
        current_stocks = [s.copy() for s in stocks]
        solution = []

        def can_place(stock, position, prod_size):
            x, y = position
            w, h = prod_size
            return np.all(stock[x:x+w,y:y+h] == -1)

        def try_place_in_stock(stock, original_idx, w, h):
            stock_w, stock_h = self._extract_usable_dimensions(stock)
            if w <= stock_w and h <= stock_h:
                for x in range(stock_w - w + 1):
                    for y in range(stock_h - h + 1):
                        if can_place(stock,(x,y),(w,h)):
                            return (original_idx,x,y,(w,h))
            if h <= stock_w and w <= stock_h:
                for x in range(stock_w - h + 1):
                    for y in range(stock_h - w + 1):
                        if can_place(stock,(x,y),(h,w)):
                            return (original_idx,x,y,(h,w))
            return None

        def select_largest_product(products):
            max_w, max_h = -1, -1
            idx = -1
            for i, p in enumerate(products):
                if p["quantity"] > 0:
                    pw, ph = p["size"]
                    if ph > pw:
                        pw, ph = ph, pw
                    if pw > max_w and ph > max_h:
                        max_w, max_h = pw, ph
                        idx = i
            return idx, (max_w, max_h)

        while True:
            idx, (pw, ph) = select_largest_product(products)
            if idx == -1:
                self.solved = True
                self.actions_queue = solution
                return
            psize = products[idx]["size"]
            placed = False
            for (original_idx, w_s, h_s) in stock_info:
                res = try_place_in_stock(current_stocks[original_idx], original_idx, psize[0], psize[1])
                if res is not None:
                    st_i, x, y, chosen_size = res
                    cw, ch = chosen_size
                    current_stocks[st_i][x:x+cw,y:y+ch] = idx
                    solution.append({"stock_idx":st_i,"size":[cw,ch],"position":(x,y)})
                    products[idx]["quantity"] -= 1
                    placed = True
                    break
            if not placed:
                self.solved = True
                self.actions_queue = solution
                return

    def _extract_usable_dimensions(self, stock):
        usable_w = np.count_nonzero(np.any(stock != -2, axis=1))
        usable_h = np.count_nonzero(np.any(stock != -2, axis=0))
        return usable_w, usable_h

    def _can_place_(self, stock, position, prod_size):
        x, y = position
        w, h = prod_size
        return np.all(stock[x:x+w,y:y+h] == -1)
