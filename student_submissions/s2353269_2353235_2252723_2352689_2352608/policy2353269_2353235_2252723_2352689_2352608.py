import numpy as np
from policy import Policy

class Policy2353259_2353235_2252723_2352689_2352608(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        
        self.policy_id = policy_id
        
        if policy_id == 1:
            pass
        elif policy_id == 2:
            self.rotation_enabled = True
            self.score_edge = 15
            self.score_gap = 8
            self.score_corner = 20
    
    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self._get_action_policy1(observation, info)
        else:
            return self._get_action_policy2(observation, info)

    def _get_action_policy1(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]
        
        valid_prods = [prod for prod in list_prods if prod["quantity"] > 0]
        if not valid_prods:
            return {"size": [0,0], "stock_idx": -1, "position": [0,0]}
        
        valid_prods.sort(key=lambda x: (
            -(x["size"][0] * x["size"][1]),
            -min(x["size"][0]/x["size"][1], x["size"][1]/x["size"][0])
        ))
        
        heuristic_solution = None
        for prod in valid_prods[:4]:
            sizes = [
                prod["size"],
                [prod["size"][1], prod["size"][0]]
            ]
            
            for size in sizes:
                for stock_idx, stock in enumerate(stocks):
                    if np.sum(stock == -1) < size[0] * size[1]:
                        continue
                        
                    pos = self._find_best_position(stock, size)
                    if pos is not None:
                        heuristic_solution = {
                            "size": size,
                            "stock_idx": stock_idx,
                            "position": pos
                        }
                        break
                if heuristic_solution:
                    break
            if heuristic_solution:
                break
            
        return heuristic_solution or {"size": [0,0], "stock_idx": -1, "position": [0,0]}

    def _get_action_policy2(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]
        
        available_products = [p for p in products if p["quantity"] > 0]
        available_products.sort(key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        optimal_choice = {"stock_idx": -1, "size": (0, 0), "position": (None, None)}
        best_leftover_space = float('inf')

        for product in available_products:
            candidate_placement, candidate_score = self._find_optimal_placement_for_product(product, stocks, best_leftover_space)
            
            if candidate_placement["stock_idx"] != -1:
                optimal_choice = candidate_placement
                best_leftover_space = candidate_score
                break

        return optimal_choice


# METHOD FOR POLICY 2
    def _find_optimal_placement_for_product(self, product, stocks, current_best):
        best_placement = {"stock_idx": -1, "size": (0, 0), "position": (None, None)}
        best_score = current_best

        p_w, p_h = product["size"]
        
        orientations = [(p_w, p_h)]
        if self.rotation_enabled and p_w != p_h:
            orientations.append((p_h, p_w))

        all_candidates = []

        for s_idx, stock in enumerate(stocks):
            stock_w, stock_h = self._get_stock_size_(stock)
            for (try_w, try_h) in orientations:
                if try_w <= stock_w and try_h <= stock_h:
                    candidates = self._get_all_candidates(stock, s_idx, try_w, try_h, stock_w, stock_h)
                    all_candidates.extend(candidates)

        for candidate in all_candidates:
            if candidate["leftover"] < best_score:
                best_score = candidate["leftover"]
                best_placement = {
                    "stock_idx": candidate["stock_idx"],
                    "size": candidate["size"],
                    "position": candidate["position"]
                }

        return best_placement, best_score

    def _get_all_candidates(self, stock, stock_index, width, height, stock_w, stock_h):
        candidates = []
        for x_coord, y_coord in self._generate_positions(stock_w, stock_h, width, height):
            if self._is_valid_position(stock, (x_coord, y_coord), (width, height)):
                if self._can_place_(stock, (x_coord, y_coord), (width, height)):
                    leftover = (stock_w * stock_h) - (width * height)
                    candidates.append({
                        "stock_idx": stock_index,
                        "size": (width, height),
                        "position": (x_coord, y_coord),
                        "leftover": leftover
                    })
        return candidates

    def _generate_positions(self, stock_w, stock_h, item_w, item_h):
        y_positions = range(stock_h - item_h + 1)
        x_positions = range(stock_w - item_w + 1)
        for y in y_positions:
            for x in x_positions:
                yield x, y


# METHOD FOR POLICY 1
    def _find_valid_positions(self, stock, prod_size):
        stock_h, stock_w = stock.shape
        prod_h, prod_w = prod_size
        positions = []
        
        corners = [(0,0), (0,stock_w-prod_w), (stock_h-prod_h,0), (stock_h-prod_h,stock_w-prod_w)]
        for x, y in corners:
            if self._is_valid_position(stock, (x,y), prod_size):
                positions.append((x,y))
        
        for x in range(0, stock_h - prod_h + 1):
            if self._is_valid_position(stock, (x,0), prod_size):
                positions.append((x,0))
            if self._is_valid_position(stock, (x,stock_w-prod_w), prod_size):
                positions.append((x,stock_w-prod_w))
            
        for y in range(0, stock_w - prod_w + 1):
            if self._is_valid_position(stock, (0,y), prod_size):
                positions.append((0,y))
            if self._is_valid_position(stock, (stock_h-prod_h,y), prod_size):
                positions.append((stock_h-prod_h,y))
        
        for x in range(1, stock_h - prod_h):
            for y in range(1, stock_w - prod_w):
                if (x,y) not in positions and self._is_valid_position(stock, (x,y), prod_size):
                    positions.append((x,y))
        
        return positions

    def _is_valid_position(self, stock, pos, size):
        x, y = pos
        h, w = size
        if x + h > stock.shape[0] or y + w > stock.shape[1]:
            return False
        return np.all(stock[x:x+h, y:y+w] == -1)

    def _find_best_position(self, stock, size):
        best_pos = None
        best_score = float('-inf')
        
        positions = self._find_valid_positions(stock, size)
            
        for pos in positions:
            score = 0
            x, y = pos
            h, w = size
            stock_h, stock_w = stock.shape
            
            if (x == 0 or x + h == stock_h) and (y == 0 or y + w == stock_w):
                score += 1000  # Góc
            elif x == 0 or x + h == stock_h or y == 0 or y + w == stock_w:
                score += 500  # Cạnh
            
            neighbors = 0
            if x > 0 and np.any(stock[x-1, y:y+w] != -1): neighbors += 1
            if x + h < stock_h and np.any(stock[x+h, y:y+w] != -1): neighbors += 1
            if y > 0 and np.any(stock[x:x+h, y-1] != -1): neighbors += 1
            if y + w < stock_w and np.any(stock[x:x+h, y+w] != -1): neighbors += 1
            score += neighbors * 300
            
            if score > best_score:
                best_score = score
                best_pos = pos
                
        return best_pos

    