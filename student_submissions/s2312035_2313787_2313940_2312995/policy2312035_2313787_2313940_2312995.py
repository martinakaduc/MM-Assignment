from policy import Policy
import numpy as np
from scipy.optimize import linprog


class Policy2312035_2313787_2313940_2312995(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2, 3], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 3:
            self.policy = Collumn_Gen() 
        elif policy_id == 2:
            self.policy = IterativeBacktrackingPolicy()
        elif policy_id == 1:
            self.policy = BestFitPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
    # Student code here
    # You can add more functions if needed
class Collumn_Gen(Policy):
    def __init__(self):
        super().__init__()
        self.patterns = {"collumn":[],"stock":[], "stock_idx":[],"Is_used":[]}
        self.can_use_stock = {"stock":[], "idx": []}
        self.num_item = None
        self.init = True
    
   
   
    def get_action(self, observation, info):
        stocks  = observation["stocks"]
        items = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        sizes = [item["size"] for item in items]
        num_item=len(sizes)
        sizes = [item["size"] for item in items]
        if self.init or num_item != self.num_item:
            self.patterns = {"collumn":[],"stock":[], "stock_idx":[],"Is_used":[]}
            self.can_use_stock = {"stock":[], "idx": []}
            self.num_item = num_item
            self.init = False
            for i in range(len(stocks)):
                self.can_use_stock["stock"].append(stocks[i])
                self.can_use_stock["idx"].append(i)

            pat_tmp = {"collumn":[],"stock":[],"stock_idx":[]}
            pat_tmp = self.init_patterns(sizes)
            self.patterns["collumn"]= pat_tmp["collumn"]
            self.patterns["stock"] = pat_tmp["stock"]
            self.patterns["stock_idx"]= pat_tmp["stock_idx"]
        
        quantity = np.array([item["quantity"] for item in items])
        while True:
            c = np.array([self.get_size(self._get_stock_size_(stk)) for stk in self.patterns["stock"]])
            A = np.array(self.patterns["collumn"]).T
            b = quantity
            RMP = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, 1), method='highs')
            if RMP.status != 0:
                break

            dual = RMP.ineqlin.marginals if hasattr(RMP.ineqlin, 'marginals') else None

            if dual is None:
                break

            new_pattern, stock_tmp,idx = self.Sub_pricing_prob(dual,sizes)

            if new_pattern is None or any(np.array_equal(new_pattern, p) for p in self.patterns):
                break
            self.patterns["collumn"].append(new_pattern)
            self.patterns["stock"].append(stock_tmp)
            self.patterns["stock_idx"].append(idx)
            self.patterns["Is_used"] = RMP.x
        action = self.action(sizes,quantity)
        return action


    def action(self, sizes,quantity):
        idx = 0
        num_item = len(sizes)
        for pattern in self.patterns["collumn"] :
            for i in range(num_item):
                if(pattern[i] <= 0 or quantity[i]==0):
                    continue
                self.patterns["collumn"][idx][i] -= 1
                pos = self.place_bot_letf(self.patterns["stock"][idx],sizes[i])
                if pos is not None:
                    if np.array_equal(self.can_use_stock["stock"], self.patterns["stock"][idx]):
                        self.can_use_stock["stock"].remove(self.patterns["stock"][idx])
                    if np.array_equal(self.can_use_stock["idx"], self.patterns["stock_idx"][idx]):
                        self.can_use_stock["idx"].remove(self.patterns["stock_idx"][idx])
                    return {"stock_idx": self.patterns["stock_idx"][idx], "size": sizes[i], "position": pos}

                pos = self.place_bot_letf(self.patterns["stock"][idx],sizes[i][::-1])
                if pos is not None:
                    return {"stock_idx": self.patterns["stock_idx"][idx], "size": sizes[i][::-1], "position": pos}
            idx += 1
        return {"stock_idx": -1, "size": [0, 0],"position": (0, 0)}

    
    def Sub_pricing_prob(self, dual, sizes):
        best_pat = None
        Best_Reduced_cost = 0
        stock_of_best = None
        idx = 0
        for stock in self.can_use_stock["stock"]:
            pat_tmp = np.zeros(len(sizes))
            W_stock, H_stock = self._get_stock_size_(stock)
            dp = np.zeros((W_stock + 1, H_stock + 1))
            
            for i in range(len(sizes)):
                w, h = sizes[i]
                if dual[i] <= 0:
                    continue
                if w <= W_stock and h > H_stock: 
                    for x in range(W_stock, w - 1, -1):
                        for y in range(H_stock, h - 1, -1):
                         dp[x][y] = max(dp[x][y], dp[x-w][y - h] + dual[i]) #khong xoay   
                if h <= W_stock and w <= H_stock: 
                    for x in range(W_stock, w - 1, -1):
                        for y in range(H_stock, h - 1, -1):
                         dp[x][y] = max(dp[x][y], dp[x-h][y - w] + dual[i]) #xoay 90 do   

            W,H = W_stock, H_stock

            for i in range(len(sizes) - 1, -1, -1):
                w, h = sizes[i]
                #khong xoay
                if dp[W][H] == dp[W - w][H - h] + dual[i] and (W - w) >= 0 and (H - h) >= 0 and dual[i] > 0:
                    pat_tmp[i] += 1
                    W -= w
                    H -= h 
                #xoay 90 do
                if dp[W][H] == dp[W - h][H - w] + dual[i] and (W - h) >= 0 and (H - w) >= 0 and dual[i] > 0:
                    pat_tmp[i] += 1
                    W -= h
                    H -= w
                  
            reduced_cost = np.dot(pat_tmp,dual) - (W_stock * H_stock) 
            if reduced_cost > Best_Reduced_cost:
                Best_Reduced_cost = reduced_cost
                best_pat = pat_tmp
                stock_of_best = stock
                best_stock_idx = self.can_use_stock["idx"][idx]
            idx+=1
        if Best_Reduced_cost > 0:
            return best_pat, stock_of_best, best_stock_idx
        else:
            return None, None, None


    def get_size(self,size):
        return size[0]*size[1]
    def init_patterns(self, products):
        patterns = {"collumn": [],"stock": [], "stock_idx": []}
        num_prods = len(products)

        for idx, stock in enumerate(self.can_use_stock["stock"]):
            stock_w, stock_h = self._get_stock_size_(stock)

            # Tạo pattern cho từng sản phẩm
            for i, product in enumerate(products):
                prod_w, prod_h = product

                # Tính số lượng tối đa của sản phẩm i có thể cắt từ stock
                max_count = (stock_w // prod_w) * (stock_h // prod_h)

                if max_count > 0:
                    # Tạo một pattern mới
                    pattern = [0] * num_prods
                    pattern[i] = max_count

                    patterns["collumn"].append(pattern)
                    patterns["stock_idx"].append(idx)

        # Lọc trùng lặp
        unique_patterns = {}
        for col, s_idx in zip(patterns["collumn"], patterns["stock_idx"]):
            key = tuple(col + [s_idx])
            if key not in unique_patterns:
                unique_patterns[key] = (col, s_idx)

        patterns["collumn"] = [v[0] for v in unique_patterns.values()]
        patterns["stock_idx"] = [v[1] for v in unique_patterns.values()]
       
        for idx_2 in patterns["stock_idx"]:
            patterns["stock"].append(self.can_use_stock["stock"][idx_2])
        return patterns
    
    
    def place_bot_letf(self,stock,size):
        W_stock,H_stock = self._get_stock_size_(stock)
        w, h = size
        if W_stock < w or H_stock < h:
            return None

        for y in range(H_stock - h + 1):
            for x in range(W_stock - h + 1):
                if self._can_place_(stock, (x, y), size):
                    return(x,y)

        return None
        
class IterativeBacktrackingPolicy(Policy):
    def __init__(self, max_iterations=10000):
        self.max_iterations = max_iterations  # Maximum number of iterations allowed

    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        # Filter products with quantity > 0 and sort by size (largest first)
        valid_products = sorted(
            [p for p in list_prods if p["quantity"] > 0],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True,
        )

        # Initialize iterative backtracking
        solution = []
        if self._iterative_backtrack(valid_products, stocks, solution):
            return solution[0]  # Return the first valid action
        else:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _iterative_backtrack(self, products, stocks, solution):
        """Iterative backtracking with progress tracking and iteration limits."""
        stack = [(0, products, stocks, solution)]  # Stack for backtracking
        iterations = 0  # Iteration counter
        progress = 0  # Tracks progress in placing products
        max_stagnation = self.max_iterations // 10  # Allow up to 10% stagnation

        while stack:
            if iterations >= self.max_iterations:
                print("Max iterations reached. Terminating epoch.")
                return False

            current_idx, remaining_products, current_stocks, current_solution = stack.pop()
            iterations += 1

            # Base case: all products are placed
            if current_idx == len(products):
                solution.extend(current_solution)
                return True

            current_product = products[current_idx]
            prod_w, prod_h = current_product["size"]

            # Try to place the product in all valid stocks
            placed = False
            for stock_idx, stock in enumerate(current_stocks):
                stock_w, stock_h = self._get_stock_size_(stock)

                # Skip this stock if the product cannot fit in either orientation
                if stock_w < prod_w and stock_h < prod_h and stock_w < prod_h and stock_h < prod_w:
                    continue

                # Try to place in default orientation
                for x, y in self._generate_positions(stock, (prod_w, prod_h)):
                    if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                        # Simulate placement
                        self._place_product(stock, (x, y), (prod_w, prod_h))
                        stack.append((current_idx + 1, remaining_products[1:], current_stocks, current_solution + [{
                            "stock_idx": stock_idx,
                            "size": [prod_w, prod_h],
                            "position": (x, y),
                        }]))
                        self._remove_product(stock, (x, y), (prod_w, prod_h))
                        placed = True
                        break

                if placed:
                    break

                # Try to place in rotated orientation
                for x, y in self._generate_positions(stock, (prod_h, prod_w)):
                    if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                        # Simulate placement
                        self._place_product(stock, (x, y), (prod_h, prod_w))
                        stack.append((current_idx + 1, remaining_products[1:], current_stocks, current_solution + [{
                            "stock_idx": stock_idx,
                            "size": [prod_h, prod_w],
                            "position": (x, y),
                        }]))
                        self._remove_product(stock, (x, y), (prod_h, prod_w))
                        placed = True
                        break

                if placed:
                    break

            if not placed:
                progress += 1

            # Exit if progress stagnates for too many iterations
            if progress >= max_stagnation:
                print("Progress stagnated. Terminating epoch.")
                return False

        return False

    def _generate_positions(self, stock, prod_size):
        """Generate candidate positions for placing the product in the stock."""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        step_x = max(1, prod_w // 2)
        step_y = max(1, prod_h // 2)
        for x in range(0, stock_w - prod_w + 1, step_x):
            for y in range(0, stock_h - prod_h + 1, step_y):
                yield x, y

    def _place_product(self, stock, position, size):
        """Simulate placing a product in the stock."""
        x, y = position
        w, h = size
        stock[x:x + w, y:y + h] = 1  # Mark the cells as occupied

    def _remove_product(self, stock, position, size):
        """Simulate removing a product from the stock."""
        x, y = position
        w, h = size
        stock[x:x + w, y:y + h] = -1  # Reset the cells to unoccupied   
class BestFitPolicy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]

        # Pre-compute stock dimensions
        stock_sizes = [self._get_stock_size_(stock) for stock in stocks]

        # Sort products by area in descending order
        sorted_products = sorted(
            (p for p in products if p["quantity"] > 0),
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True,
        )

        best_action = {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        for product in sorted_products:
            prod_size = product["size"]
            prod_w, prod_h = prod_size

            # Initialize variables to track the best placement
            min_waste = float("inf")
            best_stock_idx = -1
            best_position = None

            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = stock_sizes[stock_idx]

                # Skip stocks that can't fit the product
                if stock_w < prod_w or stock_h < prod_h:
                    continue

                # Iterate over valid positions in the stock
                for x in range(0, stock_w - prod_w + 1, max(1, prod_w // 2)):
                    for y in range(0, stock_h - prod_h + 1, max(1, prod_h // 2)):
                        if self._can_place_(stock, (x, y), prod_size):
                            # Calculate remaining space after placement
                            remaining_space = (stock_w * stock_h) - (prod_w * prod_h)

                            # Early pruning: stop if we find a near-perfect fit
                            if remaining_space == 0:
                                return {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y),
                                }

                            # Update best placement if this one is better
                            if remaining_space < min_waste:
                                min_waste = remaining_space
                                best_stock_idx = stock_idx
                                best_position = (x, y)

            # Update the best action if a placement is found
            if best_stock_idx != -1:
                return {
                    "stock_idx": best_stock_idx,
                    "size": prod_size,
                    "position": best_position,
                }

        # If no valid placement is found, return the default action
        return best_action



    def _place_product(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        stock[pos_x: pos_x + prod_w, pos_y: pos_y + prod_h] = 1  # Mark the area as occupied
        return stock

    # def _evaluate_quality(self, state):
    #     # Calculate total quality from placements
    #     return sum(prod["quantity"] for prod_idx, stock_idx, position in state["placements"])
    #Test thu code
