from policy import Policy
import numpy as np
import time
import heapq
import random

class Node:
    def __init__(self, stocks, demands, waste, assignments):
        self.stocks = stocks  # List of current stock states
        self.demands = demands  # Remaining demands
        self.waste = waste  # Total waste so far
        self.assignments = assignments  # Assignments made
    
    def __lt__(self, other):
        return self.waste < other.waste

# Greedy
#greedy
class Greedy(Policy):
    def __init__(self):
        pass

    def get_action(self, observation, info):
        # Lấy danh sách sản phẩm và các kho
        list_prods = observation["products"]
        stocks = observation["stocks"]

        prod_size = [0, 0]  # Kích thước sản phẩm đang được xét
        stock_idx = -1  # Chỉ số kho được chọn
        pos_x, pos_y = None, None  # Vị trí cắt trong kho

        # Sắp xếp sản phẩm theo diện tích giảm dần (ưu tiên cắt sản phẩm lớn trước)
        sorted_prods = sorted(
            list_prods, key=lambda p: p["size"][0] * p["size"][1], reverse=True
        )

        # Duyệt qua các sản phẩm đã sắp xếp
        for prod in sorted_prods:
            if prod["quantity"] > 0:  # Nếu còn số lượng sản phẩm cần cắt
                original_size = prod["size"]
                rotated_size = [original_size[1], original_size[0]]  # Kích thước xoay

                # Duyệt cả hai kích thước (nguyên bản và xoay)
                for prod_size in [original_size, rotated_size]:
                    prod_w, prod_h = prod_size

                    # Duyệt qua từng kho để tìm vị trí cắt hợp lệ
                    for i, stock in enumerate(stocks):
                        stock_w, stock_h = self._get_stock_size_(stock)  # Kích thước của kho

                        # Bỏ qua nếu sản phẩm không thể vừa với kho
                        if stock_w < prod_w or stock_h < prod_h:
                            continue

                        # Tìm vị trí đầu tiên (góc trên bên trái) trong kho mà sản phẩm có thể cắt
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):  # Kiểm tra vị trí hợp lệ
                                    stock_idx = i  # Lưu chỉ số kho
                                    pos_x, pos_y = x, y  # Lưu vị trí cắt
                                    break
                            if pos_x is not None and pos_y is not None:  # Nếu tìm thấy vị trí hợp lệ
                                break

                        if pos_x is not None and pos_y is not None:  # Thoát khỏi vòng lặp nếu đã tìm thấy
                            break

                    if pos_x is not None and pos_y is not None:  # Nếu tìm thấy vị trí hợp lệ, không cần kiểm tra kích thước còn lại
                        break

                if pos_x is not None and pos_y is not None:  # Thoát nếu tìm thấy vị trí hợp lệ
                    break

        # Trả về hành động cắt
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

# Hybrid Heuristic-B&B
import time
import numpy as np

class BottomLeftHeuristic:
    def __init__(self, allow_rotation=False):
        self.allow_rotation = allow_rotation

    def run(self, stocks, demands):
        items = []
        for pid, d in demands.items():
            for _ in range(d["count"]):
                items.append((pid, d["size"]))
        # Sort by area descending
        items.sort(key=lambda x: x[1][0]*x[1][1], reverse=True)

        solution_stocks = [s.copy() for s in stocks]
        solution_demands = {k: v.copy() for k,v in demands.items()}
        actions = []

        for pid, size in items:
            placed = False
            for sid, stock in enumerate(solution_stocks):
                pos = self.bottom_left_place(stock, size)
                if pos is not None:
                    self.place_product(solution_stocks[sid], pos, size)
                    solution_demands[pid]["count"] -= 1
                    actions.append({"stock_idx": sid, "size": size, "position": pos})
                    placed = True
                    break
                if self.allow_rotation:
                    rotated_size = (size[1], size[0])
                    if rotated_size != size:
                        pos = self.bottom_left_place(stock, rotated_size)
                        if pos is not None:
                            self.place_product(solution_stocks[sid], pos, rotated_size)
                            solution_demands[pid]["count"] -= 1
                            actions.append({"stock_idx": sid, "size": rotated_size, "position": pos})
                            placed = True
                            break
            # If not placed, we do nothing (some demands remain unfilled)
        return {
            "stocks": solution_stocks,
            "demands": solution_demands,
            "actions": actions
        }

    def bottom_left_place(self, stock, size):
        h, w = stock.shape
        ph, pw = size
        free_mask = (stock == -1)
        for i in range(h - ph + 1):
            for j in range(w - pw + 1):
                if np.all(free_mask[i:i+ph, j:j+pw]):
                    return (i, j)
        return None

    def place_product(self, stock, pos, size):
        i, j = pos
        ph, pw = size
        stock[i:i+ph, j:j+pw] = 1

class HybridBnB:
    def __init__(self, allow_rotation=False, bnb_time_limit=5.0):
        self.bnb_time_limit = bnb_time_limit
        self.episode_initialized = False
        self.precomputed_actions = []
        self.allow_rotation = allow_rotation
        self.best_solution = None
        self.best_trim_loss = float('inf')
        self.start_time = None

    def get_action(self, observation, info):
        if not self.episode_initialized:
            self.initialize_episode(observation, info)
            self.episode_initialized = True

        if self.precomputed_actions:
            return self.precomputed_actions.pop(0)

        action = self.find_feasible_action(observation)
        if action:
            return action
        return {"stock_idx": 0, "size": (1,1), "position": (0,0)}
    
    def initialize_episode(self, observation, info):
        stocks = observation["stocks"]
        products = observation["products"]
        demands = {i: {"size": tuple(p["size"]), "count": p["quantity"]} for i, p in enumerate(products)}

        # Get initial solution using heuristic
        heuristic = BottomLeftHeuristic(allow_rotation=self.allow_rotation)
        heuristic_solution = heuristic.run(stocks, demands)

        # Evaluate initial solution
        initial_trim = self.calculate_trim_loss(heuristic_solution["stocks"])
        self.best_solution = heuristic_solution
        self.best_trim_loss = initial_trim

        # Perform BnB search to try improving the solution
        # Note: If your problem instance is large, you might need more sophisticated pruning
        self.start_time = time.perf_counter()
        self.branch_and_bound(self.best_solution)

        # After BnB, we have best_solution improved if possible
        self.precomputed_actions = self.best_solution.get("actions", [])

    def branch_and_bound(self, current_solution):
        # Check time limit
        if (time.perf_counter() - self.start_time) > self.bnb_time_limit:
            return

        # If all demands are met, no need to branch further
        if self.all_placed(current_solution):
            return

        # Generate children solutions
        children = self.generate_children(current_solution)
        for child in children:
            child_trim = self.calculate_trim_loss(child["stocks"])
            # Prune if not better
            if child_trim >= self.best_trim_loss:
                continue
            # Update best solution if improved
            if child_trim < self.best_trim_loss:
                self.best_trim_loss = child_trim
                self.best_solution = child
            # Recurse deeper
            self.branch_and_bound(child)

    def generate_children(self, solution):
        # Find the next product that still needs to be placed
        for prod_id, d in solution["demands"].items():
            if d["count"] > 0:
                # Try placing this product in all feasible positions and possibly rotated
                return self.try_all_placements(solution, prod_id, d["size"])
        return []

    def try_all_placements(self, solution, prod_id, size):
        children = []
        ph, pw = size
        for stock_idx, stock in enumerate(solution["stocks"]):
            free_mask = (stock == -1)
            H, W = stock.shape
            for i in range(H - ph + 1):
                for j in range(W - pw + 1):
                    if np.all(free_mask[i:i+ph, j:j+pw]):
                        # Create a new child solution
                        child = self.copy_solution(solution)
                        self.place_product(child["stocks"][stock_idx], (i,j), size)
                        child["demands"][prod_id]["count"] -= 1
                        child["actions"].append({"stock_idx": stock_idx, "size": size, "position": (i,j)})
                        children.append(child)

            # If rotation allowed, try rotated size if it's different
            if self.allow_rotation and (ph != pw):
                rotated_size = (pw, ph)
                for i in range(H - rotated_size[0] + 1):
                    for j in range(W - rotated_size[1] + 1):
                        if np.all(free_mask[i:i+rotated_size[0], j:j+rotated_size[1]]):
                            child = self.copy_solution(solution)
                            self.place_product(child["stocks"][stock_idx], (i,j), rotated_size)
                            child["demands"][prod_id]["count"] -= 1
                            child["actions"].append({"stock_idx": stock_idx, "size": rotated_size, "position": (i,j)})
                            children.append(child)

        return children

    def all_placed(self, solution):
        # Check if all demands are met
        return all(d["count"] == 0 for d in solution["demands"].values())

    def copy_solution(self, solution):
        return {
            "stocks": [s.copy() for s in solution["stocks"]],
            "demands": {k: v.copy() for k,v in solution["demands"].items()},
            "actions": solution.get("actions", []).copy()
        }

    def calculate_trim_loss(self, stocks):
        total_free = 0
        total_cells = 0
        for s in stocks:
            total_free += np.count_nonzero(s == -1)
            total_cells += s.size
        if total_cells > 0:
            return total_free / total_cells
        return 0

    def place_product(self, stock, pos, size):
        i, j = pos
        ph, pw = size
        stock[i:i+ph, j:j+pw] = 1

    def find_feasible_action(self, observation):
        stocks = observation["stocks"]
        products = observation["products"]
        for i, p in enumerate(products):
            if p["quantity"] > 0:
                size = tuple(p["size"])
                # Try to place without rotation
                for si, stock in enumerate(stocks):
                    pos = self.find_position(stock, size)
                    if pos is not None:
                        return {"stock_idx": si, "size": size, "position": pos}
                # If allow rotation, try rotated size
                if self.allow_rotation and (size[0] != size[1]):
                    rotated_size = (size[1], size[0])
                    for si, stock in enumerate(stocks):
                        pos = self.find_position(stock, rotated_size)
                        if pos is not None:
                            return {"stock_idx": si, "size": rotated_size, "position": pos}
        return None

    def find_position(self, stock, size):
        h, w = stock.shape
        ph, pw = size
        free_mask = (stock == -1)
        for i in range(h - ph + 1):
            for j in range(w - pw + 1):
                if np.all(free_mask[i:i+ph, j:j+pw]):
                    return (i, j)
        return None

# Integrate into your Policy
class Policy2352089_2352158_2252783_2353288(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        super().__init__()

        if policy_id == 1:
            self.policy_impl = Greedy()
        elif policy_id == 2:
            self.policy_impl = HybridBnB(allow_rotation=True)
        
    def get_action(self, observation, info):
        return self.policy_impl.get_action(observation, info)
