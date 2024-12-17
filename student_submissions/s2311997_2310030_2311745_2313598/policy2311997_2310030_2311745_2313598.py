from policy import Policy
import numpy as np
import random
from scipy.optimize import linprog


class Policy2311997_2310030_2311745_2313598(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        
        # Student code here
        if policy_id == 1:
            self.policy = GeneticAlgorithm()
        elif policy_id == 2:
            self.policy = ColumnGeneration()
            
    
    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
    
class MaximalRectangle():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.free_rects = [{"x": int(0), "y": int(0), "width": int(width), "height": int(height)}]
        
    
    def split_rect(self, free_rect, used_rect):
        # x, y, w, h = int(free_rect["x"]), int(free_rect["y"]), int(free_rect["width"]), int(free_rect["height"])
        # px, py, pw, ph = int(placed_rect["x"]), int(placed_rect["y"]), int(placed_rect["width"]), int(placed_rect["height"])

        # new_rect = []

        # # Tính các phần không gian trống còn lại
        # if px > x:  # Left-side
        #     new_rect.append({"x": x, "y": y, "width": px - x, "height": h})
        # if py > y:  # Top-side
        #     new_rect.append({"x": x, "y": y, "width": w, "height": py - y})
        # if px + pw < x + w:  # Right-side
        #     new_rect.append({"x": px + pw, "y": y, "width": (x + w) - (px + pw), "height": h})
        # if py + ph < y + h:  # Bottom-side
        #     new_rect.append({"x": x, "y": py + ph, "width": w, "height": (y + h) - (py + ph)})

        # return new_rect
        
        new_free_rects = []
        if used_rect["width"] < free_rect["width"]:
            fw = free_rect["width"] - used_rect["width"]
            fh = free_rect["height"]
            fx = free_rect["x"] + used_rect["width"]
            fy = free_rect["y"]
            new_free_rects.append({"x": fx, "y": fy, "width": fw, "height": fh})
            
        if used_rect["height"] < free_rect["height"]:
            fw = free_rect["width"]
            fh = free_rect["height"] - used_rect["height"]
            fx = free_rect["x"]
            fy = free_rect["y"] + used_rect["height"]
            new_free_rects.append({"x": fx, "y": fy, "width": fw, "height": fh})
            
        return new_free_rects
        
    def remove_contained_rect(self):
        filtered_rect = []
        for r1 in self.free_rects:
            contained = False
            for r2 in self.free_rects:
                if r1 == r2:
                    continue
                if (r1["x"] >= r2["x"] and r1["y"] >= r2["y"] and
                    r1["x"] + r1["width"] <= r2["x"] + r2["width"] and
                    r1["y"] + r1["height"] <= r2["y"] + r2["height"]):
                    contained = True
                    break
            if not contained:
                filtered_rect.append(r1)
        self.free_rects = filtered_rect
        
    def find_place(self, width, height):
        best_rect = None
        best_placed = None
        
        ######################################### Best Area Fit ########################################
        best_short_side = float('inf')
        best_area_diff = float('inf')
        
        for rect in self.free_rects:
            # Check without rotation
            if width <= rect["width"] and height <= rect["height"]:
                area_diff = rect["width"] * rect["height"] - width * height
                short_side = min(rect["width"] - width, rect["height"] - height)
                if(area_diff < best_area_diff or (area_diff == best_area_diff and short_side < best_short_side)):
                    best_area_diff = area_diff
                    best_short_side = short_side
                    best_rect = rect
                    best_placed = {"x": rect["x"], "y": rect["y"], "width": width, "height": height}
            
            # Check with rotation
            if height <= rect["width"] and width <= rect["height"]:
                area_diff = rect["width"] * rect["height"] - width * height
                short_side = min(rect["width"] - height, rect["height"] - width)
                if (area_diff < best_area_diff or (area_diff == best_area_diff and short_side < best_short_side)):
                    best_area_diff = area_diff
                    best_short_side = short_side
                    best_rect = rect
                    best_placed = {"x": rect["x"], "y": rect["y"], "width": height, "height": width}
        return best_rect, best_placed
    
        ################################### Best Short Side Fit #############################
        # best_long_side = float('inf')  # Khởi tạo nhỏ nhất để tìm max chiều dài dư thừa
        # best_short_side = float('inf')
        # for rect in self.free_rects: 
        #     if width <= rect["width"] and height <= rect["height"]:
        #         long_side = max(rect["width"] - width, rect["height"] - height)  # Tính chiều dài dư thừa theo chiều ngang
        #         short_side = min(rect["width"] - width, rect["height"] - height)
        #         if (short_side < best_short_side) or (short_side == best_short_side and long_side < best_long_side):
        #             best_long_side = long_side
        #             best_short_side = short_side
        #             best_rect = rect
        #             best_placed = {"x": rect["x"], "y": rect["y"], "width": width, "height": height}
            
        #     if height <= rect["width"] and width <= rect["height"]:
        #         long_side = max(rect["width"] - height, rect["height"] - width)  # Tính chiều dài dư thừa theo chiều dọc
        #         short_side = min(rect["width"] - height, rect["height"] - width)
        #         if (short_side < best_short_side) or (short_side == best_short_side and long_side < best_long_side):
        #             best_long_side = long_side
        #             best_short_side = short_side
        #             best_rect = rect
        #             best_placed = {"x": rect["x"], "y": rect["y"], "width": height, "height": width}
        # return best_rect, best_placed
        
        #################################### Worst Long Side Fit #################################
        # best_long_side = float('-inf')  # Khởi tạo nhỏ nhất để tìm max chiều dài dư thừa
        # best_short_side = float('-inf')
        # for rect in self.free_rects: 
        #     if width <= rect["width"] and height <= rect["height"]:
        #         long_side = max(rect["width"] - width, rect["height"] - height)  # Tính chiều dài dư thừa theo chiều ngang
        #         short_side = min(rect["width"] - width, rect["height"] - height)
        #         if (long_side > best_long_side) or (long_side == best_long_side and short_side > best_short_side):
        #             best_long_side = long_side
        #             best_short_side = short_side
        #             best_rect = rect
        #             best_placed = {"x": rect["x"], "y": rect["y"], "width": width, "height": height}
            
        #     if height <= rect["width"] and width <= rect["height"]:
        #         long_side = max(rect["width"] - height, rect["height"] - width)  # Tính chiều dài dư thừa theo chiều dọc
        #         short_side = min(rect["width"] - height, rect["height"] - width)
        #         if (long_side > best_long_side) or (long_side == best_long_side and short_side > best_short_side):
        #             best_long_side = long_side
        #             best_short_side = short_side
        #             best_rect = rect
        #             best_placed = {"x": rect["x"], "y": rect["y"], "width": height, "height": width}
        # return best_rect, best_placed
        
    
    def place_rect(self, width, height):
        best_rect, best_placed = self.find_place(width, height)
        if not best_rect or not best_placed:
            return False  # Không tìm được vị trí đặt

        # Tách không gian trống
        new_rect = self.split_rect(best_rect, best_placed)
        self.free_rects.remove(best_rect)  # Loại bỏ free_rect đã sử dụng

        # Nếu có không gian mới, thêm vào
        if new_rect:
            self.free_rects.extend(new_rect)

        # Loại bỏ không gian bị bao trùm (nếu có)
        self.remove_contained_rect()

        return best_placed
        
    
class GeneticAlgorithm(Policy2311997_2310030_2311745_2313598):
    def __init__(self):
        self.stockList = []
        self.productList = []
        self.population_size = 50
        self.numOfGenerations = 30
        self.mutation_rate = 0.05
        self.current_solution = None
        self.rectStocks = []
        self.current_prod_idx = None
        
    #Firstly initialize the population with different solutions
    def initAllSolution(self):
        solutions = []
        
        #First solution: Prioritize large products first
        large_product_first = {
            "productPriority": np.argsort(
                [-p["size"][0] * p["size"][1] for p in self.productList]
            ),
            "stockPriority": np.random.permutation(len(self.stockList)),
        }
        solutions.append(large_product_first)
        
        #Second solution: Prioritize small products first
        small_product_first = {
            "productPriority": np.argsort(
                [p["size"][0] * p["size"][1] for p in self.productList]
            ),
            "stockPriority": np.random.permutation(len(self.stockList)),
        }
        solutions.append(small_product_first)
        
        #Third solution: Prioritize large stocks first
        large_stocks_first = {
            "productPriority": np.random.permutation(len(self.productList)),
            "stockPriority": np.argsort(
                [-np.prod(self._get_stock_size_(s)) for s in self.stockList]
            ),
        }
        solutions.append(large_stocks_first)

        #Fourth solution: Prioritize small stocks first
        small_stocks_first = {
            "productPriority": np.random.permutation(len(self.productList)),
            "stockPriority": np.argsort(
                [np.prod(self._get_stock_size_(s)) for s in self.stockList]
            ),
        }
        solutions.append(small_stocks_first)
        
        
        #Randomly generate the rest of the solutions
        seen = set()

        while len(solutions) < self.population_size:
            new_solution = {
                "productPriority": np.random.permutation(len(self.productList)).tolist(),
                "stockPriority": np.random.permutation(len(self.stockList)).tolist(),
            }
            
            solution_key = (
                tuple(new_solution["productPriority"]),
                tuple(new_solution["stockPriority"]),
            )
            
            if solution_key not in seen:
                solutions.append(new_solution)
                seen.add(solution_key)
        
        return solutions
        
    def place_product(self, stock, position, prod_size):
        pos_x, pos_y = position
        product_w, product_h = prod_size
        stock[pos_x : pos_x + product_w, pos_y : pos_y + product_h] = 1
        
    def fitness(self, solution):
        waste = 0
        for stock in solution:
            waste += np.sum(stock == -1)
        return -waste
    
    def cal_waste(self, stock, prod_size):
        prod_w, prod_h = prod_size
        remain_area = np.sum(stock == -1)
        return remain_area - prod_w * prod_h

    
    def cal_fitness(self, solution):
        temp_products = [product.copy() for product in self.productList]
        temp_stocks = [stock.copy() for stock in self.stockList]
        
        used_stock = []
        
        rectStocks = [MaximalRectangle(self._get_stock_size_(stock)[0], self._get_stock_size_(stock)[1]) for stock in temp_stocks]
        
        stock_id_to_index = {idx: idx for idx in range(len(temp_stocks))}
            
        
        for prod_idx in solution["productPriority"]:
            product = temp_products[prod_idx]
            prod_size = product["size"]
            if product["quantity"] == 0:
                continue
            
            # while product["quantity"] > 0:
            for stock_idx in solution["stockPriority"]:
                stock_id = stock_id_to_index[stock_idx]
                stock = temp_stocks[stock_id]
                rectStock = rectStocks[stock_id]
                
                # Cố gắng đặt sản phẩm theo cả hai hướng
                result = rectStock.place_rect(prod_size[0], prod_size[1])
                if result:
                    new_prod_size = (result["width"], result["height"])
                    if self._can_place_(stock, (result["x"], result["y"]), new_prod_size):
                        self.place_product(stock, (result["x"], result["y"]), new_prod_size)
                        rectStocks[stock_id] = rectStock
                        if not any(np.array_equal(stock, s) for s in used_stock):
                            used_stock.append(stock)
                        product["quantity"] -= 1
                        break  # Thoát khỏi vòng lặp stock khi đã đặt thành công
                            
        
        # utilitization = total_used_area / total_stock_area if total_stock_area > 0 else 0
        
        total_waste = 0
        for stock in used_stock:
            total_waste += np.sum(stock == -1)
            
        return -total_waste
    
    def crossOver(self, firstParent, secondParent):
        slicePoint = None
        if len(firstParent["productPriority"]) > 2:
            slicePoint = np.random.randint(1, len(firstParent["productPriority"])-1)
        else:
            slicePoint = 1
            
        productPriority = list(firstParent["productPriority"][:slicePoint]) + [i for i in list(secondParent["productPriority"]) if i not in list(firstParent["productPriority"][:slicePoint])]
        stockPriority = list(firstParent["stockPriority"][:slicePoint]) + [i for i in list(secondParent["stockPriority"]) if i not in list(firstParent["stockPriority"][:slicePoint])]
        return {"productPriority": productPriority, "stockPriority": stockPriority}
        
    def mutate(self, solution):
        random_rate = np.random.rand()
        if random_rate < self.mutation_rate and len(solution["productPriority"]) > 1:
            i, j = np.random.choice(len(solution["productPriority"]), 2, replace=False)
            solution["productPriority"][i], solution["productPriority"][j] = solution["productPriority"][j], solution["productPriority"][i]
            
            m, n = np.random.choice(len(solution["stockPriority"]), 2, replace=False)
            solution["stockPriority"][m], solution["stockPriority"][n] = solution["stockPriority"][n], solution["stockPriority"][m]
                
                    
    
    def evolve(self, solutions):
        fitnessOfAllSolutions = [(self.cal_fitness(solution), solution) for solution in solutions]
        fitnessOfAllSolutions.sort(reverse=True, key= lambda x: x[0])
        
        bestHalf = [solution for _, solution in fitnessOfAllSolutions[:len(fitnessOfAllSolutions)//2]]

        next_gen = []
        while len(next_gen) < self.population_size:
            firstParent, secondParent = np.random.choice(bestHalf, 2, replace=False)
            child = self.crossOver(firstParent, secondParent)
            self.mutate(child)
            next_gen.append(child)
        return next_gen
    
    def optimize(self, observation, info):
        self.productList = observation["products"]
        self.stockList = observation["stocks"]
        allSolutions = self.initAllSolution()
        
        for i in range(self.numOfGenerations):
            allSolutions = self.evolve(allSolutions)
        
        best_solution = max(allSolutions, key = self.cal_fitness)
        return best_solution
    
                    
    def get_action(self, observation, info):
        # Student code here
        if (
            self.current_solution is None  # No solution initialized
            or len(observation["products"]) != len(getattr(self, "_prev_product_list", []))  # Different lengths
            or any(
                tuple(obs_prod["size"]) != tuple(prev_prod["size"])
                for obs_prod, prev_prod in zip(
                    observation["products"], getattr(self, "_prev_product_list", [])
                )
            )
        ):
            # Reinitialize solution
            self.productList = observation["products"]
            self.stockList = observation["stocks"]
            self.current_solution = self.optimize(observation, info)
            self._prev_product_list = observation["products"]  # Save the current product list
            self.rectStocks = [MaximalRectangle(self._get_stock_size_(stock)[0], self._get_stock_size_(stock)[1]) for stock in self.stockList]
            self.current_prod_idx = 0
        
        products = observation["products"]
        stocks = observation["stocks"]
        
        valid_stock_ids = set(range(len(stocks)))
        
        for prod_idx in self.current_solution["productPriority"]:
            product = products[prod_idx]
            if product["quantity"] == 0:
                continue
            prod_size = product["size"]
            for stock_idx in self.current_solution["stockPriority"]:
                if stock_idx not in valid_stock_ids:
                    continue
                rectStock = self.rectStocks[stock_idx]
                result = rectStock.place_rect(prod_size[0], prod_size[1])
                if result:
                    new_prod_size = (result["width"], result["height"])
                    if self._can_place_(stocks[stock_idx], (result["x"], result["y"]), prod_size):
                        return {"stock_idx": stock_idx, "size": new_prod_size, "position": (result["x"], result["y"])}

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
            
class ColumnGeneration(Policy2311997_2310030_2311745_2313598):
    def __init__(self):
        super().__init__()
        self.current_stock_idx = 0
    def _subproblem(self, stock, products):
        """Tạo ra các cột mới (cách cắt mới) từ stock và danh sách sản phẩm"""
        possible_cuts = []
        stock_w, stock_h = self._get_stock_size_(stock)

        for product in products:
            if product["quantity"] <= 0:
                continue

            prod_w, prod_h = product["size"]

            # Thử cắt không xoay
            best_cut_no_rotation = None
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, (x, y), product["size"]):
                        if best_cut_no_rotation is None or (x < best_cut_no_rotation[0] or (x == best_cut_no_rotation[0] and y < best_cut_no_rotation[1])):
                            best_cut_no_rotation = (x, y, prod_w, prod_h, product)

            if best_cut_no_rotation is not None:
                possible_cuts.append(best_cut_no_rotation)

            # Thử cắt xoay
            if prod_w != prod_h:
                best_cut_rotation = None
                for x in range(stock_w - prod_h + 1):
                    for y in range(stock_h - prod_w + 1):
                        if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                            if best_cut_rotation is None or (x < best_cut_rotation[0] or (x == best_cut_rotation[0] and y < best_cut_rotation[1])):
                                best_cut_rotation = (x, y, prod_h, prod_w, product)

                if best_cut_rotation is not None:
                    possible_cuts.append(best_cut_rotation)

        return possible_cuts

    def _master_problem(self, possible_cuts, demand, products):
        """Giải quyết bài toán chính (Master Problem)"""
        num_cuts = len(possible_cuts)

        c = [-1] * num_cuts

        A_eq = np.zeros((len(products), num_cuts))
        b_eq = np.array(demand)

        for i, product in enumerate(products):
            for j, cut in enumerate(possible_cuts):
                if cut[4]["size"][0] == product["size"][0] and cut[4]["size"][1] == product["size"][1]:
                    A_eq[i, j] = 1

        bounds = [(0, None)] * num_cuts

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if result.success:
            return result.x
        else:
            return None

    def _branch_and_price(self, possible_cuts, demand, products):
        """Thực hiện Branch-and-Price để tìm giải pháp nguyên"""
        num_cuts = len(possible_cuts)
        A_eq = np.zeros((len(products), num_cuts))
        b_eq = np.array(demand)

        for i, product in enumerate(products):
            for j, cut in enumerate(possible_cuts):
                if cut[4]["size"][0] == product["size"][0] and cut[4]["size"][1] == product["size"][1]:
                    A_eq[i, j] = 1

        master_solution = self._master_problem(possible_cuts, demand, products)

        if master_solution is None:
            return None

        if all(x.is_integer() for x in master_solution):
            return master_solution

        non_integer_idx = next(i for i, x in enumerate(master_solution) if not x.is_integer())
        lower_bound = int(np.floor(master_solution[non_integer_idx]))
        upper_bound = int(np.ceil(master_solution[non_integer_idx]))

        bounds_low = [(0, None)] * len(possible_cuts)
        bounds_low[non_integer_idx] = (0, lower_bound)
        result_low = linprog(
            c=[-1] * len(possible_cuts),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_low,
            method="highs",
        )

        bounds_high = [(0, None)] * len(possible_cuts)
        bounds_high[non_integer_idx] = (upper_bound, None)
        result_high = linprog(
            c=[-1] * len(possible_cuts),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_high,
            method="highs",
        )

        if result_low.success and (not result_high.success or result_low.fun < result_high.fun):
            return result_low.x
        elif result_high.success:
            return result_high.x
        return None

    def get_action(self, observation, info):
        """Lấy hành động tối ưu dựa trên Column Generation và Branch-and-Price"""
        filled_ratio = info["filled_ratio"]
        if filled_ratio == 0:
            self.current_stock_idx = 0
        products = observation["products"]
        stocks = observation["stocks"]
        demand = [product["quantity"] for product in products]
        while self.current_stock_idx < len(stocks):
            stock_idx = self.current_stock_idx
            stock = stocks[stock_idx]
            possible_cuts = self._subproblem(stock, products)
            if possible_cuts:
                master_solution = self._branch_and_price(possible_cuts, demand, products)

                if master_solution is not None:
                    selected_cuts = [
                        cut for i, cut in enumerate(possible_cuts) if master_solution[i] > 0.5
                    ]
                    if selected_cuts:
                        chosen_cut = max(selected_cuts, key=lambda cut: master_solution[possible_cuts.index(cut)])
                        x, y, prod_w, prod_h, product = chosen_cut
                        return {
                            "stock_idx": stock_idx,
                            "size": np.array([prod_w, prod_h]),
                            "position": np.array([x, y]),
                        }
                chosen_cut = random.choice(possible_cuts)
                x, y, prod_w, prod_h, product = chosen_cut
                return {
                    "stock_idx": stock_idx,
                    "size": np.array([prod_w, prod_h]),
                    "position": np.array([x, y]),
                }
            self.current_stock_idx += 1
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}