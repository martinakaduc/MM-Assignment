import numpy as np
from policy import Policy
from scipy.optimize import linprog

class Policy2110513_2212074_2252722_2252061(Policy):
    def __init__(self, policy_id=1):
        super().__init__()
        self.list_prods = []
        self.list_stocks = []
        self.cuts = []
        self.decision_algorithm = policy_id
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            # clear all the variables
            self.list_prods = []
            self.list_stocks = []
            self.cuts = []
            self.firstIte = True
            self.decision_algorithm = policy_id
            # make the trigger for the first iteration
            pass
        elif policy_id == 2:
            # clear all the variables
            self.best_solution = None
            self.best_objective = np.inf  # Minimization problem, so start with infinity
            self.max_depth = 50  # Giới hạn độ sâu đệ quy
            self.decision_algorithm = policy_id
            # make the trigger for the first iteration
            pass

    def get_action(self, observation, info):
        if self.decision_algorithm == 1:
            if self.firstIte or len(self.cuts) == 0:
                temp_prod = []
                for i, prod in enumerate(observation["products"]):
                    prod_size = prod["size"][:]
                    prod_quantity = prod["quantity"]
                    temp_prod.append({
                        "size": prod_size,
                        "quantity": prod_quantity
                    })
                self.list_prods = sorted(
                    temp_prod, 
                    key=lambda prod: (prod["size"][0], prod["size"][1]),  # Sort by width and length
                    reverse=True
                )
                # print(self.list_prods) 

                list_stock = []
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    temp_stk = {
                        "stock_idx": i,
                        "stock_w": stock_w,
                        "stock_h": stock_h,
                        "arr": stock.copy()
                    }
                    list_stock.append(temp_stk)
                self.list_stocks = list_stock
                # print(self.list_stocks)
                self.cuts = self.guillotine_cut(observation)
                # print(self.cuts)
                self.firstIte = False

            prod_size = [0, 0]
            stock_idx = -1
            pos_x, pos_y = 0, 0

            if (self.cuts):
                cut = self.cuts.pop(0)
                stock_idx = cut["stock"]
                prod_size = cut["size"]
                pos_x = cut["x"]
                pos_y = cut["y"]

            # print(self._get_stock_size_(observation["stocks"][stock_idx]))
            # print(stock_idx, prod_size, pos_x, pos_y, self._get_stock_size_(observation["stocks"][stock_idx]))
            # print(observation["products"])
            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
        
        elif self.decision_algorithm == 2:
            list_prods = observation["products"]
            stocks = observation["stocks"]

            stock_areas = [self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for stock in stocks]
            sorted_stocks = sorted(enumerate(stocks), key=lambda x: stock_areas[x[0]], reverse=True)

            for stock_idx, stock in sorted_stocks:
                stock_w, stock_h = self._get_stock_size_(stock)
                
                valid_prods = []
                for prod_idx, prod in enumerate(list_prods):
                    if prod["quantity"] > 0:
                        prod_w, prod_h = prod["size"]
                        if prod_w <= stock_w and prod_h <= stock_h:
                            valid_prods.append((prod_idx, prod))
                
                if not valid_prods:
                    continue

                c = [-prod["quantity"] for _, prod in valid_prods]
                A = []  
                b = []

                area_constraint = [prod["size"][0] * prod["size"][1] for _, prod in valid_prods]
                A.append(area_constraint)
                b.append(stock_w * stock_h)

                # width_constraint = [prod["size"][0] for _, prod in valid_prods]
                # A.append(width_constraint)
                # b.append(stock_w)

                # height_constraint = [prod["size"][1] for _, prod in valid_prods]
                # A.append(height_constraint)
                # b.append(stock_h)

                bounds = [(0, prod["quantity"]) for _, prod in valid_prods]
                
                self.best_solution = None
                self.best_objective = np.inf
                solution = self.branch_and_bound_ilp(c, A_ub=A, b_ub=b, bounds=bounds)
                
                if solution is not None:
                    selected_prods = []
                    for i, (prod_idx, prod) in enumerate(valid_prods):
                        if solution[i] > 0:
                            selected_prods.append((prod_idx, prod, solution[i]))
                    
                    selected_prods.sort(key=lambda x: x[1]["size"][0] * x[1]["size"][1], reverse=True)
                    
                    for prod_idx, prod, quantity in selected_prods:
                        prod_size = prod["size"]
                        for x in range(stock_w - prod_size[0] + 1):
                            for y in range(stock_h - prod_size[1] + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)
                                    }
            
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
            
    # Student code here
    # You can add more functions if needed
    
    ############################# ALGORITHM 01 #############################

    # def greedy(self):
    #     # Run greedy algorithm
    #     # print (self.list_prods)
    #     print("Processing greedy algorithm")
    #     local_cuts = []
    #     for stock in self.list_stocks:
    #         stock_w, stock_h = self._get_stock_size_(stock["arr"])
    #         pos_x, pos_y = None, None
    #         prod_size = [0, 0]
    #         for y in range(stock_h):    
    #             for x in range(stock_w):
    #                 for prod in self.list_prods:
    #                     if prod["quantity"] > 0:
    #                         prod_size = prod["size"]

    #                         if (stock_w - x < prod_size[0] or stock_h - y < prod_size[1]):
    #                             continue   ## Skip if the product is too big for the stock
    #                         if self._can_place_(stock["arr"], (x, y), prod_size):
    #                             prod["quantity"] -= 1
    #                             pos_x, pos_y = x, y
    #                             stock["arr"][x:x + prod_size[0], y:y + prod_size[1]] = 3
    #                             local_cuts.append({
    #                                 "stock": stock["stock_idx"],
    #                                 "size": prod_size,
    #                                 "x": pos_x,
    #                                 "y": pos_y
    #                             })
    #                             break
    #     # print(local_cuts)
    #     if len(local_cuts) == 0:
    #         local_cuts.append({
    #             "stock": -1,
    #             "size": [0, 0],
    #             "x": 0,
    #             "y": 0
    #         })
    #     return local_cuts
    

    def guillotine_cut(self, observation):
        # self.list_prods.sort(key=lambda p: p["size"][0] * p["size"][1], reverse=True)
        self.list_stocks.sort(key=lambda s: s["stock_w"] * s["stock_h"], reverse=True)

        def guillotine(stock_idx, stock_w, stock_h, products, demands, x_offset, y_offset, observation):
            if all(d == 0 for d in demands):  # Nếu tất cả demand đã được đáp ứng
                return
            if stock_w <= 0 or stock_h <= 0:  # Tấm không còn khả dụng
                return

            for i, (w_p, h_p) in enumerate(products):
                if demands[i] > 0:
                    # Thử cắt dọc
                    if w_p <= stock_w and h_p <= stock_h:
                        # if (self._can_place_(observation["stocks"][stock_idx], (x_offset, y_offset), [w_p, h_p])):
                        # print(self.cuts)
                        demands[i] -= 1
                        # Lưu vị trí cắt dọc với tọa độ (x, y)
                        self.cuts.append({
                            # "direction": "Vertical",
                            "x": x_offset,
                            "y": y_offset,
                            "size": [w_p, h_p],
                            "stock": stock_idx
                        })
                        # Cắt tấm stock còn lại và tiếp tục đệ quy
                        guillotine(stock_idx, stock_w - w_p, stock_h, products, demands, x_offset + w_p, y_offset, observation)

                        # Thử cắt ngang
                        # Cắt tấm stock còn lại và tiếp tục đệ quy
                        guillotine(stock_idx, w_p, stock_h - h_p, products, demands, x_offset, y_offset + h_p, observation)
                        break

            return

        # Lặp qua các tấm stock
        self.cuts = []
        products = [(prod["size"]) for prod in self.list_prods]
        demands = [prod["quantity"] for prod in self.list_prods]
        # print(demands)

        for i,stock in enumerate(self.list_stocks):
            # print("Processing guillotine algorithm: ", i)
            # print(self.cuts) if i>0 else None
            stock_w, stock_h = stock["stock_w"], stock["stock_h"]
            stock_idx = stock["stock_idx"]
            guillotine(stock_idx, stock_w, stock_h, products, demands, 0, 0, observation)
            if all(d <= 0 for d in demands):
                # print("AAAAAAAAAAAAAAAA")
                # print(demands)
                break
        return self.cuts
    
    ############################# ALGORITHM 02 #############################
    def branch_and_bound_ilp(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, bounds=None, depth=0):
        """
        Giải bài toán ILP (Integer Linear Programming) bằng phương pháp Branch-and-Bound với sử dụng incumbent.
        """
        # Giới hạn độ sâu để tránh lỗi đệ quy
        if depth > self.max_depth:
            return None
        
        # Giải bài toán tuyến tính thực sự
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        
        if not res.success or res.fun >= self.best_objective:
            return None  # Không có nghiệm khả thi hoặc nghiệm không tốt hơn incumbent hiện tại
        
        x = res.x
        
        if np.all(np.floor(x) == x):  # Nếu tất cả biến đã nguyên
            objective_value = np.dot(c, x)
            if objective_value < self.best_objective:  # Cập nhật incumbent
                self.best_solution = x
                self.best_objective = objective_value
            return x  # Trả về nghiệm nguyên
        
        fractional_indices = np.where(x != np.floor(x))[0]
        if len(fractional_indices) == 0:  # Không còn biến phân số nào
            return x  # Nghiệm đã nguyên
        
        idx = fractional_indices[0]  # Lấy biến phân số đầu tiên
        x_floor = np.floor(x[idx])
        x_ceil = np.ceil(x[idx])
        
        # Nhánh bên trái (thêm ràng buộc x[idx] ≤ x_floor)
        A_ub_left = np.copy(A_ub) if A_ub is not None else None
        b_ub_left = np.copy(b_ub) if b_ub is not None else None
        if A_ub_left is None:
            A_ub_left = np.zeros((1, len(c)))
            b_ub_left = np.zeros(1)
        else:
            new_constraint = np.zeros(len(c))
            new_constraint[idx] = 1
            A_ub_left = np.vstack([A_ub_left, new_constraint])
            b_ub_left = np.append(b_ub_left, x_floor)
        
        # Nhánh bên phải (thêm ràng buộc x[idx] ≥ x_ceil)
        A_ub_right = np.copy(A_ub) if A_ub is not None else None
        b_ub_right = np.copy(b_ub) if b_ub is not None else None
        if A_ub_right is None:
            A_ub_right = np.zeros((1, len(c)))
            b_ub_right = np.zeros(1)
        else:
            new_constraint = np.zeros(len(c))
            new_constraint[idx] = -1
            A_ub_right = np.vstack([A_ub_right, new_constraint])
            b_ub_right = np.append(b_ub_right, -x_ceil)
        
        # Gọi đệ quy cho cả hai nhánh
        self.branch_and_bound_ilp(c, A_ub=A_ub_left, b_ub=b_ub_left, A_eq=A_eq, b_eq=b_eq, bounds=bounds, depth=depth+1)
        self.branch_and_bound_ilp(c, A_ub=A_ub_right, b_ub=b_ub_right, A_eq=A_eq, b_eq=b_eq, bounds=bounds, depth=depth+1)
        
        return self.best_solution  # Trả về nghiệm tốt nhất tìm được