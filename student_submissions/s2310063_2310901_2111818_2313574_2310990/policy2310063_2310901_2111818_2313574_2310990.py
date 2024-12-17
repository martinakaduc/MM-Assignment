import numpy as np
from scipy.optimize import linprog
from policy import Policy
class Policy2310063_2310901_2111818_2313574_2310990(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = local_search(policy_id)
        elif policy_id == 2:
            self.policy = column(policy_id)
    def get_action(self, observation, info):
        return self.policy.get_action(observation, info) 

class local_search(Policy):
    def __init__(self):
        self.tabu_tenure = 10  # Kích thước tối đa của Tabu list
        self.max_iterations = 5  # Số vòng lặp tối đa cho thuật toán
        self.check = False
        self.solution = []
        self.current_index = 0 
        self.numsolution = 20
        self.count = 3
        
    
    def get_action(self, observation, info):
        
        if not self.check:
            current_solution = []
            current_objective = 9999999999

            more_sol = self.make_more_solution(observation)
            for sol in more_sol:
                for _ in range (self.count):
                    if (self.objective_function(sol, observation) - self.minstock(sol, observation))  > self.cal_prod(observation):
                        break
                    sol = self.generate_initial_solution(observation)
                # Thực hiện tìm kiếm cục bộ
                for _ in range(self.max_iterations):
                    neighbors = self.generate_neighbors(sol,observation)
                    best_neighbor = None
                    best_objective = float('inf')

                    
                    # Lặp qua các giải pháp lân cận và chọn giải pháp tốt nhất
                    for neighbor in neighbors:
                        neighbor_objective = self.objective_function(neighbor, observation)
                        if neighbor_objective < best_objective:
                            best_objective = neighbor_objective
                            best_neighbor = neighbor

                        # Nếu không có cải thiện, dừng lại
                    if best_objective >= current_objective:
                        break
                    # Cập nhật giải pháp và hàm mục tiêu
                    current_solution = best_neighbor
                    current_objective = best_objective
            #print(current_solution)
            # Lưu giải pháp và thiết lập trạng thái
            
            self.solution = current_solution
            self.check = True
            self.current_index = 0

        # Khi check=True, trả về giải pháp theo chỉ số hiện tại
        if self.check:
            if self.current_index < len(self.solution):
                action = self.convert_to_action(self.solution[self.current_index:self.current_index + 1], observation)
                self.current_index += 1
                return action
            else:
                # Khi đã duyệt hết danh sách, đặt lại trạng thái
                self.check = False
                return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}
            #return self.convert_to_action(current_solution, observation)
        
       

    def cal_prod(self,observation):
        list_prods = observation["products"]
        area = 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                width, height = prod["size"]  # Lấy chiều rộng và chiều cao
                area += prod["quantity"] * width * height  # Tính diện tích và cộng dồn
        return area
    
    def minstock(self,solution,observation):
        # Tìm kích thước nhỏ nhất dựa trên diện tích
        minst = float('inf')  # Giá trị ban đầu lớn vô hạn
        for prod in solution:
            area = prod["size"][0] * prod["size"][1]  # Tính diện tích
            if area < minst:
                minst = area
        return minst
        
    def generate_neighbors(self, current_solution,observation):
        """
        Hàm tạo giải pháp lân cận bằng cách sử dụng kỹ thuật tái cấu trúc (Repackaging).
        Cụ thể, ta sẽ tái cấu trúc các sản phẩm giữa các tấm vật liệu để tối ưu hóa việc sử dụng diện tích.
        """
        neighbors = []
        n = len(current_solution)
        #print("-----------------")
        for _ in range(self.tabu_tenure):
            nei = self.makeneighbor(current_solution,observation)
            #print("nei", nei)
            #print("----------------------")
            neighbors.append(nei)
        return neighbors
    
    def makeneighbor(self,current_solution,observation):
        stock_indices = [item['stock_idx'] for item in current_solution]
        unique_values = list(set(stock_indices))
        np.random.shuffle(unique_values)
        list_prods = observation["products"]
        stocks = observation["stocks"]
        stock_grids = [
            [[False] * self._get_stock_size_(stock)[0] for _ in range(self._get_stock_size_(stock)[1])]
            for stock in stocks
        ]
    
        solution = []
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_w, prod_h = prod["size"]
                for _ in range(prod["quantity"]):  # Handle multiple quantities
                    placed = False
                    for stock_idx in unique_values:
                        stock = stocks[stock_idx]
                        pos_x, pos_y,rotated = self.find_first_fit_position(stock_grids[stock_idx], prod["size"])
                        if pos_x is not None and pos_y is not None:
                            if rotated:
                                prod_size = (prod["size"][1], prod["size"][0])  # Cập nhật kích thước nếu xoay
                                self._mark_grid_(stock_grids[stock_idx], (pos_x, pos_y), prod_size)
                                solution.append({
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (pos_x, pos_y),
                                    "rotated": rotated
                                })
                            else:
                                self._mark_grid_(stock_grids[stock_idx], (pos_x, pos_y), prod["size"])
                                solution.append({
                                    "stock_idx": stock_idx,
                                    "size": prod["size"],
                                    "position": (pos_x, pos_y)
                                })
                            placed = True
                            break
                    if not placed:
                        return current_solution
        return solution
        
    def make_more_solution(self, observation):
        solution = []
        for _ in range (self.numsolution):
            solution.append(self.generate_initial_solution(observation))
        return solution
        

    def generate_initial_solution(self, observation):
        list_prods = observation["products"]
        if isinstance(list_prods, tuple):  # Kiểm tra nếu list_prods là tuple
            list_prods = list(list_prods)  # Chuyển tuple thành list

        np.random.shuffle(list_prods)  # Bây giờ có thể xáo trộn
        stocks = observation["stocks"]
        stock_indices_sorted = list(range(len(stocks)))  # Create a list of indices
        np.random.shuffle(stock_indices_sorted)  # Shuffle the indices randomly
        
        stock_grids = [
            [[False] * self._get_stock_size_(stock)[0] for _ in range(self._get_stock_size_(stock)[1])]
            for stock in stocks
        ]
    
        solution = []
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_w, prod_h = prod["size"]
                for _ in range(prod["quantity"]):  # Handle multiple quantities
                    placed = False
                    for stock_idx in stock_indices_sorted:
                        stock = stocks[stock_idx]
                        pos_x, pos_y, rotated = self.find_first_fit_position(stock_grids[stock_idx], prod["size"])
                        if pos_x is not None and pos_y is not None:
                            if rotated:
                                prod_size = (prod["size"][1], prod["size"][0])  # Cập nhật kích thước nếu xoay
                                self._mark_grid_(stock_grids[stock_idx], (pos_x, pos_y), prod_size)
                                solution.append({
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (pos_x, pos_y),
                                    "rotated": rotated
                                })
                            else:
                                self._mark_grid_(stock_grids[stock_idx], (pos_x, pos_y), prod["size"])
                                solution.append({
                                    "stock_idx": stock_idx,
                                    "size": prod["size"],
                                    "position": (pos_x, pos_y)
                                })
                            placed = True
                            break
                    if not placed:
                        print(f"Unable to place product {prod} due to lack of space.")
        return solution

    def _mark_grid_(self, grid, position, prod_size):
        x, y = position
        prod_w, prod_h = prod_size

        for dy in range(prod_h):
            for dx in range(prod_w):
                grid[y + dy][x + dx] = True
    
    def find_first_fit_position(self, grid, prod_size):
        prod_w, prod_h = prod_size
        stock_h, stock_w = len(grid), len(grid[0])

        # Kiểm tra với kích thước ban đầu
        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if all(
                    not grid[y + dy][x + dx]
                    for dy in range(prod_h)
                    for dx in range(prod_w)
                ):
                    return x, y, False  # False nghĩa là không xoay

        # Kiểm tra với kích thước sau khi xoay
        prod_w, prod_h = prod_h, prod_w  # Hoán đổi width và height
        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if all(
                    not grid[y + dy][x + dx]
                    for dy in range(prod_h)
                    for dx in range(prod_w)
                ):
                    return x, y, True  # True nghĩa là đã xoay 90 độ

        # Không tìm được vị trí phù hợp
        return None, None, None
    
    def _get_stock_area_(self, stock):
        stock_w, stock_h = self._get_stock_size_(stock)
        return stock_w * stock_h
    
    def objective_function(self, solution,observation):
        if not solution:
            return 0
        total_stock_area = 0
        used_stocks = set()  # Tập hợp để lưu các stock_idx đã được xử lý

        for prod in solution:
            stock_idx = prod["stock_idx"]
            if 0 <= stock_idx < len(observation["stocks"]) and stock_idx not in used_stocks:
                used_stocks.add(stock_idx)  # Thêm stock_idx vào tập hợp
                stock = observation["stocks"][stock_idx]
                #stock_w, stock_h = self._get_stock_size_(stock)
                #print(stock_w, "   ",stock_h, "    ",self._get_stock_area_(stock) )
                total_stock_area += self._get_stock_area_(stock)
        return total_stock_area
    
    def convert_to_action(self, solution, observation):
        #Chuyển đổi giải pháp tốt nhất thành hành động cụ thể
        for item in solution:
            if self._can_place_(observation["stocks"][item["stock_idx"]], item["position"], item["size"]):
                return {"stock_idx": item["stock_idx"], "size": item["size"], "position": item["position"]}
        
        # Nếu không có vị trí phù hợp
        return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}
    
class column(Policy):
    def __init__(self):
        self.patterns = []  # Danh sách các mẫu cắt ban đầu
        self.check = False
        self.solution = []
        self.current_index = 0
        self.use_pattern = []  
    
    def calculate_remaining_space(self, stock_grid, prod_w, prod_h):
    
        total_space = len(stock_grid) * len(stock_grid[0])  # Tổng số ô trong lưới
        used_space = sum(row.count(True) for row in stock_grid)  # Số ô đã được sử dụng
        product_space = prod_w * prod_h  # Diện tích sản phẩm
        remaining_space = total_space - used_space - product_space  # Trừ diện tích sản phẩm khỏi không gian trống

        # Nếu không đủ không gian sau khi đặt sản phẩm, trả về giá trị âm để phản ánh điều đó
        return remaining_space if remaining_space >= 0 else float('-inf')

    
    
    def solve_master_problem(self, stocks, products, patterns):
        patterns = np.array(patterns)
    
        # Tạo ma trận A_eq từ patterns
        A_eq = patterns[:, 1:].T  # Transpose để cột là các pattern
        b_eq = np.array([product["quantity"] for product in products])  # Nhu cầu từng sản phẩm

        # Chi phí sử dụng từng stock (diện tích)
        stock_sizes = [self._get_stock_size_(stock) for stock in stocks]
        stock_areas = [length * width for length, width in stock_sizes]
        c = np.array([stock_areas[pattern[0]] for pattern in patterns])

        # Ràng buộc không vượt quá số lượng stock
        num_stocks = len(stocks)
        num_patterns = len(patterns)
        A_stock = np.zeros((num_stocks, num_patterns))

        for j in range(num_patterns):
            stock_index = patterns[j, 0]
            A_stock[stock_index, j] = 1

        b_ub = np.ones(num_stocks)

        # Giải bài toán tuyến tính
        res = linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_stock, b_ub=b_ub, bounds=[(0, None)] * num_patterns, method='highs')

        if res.success:
        # Lấy giá trị đối ngẫu (dual)
            dual_values = res.marginals['lower']  # Hoặc kiểm tra 'upper', tùy thuộc vào bài toán
            return res.x, dual_values
        else:
            raise ValueError(f"Không tìm được lời giải cho Master Problem. Thông báo lỗi: {res.message}")





    def solve_subproblem(self, stocks, products, dual_values):
        new_patterns = []
        stock_sizes = [self._get_stock_size_(stock) for stock in stocks]

        for stock_index, (stock_w, stock_h) in enumerate(stock_sizes):
            num_products = len(products)
            c = [-dual_values[i] for i in range(num_products)]  # Reduced cost

            A_ub = []
            b_ub = []

            for i, product in enumerate(products):
                prod_w, prod_h = product["size"]
                A_ub.append([prod_w if j == i else 0 for j in range(num_products)])
                b_ub.append(stock_w)

                A_ub.append([prod_h if j == i else 0 for j in range(num_products)])
                b_ub.append(stock_h)

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * num_products, method='highs')

            if res.success:
            # Chỉnh lại công thức tính reduced_cost
                reduced_cost = np.dot(res.x, dual_values) - (stock_w * stock_h)
                if reduced_cost < 0:
                    pattern = [stock_index] + list(map(lambda x: int(round(x)), res.x))
                    new_patterns.append(pattern)
            else:
                print(f"Subproblem for stock {stock_index} failed to find a solution: {res.message}")

        return new_patterns

    
    def init_solution(self, list_stock, list_prods, stock_grids):
        patterns = []
        solution = []

        for k, prod in enumerate(list_prods):
            prod_w, prod_h = prod["size"]

            for _ in range(prod["quantity"]):
                min_remaining_space = float('inf')
                selected_stock = -1
                pos_x, pos_y = None, None

                # Duyệt qua từng stock để tìm vị trí tốt nhất
                for i, stock in enumerate(list_stock):
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # Kiểm tra chỉ một hướng của sản phẩm (không xoay)
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_stock(stock_grids[i], (x, y), (prod_w, prod_h)):
                                    # Tính toán không gian còn lại nếu đặt sản phẩm ở vị trí này
                                    remaining_space = self.calculate_remaining_space(
                                        stock_grids[i], prod_w, prod_h
                                    )
                                    if remaining_space < min_remaining_space:
                                        min_remaining_space = remaining_space
                                        selected_stock = i
                                        pos_x, pos_y = x, y
                
                # Nếu tìm thấy vị trí phù hợp, cập nhật lưới và giải pháp
                if selected_stock != -1 and pos_x is not None and pos_y is not None:
                    # Đánh dấu các ô đã chiếm trên lưới
                    for xx in range(pos_x, pos_x + prod_w):
                        for yy in range(pos_y, pos_y + prod_h):
                            stock_grids[selected_stock][yy][xx] = True

                    # Cập nhật giải pháp
                    solution.append({
                        "stock_idx": int(selected_stock),
                        "size": (int(prod_w), int(prod_h)),
                        "position": (int(pos_x), int(pos_y))
                    })

                    # Cập nhật patterns
                    if len(patterns) == 0 or selected_stock not in [pattern[0] for pattern in patterns]:
                        new_column = [selected_stock]  # Dòng đầu tiên là chỉ số stock
                        new_column.extend([0] * len(list_prods))  # Khởi tạo các giá trị 0 cho các sản phẩm khác
                        new_column[k + 1] = 1  # Cập nhật số lượng sản phẩm loại k+1 vào cột mới
                        patterns.append(new_column)
                    else:
                        column_index = next(i for i, pattern in enumerate(patterns) if pattern[0] == selected_stock)
                        patterns[column_index][k + 1] += 1

        return patterns, solution
    def objective_function(self, solution,observation):
        if not solution:
            return 0
        total_stock_area = 0
        used_stocks = set()  # Tập hợp để lưu các stock_idx đã được xử lý

        for prod in solution:
            stock_idx = prod["stock_idx"]
            if 0 <= stock_idx < len(observation["stocks"]) and stock_idx not in used_stocks:
                used_stocks.add(stock_idx)  # Thêm stock_idx vào tập hợp
                stock = observation["stocks"][stock_idx]
                #stock_w, stock_h = self._get_stock_size_(stock)
                #print(stock_w, "   ",stock_h, "    ",self._get_stock_area_(stock) )
                total_stock_area += self._get_stock_area_(stock)
        return total_stock_area
    def cal_prod(self,observation):
        list_prods = observation["products"]
        area = 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                width, height = prod["size"]  # Lấy chiều rộng và chiều cao
                area += prod["quantity"] * width * height  # Tính diện tích và cộng dồn
        return area
    def get_action(self, observation, info):
        
            if not self.check:
                list_stock = observation["stocks"]
                list_prods = sorted(observation["products"], key=lambda x: x['size'][0] * x['size'][1], reverse=True)
           
                stock_grids = [
                    [[False] * self._get_stock_size_(stock)[0] for _ in range(self._get_stock_size_(stock)[1])]
                    for stock in list_stock
                    ]
                self.patterns, self.solution = self.init_solution(list_stock, list_prods, stock_grids)   
                while False:
                # Giải bài toán master để tìm số lần sử dụng mỗi pattern tối ưu
                    usage, dual_values = self.solve_master_problem(list_stock, list_prods, self.patterns)
            
                # Giải bài toán subproblem để tạo ra các patterns mới   
                    new_patterns = self.solve_subproblem(list_stock, list_prods, dual_values)
            
                # Nếu không có cột mới, kết thúc thuật toán
                    if not new_patterns:
                        break
            
                # Thêm các patterns mới vào danh sách patterns
                    self.patterns.extend(new_patterns)
                    for i, use in enumerate(usage):
                        if use > 0:  # Nếu pattern được sử dụng
                            self.use_pattern.append(self.patterns[i])
           
                self.check = True
        # Khi check=True, trả về giải pháp theo chỉ số hiện tại
           
            if self.check:
                
                if self.current_index < len(self.solution):
                    action = self.convert_to_action(self.solution[self.current_index:self.current_index + 1], observation)
                    self.current_index += 1
                    return action
                else:
                # Khi đã duyệt hết danh sách, đặt lại trạng thái
                    self.check = False
                    return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}

            #return self.convert_to_action(current_solution, observation

    def convert_to_action(self, solution, observation):
        # Chuyển đổi giải pháp tốt nhất thành hành động cụ thể
        for item in solution:
            if self._can_place_(observation["stocks"][item["stock_idx"]], item["position"], item["size"]):
                return {"stock_idx": item["stock_idx"], "size": item["size"], "position": item["position"]}
        
        # Nếu không có vị trí phù hợp
        return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}

    def _can_place_stock(self, grid, pos, size):
        x, y = pos
        w, h = size

    # Kiểm tra giới hạn của grid
        if y + h > len(grid) or x + w > len(grid[0]):
            return False

    # Kiểm tra các ô trong grid
        for row in grid[y:y + h]:
            if any(row[x:x + w]):  # Nếu bất kỳ ô nào đã bị chiếm
                return False
        return True
    def _get_stock_area_(self, stock):
        stock_w, stock_h = self._get_stock_size_(stock)
        return stock_w * stock_h
