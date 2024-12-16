from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2311708_2313870_2311735_2311338_2311390(Policy):
    def __init__(self, policy_id = 1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = ColumnGeneration()
        elif policy_id == 2:
            self.policy = BranchAndBound()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
class ColumnGeneration(Policy):
    def __init__(self):
        self.patterns = []  # List of patterns (columns)
        self.num_products = 0
        
    def get_action(self, observation, info):
        stocks = np.copy(observation["stocks"])
        products = list(np.copy(observation["products"]))
        self.num_products = len(products)

        #Sort products by area
        products.sort(key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        
        self.patterns = self.generate_initial_patterns(products)
        dual_prices = self.solve_master_problem(self.patterns, products)
        new_pattern, reduced_cost = self.solve_subproblem(stocks, products, dual_prices)
        #Use while loop to add new patterns until reduced cost is positive, then select best action
        while True:
            dual_prices = self.solve_master_problem(self.patterns, products)
            new_pattern, reduced_cost = self.solve_subproblem(stocks, products, dual_prices)

            if reduced_cost >= 0:
                break
            #Add new pattern to the list of patterns and repeat the process
            self.patterns.append(new_pattern)

        # Select the best action based on the dual prices
        # Find the best pattern that maximizes the coverage and minimizes the cost
        best_pattern = max(self.patterns, key=lambda x: sum(dual_prices[i] for i in range(self.num_products) if x[i] > 0))
        action = self.select_best_action(best_pattern, stocks, products)
        return action

    def generate_initial_patterns(self, products):
        num_products = len(products)
        patterns = []

        for i, product in enumerate(products):
            if product["quantity"] > 0:
                pattern = [0] * num_products
                pattern[i] = 1
                patterns.append(pattern)

        return np.array(patterns)

    def solve_master_problem(self, patterns, products):
        #Solve the master problem to find the dual prices
        patterns = np.array(patterns)
        num_patterns, num_products = patterns.shape

        c = np.ones(num_patterns)
        A_eq = patterns.T
        b_eq = np.array([product["quantity"] for product in products])

        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        #If the master problem is solved successfully, return the dual prices
        #Otherwise, raise an error
        if result.success:
            #If the result has attribute y, return the dual prices
            #Otherwise, compute the dual prices using perturbation method
            return result.y if hasattr(result, "y") else self.compute_dual_prices(A_eq, b_eq, result.x)
        else:
            raise ValueError(f"Master problem failed to solve: {result.message}")

    def compute_dual_prices(self, A_eq, b_eq, x_solution):
        #Compute dual prices using perturbation method in linprog library
        num_constraints = len(b_eq)
        dual_prices = []

        for i in range(num_constraints):
            perturbed_b_eq = b_eq.copy()
            perturbed_b_eq[i] += 1e-6

            result = linprog(
                np.zeros_like(x_solution), A_eq=A_eq, b_eq=perturbed_b_eq, bounds=(0, None), method="highs"
            )

            if result.success:
                dual_price = (result.fun - sum(x_solution)) / 1e-6
            else:
                dual_price = 0

            dual_prices.append(dual_price)

        return np.array(dual_prices)

    def solve_subproblem(self, stocks, products, dual_prices):
        """
        Solve the subproblem using dynamic programming and knapsack approach
        
        Args:
        - stocks (list): Available stock configurations
        - products (list): List of product specifications
        - dual_prices (list): Dual prices for each product type
        
        Returns:
        - tuple: Best pattern and its reduced cost
        """
        num_products = len(products)
        best_pattern = None
        best_reduced_cost = float("inf")
        
        for stock in stocks:
            # Determine stock dimensions
            stock_width = np.sum(np.any(stock != -2, axis=1))
            stock_height = np.sum(np.any(stock != -2, axis=0))
            
            # Consider both original and rotated dimensions
            stock_dimensions = [(stock_width, stock_height), (stock_height, stock_width)]
            
            for stock_width, stock_height in stock_dimensions:
                # Sort product indices by width and height
                sorted_indices = sorted(
                    range(num_products), 
                    key=lambda k: (products[k]['size'][0], products[k]['size'][1])
                )
                
                # Dynamic programming table for width constraint
                width_dp = [[{} for _ in range(num_products + 1)] for _ in range(stock_width + 1)]
                
                # Fill width dynamic programming table
                for w in range(1, stock_width + 1):
                    for i, prod_idx in enumerate(sorted_indices, 1):
                        product = products[prod_idx]
                        width = product['size'][0]
                        
                        # Copy previous state
                        width_dp[w][i] = width_dp[w][i-1].copy()
                        
                        # Try to add current product
                        if width <= w:
                            # Candidate solution
                            candidate = width_dp[w - width][i-1].copy()
                            candidate[prod_idx] = candidate.get(prod_idx, 0) + 1
                            
                            # Update if better solution found
                            if (not width_dp[w][i] or 
                                len(candidate) > len(width_dp[w][i])):
                                width_dp[w][i] = candidate
                
                # Find best width solutions
                width_solutions = [
                    width_dp[stock_width][j] 
                    for j in range(num_products + 1) 
                    if width_dp[stock_width][j]
                ]
                
                # Check height constraint for each width solution
                for width_solution in width_solutions:
                    # Dynamic programming for height constraint
                    height_dp = [[{} for _ in range(num_products + 1)] for _ in range(stock_height + 1)]
                    
                    # Initialize first row with width solution
                    for h in range(stock_height + 1):
                        height_dp[h][0] = width_solution.copy()
                    
                    # Fill height dynamic programming table
                    for h in range(1, stock_height + 1):
                        for i, prod_idx in enumerate(sorted_indices, 1):
                            product = products[prod_idx]
                            height = product['size'][1]
                            
                            # Copy previous state
                            height_dp[h][i] = height_dp[h][i-1].copy()
                            
                            # Skip if product already in width solution
                            if prod_idx in height_dp[h][i]:
                                continue
                            
                            # Try to add current product
                            if height <= h:
                                # Candidate solution
                                candidate = height_dp[h - height][i-1].copy()
                                candidate[prod_idx] = candidate.get(prod_idx, 0) + 1
                                
                                # Update if better solution found
                                if (not height_dp[h][i] or 
                                    len(candidate) > len(height_dp[h][i])):
                                    height_dp[h][i] = candidate
                    
                    # Find best height solutions
                    best_height_solutions = [
                        height_dp[stock_height][j] 
                        for j in range(num_products + 1) 
                        if height_dp[stock_height][j]
                    ]
                    
                    # Evaluate solutions
                    for solution in best_height_solutions:
                        # Calculate reduced cost
                        reduced_cost = 1 - sum(
                            dual_prices[prod_idx] * count 
                            for prod_idx, count in solution.items()
                        )
                        
                        # Update best solution
                        if reduced_cost < best_reduced_cost:
                            best_pattern = [
                                solution.get(i, 0) 
                                for i in range(num_products)
                            ]
                            best_reduced_cost = reduced_cost
            
        return best_pattern, best_reduced_cost

    def select_best_action(self, pattern, stocks, products):
        for i, count in enumerate(pattern):
            if count > 0:
                product = products[i]
                size = product["size"]
                
                for stock_idx, stock in enumerate(stocks):
                    stock_width, stock_height = self._get_stock_size_(stock)
                    
                    for x in range(stock_width - size[0] + 1):
                        for y in range(stock_height - size[1] + 1):
                            if np.all(stock[x:x+size[0], y:y+size[1]] == -1):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": size,
                                    "position": [x, y]
                                }
        
        return {"stock_idx": -1, "size": [0, 0], "position": [0, 0]}
        
class BranchAndBound(Policy):
    def __init__(self):
        super().__init__()
        self.best_solution = None
        self.product_counter = 1  # Khởi tạo bộ đếm cho ID sản phẩm
    def get_action(self, observation, info):
        list_prods = observation["products"]

        # Lọc ra các sản phẩm cần cắt (có số lượng > 0)
        products = [prod for prod in list_prods if prod["quantity"] > 0]
        # Sắp xếp các product theo thư tự từ lớn đến nhỏ
        products.sort(key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)


        # Đặt lại vị trí tốt nhất trước khi bắt đầu tìm kiếm
        self.best_position = None
        self.min_waste = float('inf')

        # Lặp qua tất cả các tấm nguyên liệu có sẵn
        for i, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            if stock_w==0   or  stock_h==0:
                continue
        

            # Khởi động thuật toán Branch and Bound cho từng tấm nguyên liệu
            self.branch_and_bound(stock, products, 0, stock_idx=i)

        # Trả về thông tin của vị trí và kích thước sản phẩm tối ưu đã tìm được
        if self.best_position:
            return self.best_position
        else:
            return {
                "stock_idx": -1,  # Nếu không tìm thấy vị trí phù hợp
                "size": [0, 0],
                "position": (0, 0)
        }
    
    def branch_and_bound(self, stock, products, current_waste, stock_idx):
        # Điều kiện cắt tỉa: nếu lãng phí hiện tại lớn hơn lãng phí nhỏ nhất đã biết
        if current_waste >= self.min_waste:
            return
        stock=np.array(stock)
        visited_states = set()
        state_key = (tuple(stock.flatten()), tuple(prod["quantity"] for prod in products))
        if state_key in visited_states:
            return
        visited_states.add(state_key)


        # Kiểm tra nếu tất cả sản phẩm đều đã được đặt
        if all(prod["quantity"] == 0 for prod in products):
            # Nếu lãng phí hiện tại nhỏ hơn lãng phí tốt nhất, cập nhật nghiệm tối ưu
            self.best_solution = np.copy(stock)
            self.min_waste = current_waste
            return

        # Lặp qua từng sản phẩm
        for idx, prod in enumerate(products):
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                prod_w, prod_h = prod_size
                stock_w, stock_h = self._get_stock_size_(stock)

                if stock_w < prod_w or stock_h < prod_h:
                    continue

                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                        # Tạo bản sao của stock trước khi thay đổi
                            new_stock = self.place_product(stock, (x, y), prod_size)
                            prod["quantity"] -= 1
                            new_waste = self.calculate_waste(new_stock)

                            if new_waste < self.min_waste:
                                self.best_position = {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y)
                            }
                                self.min_waste = new_waste

                            self.branch_and_bound(new_stock, products, new_waste, stock_idx)

                        # Quay lui: hoàn tác thay đổi
                            prod["quantity"] += 1  
                else:
                    # Kiểm tra trường hợp xoay sản phẩm
                    rotated_size = prod_size[::-1]  # Đổi chiều sản phẩm
                    rot_w, rot_h = rotated_size

                    if stock_w >= rot_w and stock_h >= rot_h:  # Kiểm tra kích thước sau khi xoay
                        for x in range(stock_w - rot_w + 1):
                            for y in range(stock_h - rot_h + 1):
                                if self._can_place_(stock, (x, y), rotated_size):
                    # Tạo bản sao của stock trước khi thay đổi
                                    new_stock = self.place_product(stock, (x, y), rotated_size)
                                    prod["quantity"] -= 1
                                    new_waste = self.calculate_waste(new_stock)

                                    if new_waste < self.min_waste:
                                        self.best_position = {
                                        "stock_idx": stock_idx,
                                        "size": rotated_size,
                                        "position": (x, y)
                                    }
                                        self.min_waste = new_waste

                    # Đệ quy với trạng thái mới
                                    self.branch_and_bound(new_stock, products, new_waste, stock_idx)

                    # Quay lui: hoàn tác thay đổi
                                    prod["quantity"] += 1
                            
    
    def place_product(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        # Sao chép stock ban đầu để không thay đổi trực tiếp
        new_stock = np.copy(stock)
        new_stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] = 1  # Đặt giá trị 1 cho sản phẩm
        
        return new_stock
    def calculate_waste(self, stock):
        empty_spaces = np.sum(stock == 0)
        return empty_spaces
