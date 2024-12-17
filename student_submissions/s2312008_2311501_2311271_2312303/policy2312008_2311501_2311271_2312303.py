from policy import Policy
from math import ceil
from random import randint, random, choices
import numpy as np
from scipy.optimize import linprog

class Policy2312008_2311501_2311271_2312303(Policy):
    # Khởi tạo các tham số cấu hình cho thuật toán
    def __init__(self, policy_id, population_size=300, mutation_rate=0.1, max_generations=100):
        # Kiểm tra giá trị hợp lệ của policy_id
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # THỰC HIỆN GIẢI THUẬT GENETIC ALGORITHM NẾU ID = 1 #
        if self.policy_id == 1:
            # Khởi tạo các tham số thuật toán di truyền
            self.policy_id = policy_id
            self.population_size = population_size
            self.mutation_rate = mutation_rate
            self.max_generations = max_generations
            
            # Các tham số cho việc tính toán kích thước kho và các sản phẩm
            self.stock_length = 0
            self.stock_width = 0
            self.length_arr = []
            self.width_arr = []
            self.demand_arr = []
            self.N = None  # Số lượng sản phẩm
        
        # THỰC HIỆN GIẢI THUẬT COLUMN GENERATION NẾU ID = 2 #
        elif self.policy_id == 2:
            super().__init__()
            self.patterns = []  # Danh sách các mẫu (pattern)
            self.init = False  # Cờ đánh dấu đã khởi tạo các mẫu hay chưa
            self.num_prods = None  # Số lượng sản phẩm

    # Phương thức lấy hành động dựa trên chiến lược (policy)
    def get_action(self, observation, info=None):
        if self.policy_id == 1:
            return self.run_genetic_algorithm(observation)  # Dùng thuật toán di truyền nếu policy_id = 1
        elif self.policy_id == 2:
            products = observation["products"]
            stocks = observation["stocks"]

            # Nhu cầu (demand) và kích thước của từng sản phẩm
            demand = np.array([prod["quantity"] for prod in products])
            sizes = [prod["size"] for prod in products]
            num_prods = len(products)

            self._initialize_patterns_if_needed(num_prods, sizes, stocks)

            # Lặp để giải bài toán tối ưu tuyến tính và thêm mẫu mới nếu cần
            while True:
                result = self._solve_linear_program(demand)
                if result.status != 0 or not self._has_valid_dual_prices(result):
                    break

                dual_prices = result.ineqlin.marginals
                new_pattern = self.solve_pricing_problem(dual_prices, sizes, stocks)

                # Nếu mẫu mới bị trùng lặp hoặc không hợp lệ, dừng lặp
                if self._is_duplicate_pattern(new_pattern):
                    break

                self.patterns.append(new_pattern)

            # Chọn mẫu tốt nhất và chuyển đổi thành hành động
            best_pattern = self.select_best_pattern(self.patterns, demand)
            return self.pattern_to_action(best_pattern, sizes, stocks)


###############################################
# CÁC HÀM HỖ TRỢ GIẢI THUẬT GENETIC ALGORITHM #
###############################################

    # Thuật toán di truyền
    def run_genetic_algorithm(self, observation):
        self.initialize_parameters(observation)  # Khởi tạo các tham số sản phẩm và kho
        patterns = self.generate_efficient_patterns()  # Sinh ra các pattern hiệu quả
        max_repeat_arr = self.calculate_max_pattern_repetition(patterns)  # Tính số lần lặp lại tối đa của mỗi pattern
        population = self.initialize_population(max_repeat_arr)  # Khởi tạo quần thể (population)

        # Tiến hành tiến hóa qua các thế hệ
        for _ in range(self.max_generations):
            fitness_scores = np.array([self.evaluate_fitness(chrom, patterns) for chrom in population])  # Đánh giá fitness của các cá thể
            sorted_indices = np.argsort(-fitness_scores)  # Sắp xếp cá thể theo độ phù hợp giảm dần
            population = [population[i] for i in sorted_indices]

            # Giữ lại các cá thể xuất sắc nhất (elitism)
            next_generation = population[:max(3, self.population_size // 10)]

            # Tạo các cá thể con bằng cách lai ghép và đột biến
            while len(next_generation) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_scores)
                parent2 = self.tournament_selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)  # Lai ghép
                next_generation.extend([self.mutate(child1, max_repeat_arr), self.mutate(child2, max_repeat_arr)])  # Đột biến

            population = next_generation[:self.population_size]  # Cập nhật quần thể

        best_solution = population[0]  # Chọn giải pháp tốt nhất từ quần thể
        return self.format_action(best_solution, patterns, observation)  # Định dạng hành động (vị trí và kích thước)

    # Khởi tạo các tham số từ observation
    def initialize_parameters(self, observation):
        products = observation.get("products", [])  # Lấy danh sách sản phẩm từ observation
        valid_products = [prod for prod in products if prod["quantity"] > 0]  # Lọc các sản phẩm có số lượng lớn hơn 0

        # Tạo các mảng lưu trữ thông tin về kích thước và nhu cầu của sản phẩm
        self.length_arr = [prod["size"][0] for prod in valid_products]
        self.width_arr = [prod["size"][1] for prod in valid_products]
        self.demand_arr = [prod["quantity"] for prod in valid_products]
        self.N = len(self.length_arr)

        if self.N == 0:
            raise ValueError("No products available for processing.")  # Kiểm tra nếu không có sản phẩm hợp lệ

        # Lấy kích thước kho từ observation (ví dụ: sử dụng kho đầu tiên)
        first_stock = observation.get("stocks", [])[0]
        self.stock_length, self.stock_width = self.get_stock_size(first_stock)

    # Sinh ra các pattern sắp xếp hiệu quả cho các sản phẩm
    def generate_efficient_patterns(self):
        patterns = []
        stack = [([0] * self.N, 0, 0)]  # Lưu trữ các pattern đã được tạo ra

        while stack:
            current_pattern, length_used, width_used = stack.pop()  # Lấy pattern hiện tại

            # Lặp qua từng sản phẩm và tạo các pattern mới
            for i in range(self.N):
                max_repeat = min(
                    (self.stock_length - length_used) // self.length_arr[i],  # Kiểm tra không vượt quá chiều dài kho
                    (self.stock_width - width_used) // self.width_arr[i],  # Kiểm tra không vượt quá chiều rộng kho
                    self.demand_arr[i]  # Không vượt quá nhu cầu sản phẩm
                )
                if max_repeat > 0:
                    new_pattern = current_pattern.copy()  # Tạo bản sao pattern hiện tại
                    new_pattern[i] += max_repeat  # Cập nhật số lượng sản phẩm i
                    patterns.append(new_pattern)  # Thêm pattern mới vào danh sách
                    stack.append((new_pattern, length_used + max_repeat * self.length_arr[i], width_used + max_repeat * self.width_arr[i]))  # Tiến hành đệ quy

        # Sắp xếp các pattern theo độ hiệu quả sử dụng không gian
        patterns.sort(key=lambda p: sum(p[i] * self.length_arr[i] * self.width_arr[i] for i in range(self.N)), reverse=True)
        return patterns

    # Tính toán số lần tối đa có thể lặp lại mỗi pattern
    def calculate_max_pattern_repetition(self, patterns):
        return [
            max(ceil(self.demand_arr[i] / pattern[i]) for i in range(self.N) if pattern[i] > 0)
            for pattern in patterns
        ]

    # Khởi tạo population (quần thể các cá thể)
    def initialize_population(self, max_repeat_arr):
        population = []
        for _ in range(self.population_size):
            chromosome = [(i, randint(1, max(1, max_repeat_arr[i]))) for i in range(self.N)]  # Đảm bảo giá trị >= 1
            chromosome_flat = [item for sublist in chromosome for item in sublist]  # Phẳng hóa chromosome

            if len(chromosome_flat) >= 4:  # Chỉ thêm cá thể đủ chiều dài
                population.append(chromosome_flat)

        # Nếu quần thể vẫn rỗng, tạo một cá thể mặc định để đảm bảo không bị lỗi
        if not population:
            default_chromosome = [(i, 1) for i in range(self.N)]  # Tạo cá thể mặc định với số lượng nhỏ nhất
            population.append([item for sublist in default_chromosome for item in sublist])
        
        return population

    # Đánh giá fitness của một cá thể
    def evaluate_fitness(self, chromosome, patterns, wf=0.5, wg=0.5):
        provided = [0] * self.N  # Lượng sản phẩm đã cung cấp
        total_area = self.stock_length * self.stock_width  # Diện tích kho
        used_area = 0  # Diện tích đã sử dụng

        for i in range(0, len(chromosome), 2):
            pattern_index = chromosome[i]
            repetition = chromosome[i + 1]
            pattern = patterns[pattern_index]  # Lấy pattern tương ứng

            for j in range(self.N):
                provided[j] += pattern[j] * repetition  # Cập nhật lượng sản phẩm đã cung cấp

            # Tính toán diện tích sử dụng của pattern
            pattern_area = sum(pattern[k] * self.length_arr[k] * self.width_arr[k] for k in range(self.N))
            used_area += pattern_area * repetition

        unsupplied = sum(
            max(0, self.demand_arr[k] - provided[k]) * self.length_arr[k] * self.width_arr[k]
            for k in range(self.N)
        )

        # Hàm f(S): Tỷ lệ sử dụng diện tích kho
        f_s = used_area / total_area

        # Hàm g(S): Tỷ lệ lãng phí (trim loss)
        g_s = unsupplied / total_area

        # Tính fitness của cá thể với trọng số wf và wg
        return wf * f_s - wg * g_s

    # Lựa chọn bố mẹ bằng phương pháp thi đấu (tournament selection)
    def tournament_selection(self, population, fitness_scores, k=3):
        if len(population) < k:  # Nếu số cá thể trong quần thể nhỏ hơn k
            k = len(population)  # Chỉ chọn số cá thể hiện có
        if k == 0:  # Nếu không có cá thể nào
            # Trả về cá thể mặc định thay vì lỗi
            return population[0] if population else [0] * self.N

        selected_indices = choices(range(len(population)), k=k)  # Chọn k cá thể ngẫu nhiên
        best_index = max(selected_indices, key=lambda idx: fitness_scores[idx])  # Chọn cá thể tốt nhất
        return population[best_index]

    # Lai ghép (crossover) hai cá thể
    def crossover(self, parent1, parent2):
        if len(parent1) < 3 or len(parent2) < 3:
            return parent1[:], parent2[:]  # Trả về bản sao nếu không đủ chiều dài để lai ghép

        point = randint(1, len(parent1) - 2)  # Chọn điểm lai ghép ngẫu nhiên
        return (
            parent1[:point] + parent2[point:],  # Lai ghép giữa parent1 và parent2
            parent2[:point] + parent1[point:],
        )

    # Đột biến (mutation) một cá thể
    def mutate(self, chromosome, max_repeat_arr):
        mutated = chromosome[:]
        for i in range(0, len(mutated), 2):
            if random() < self.mutation_rate:  # Đột biến với tỷ lệ mutation_rate
                mutated[i + 1] = randint(1, max_repeat_arr[mutated[i]])  # Thay đổi số lần lặp lại của sản phẩm
        return mutated

    # Định dạng hành động (vị trí và kích thước sản phẩm)
    def format_action(self, best_solution, patterns, observation):
        stocks = observation.get("stocks", [])

        for i in range(0, len(best_solution), 2):
            pattern_index = best_solution[i]
            repetition = best_solution[i + 1]
            pattern = patterns[pattern_index]

            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self.get_stock_size(stock)

                for x in range(stock_w):
                    for y in range(stock_h):
                        for k, count in enumerate(pattern):
                            if count > 0 and self._can_place_(stock, (x, y), (self.length_arr[k], self.width_arr[k])):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": [self.length_arr[k], self.width_arr[k]],
                                    "position": (x, y),
                                }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    # Lấy kích thước kho từ thông tin của kho
    def get_stock_size(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))  # Tính chiều dài kho
        stock_h = np.sum(np.any(stock != -2, axis=0))  # Tính chiều rộng kho
        return stock_w, stock_h


###############################################
# CÁC HÀM HỖ TRỢ GIẢI THUẬT COLUMN GENERATION #
###############################################

    def _initialize_patterns_if_needed(self, num_prods, sizes, stocks):
        # Kiểm tra nếu cần khởi tạo lại danh sách mẫu
        if not self.init or self.num_prods != num_prods:
            self.init_patterns(num_prods, sizes, stocks)
            self.init = True
            self.num_prods = num_prods

    def _solve_linear_program(self, demand):
        # Giải bài toán tối ưu tuyến tính để tìm các mẫu phù hợp
        c = np.ones(len(self.patterns)) 
        A = np.array(self.patterns).T 
        b = demand 
        return linprog(c, A_ub=-A, b_ub=-b, bounds=(0, None), method='highs')

    def _has_valid_dual_prices(self, result):
        # Kiểm tra nếu kết quả chứa giá trị kép hợp lệ
        return hasattr(result.ineqlin, 'marginals') and result.ineqlin.marginals is not None

    def _is_duplicate_pattern(self, new_pattern):
        # Kiểm tra nếu mẫu mới trùng lặp với các mẫu đã có
        return new_pattern is None or any(np.array_equal(new_pattern, p) for p in self.patterns)

    def init_patterns(self, num_prods, sizes, stocks):
        # Khởi tạo danh sách các mẫu ban đầu dựa trên sản phẩm và tồn kho
        self.patterns = []
        for j, stock in enumerate(stocks):
            stock_size = self._get_stock_size_(stock)
            for i in range(num_prods):
                if stock_size[0] >= sizes[i][0] and stock_size[1] >= sizes[i][1]:
                    pattern = np.zeros(num_prods, dtype=int)
                    pattern[i] = 1
                    self.patterns.append(pattern)
        # Loại bỏ các mẫu trùng lặp
        self.patterns = list({tuple(p): p for p in self.patterns}.values())

    def solve_pricing_problem(self, dual_prices, sizes, stocks):
        # Giải bài toán pricing để tìm mẫu mới
        best_pattern = None
        best_reduced_cost = -1
        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            if stock_w <= 0 or stock_h <= 0:
                continue

            c = -np.array(dual_prices)  
            A_ub, b_ub = self._create_pricing_constraints(sizes, stock_w, stock_h)

            # Giải bài toán tối ưu
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * len(sizes), method='highs')

            if result.success:
                reduced_cost = np.dot(result.x, dual_prices) - 1
                if reduced_cost > best_reduced_cost:
                    best_reduced_cost = reduced_cost
                    best_pattern = result.x

        return best_pattern if best_reduced_cost > 1e-6 else None

    def _create_pricing_constraints(self, sizes, stock_w, stock_h):
        # Tạo các ràng buộc cho bài toán pricing
        n = len(sizes)
        A_ub = []
        b_ub = []

        for i in range(n):
            w, h = sizes[i]
            A_ub.append([1 if j == i else 0 for j in range(n)])
            b_ub.append(stock_w // w * stock_h // h)

        return A_ub, b_ub

    def select_best_pattern(self, patterns, demand):
        # Chọn mẫu có độ phủ tốt nhất và chi phí thấp nhất
        best_pattern = None
        best_coverage = -1
        best_cost = float('inf')

        for pattern in patterns:
            coverage = np.sum(np.minimum(pattern, demand))
            cost = np.sum(pattern)
            if coverage > best_coverage or (coverage == best_coverage and cost < best_cost):
                best_coverage = coverage
                best_cost = cost
                best_pattern = pattern

        return best_pattern

    def pattern_to_action(self, pattern, sizes, stocks):
        # Chuyển đổi mẫu thành hành động cụ thể
        for i, count in enumerate(pattern):
            if count > 0:
                prod_size = sizes[i]
                for stock_idx, stock in enumerate(stocks):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        position = self.bottom_left_place(stock, prod_size)
                        if position is not None:
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": position
                            }
        return {
            "stock_idx": -1,
            "size": [0, 0],
            "position": (0, 0)
        }

    def bottom_left_place(self, stock, prod_size):
        # Tìm vị trí đặt sản phẩm trong tồn kho theo thuật toán bottom-left
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        if stock_w < prod_w or stock_h < prod_h:
            return None
        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None
