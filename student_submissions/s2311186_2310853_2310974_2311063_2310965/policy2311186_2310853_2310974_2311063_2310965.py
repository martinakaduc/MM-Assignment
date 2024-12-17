from policy import Policy
import random as rd
import numpy as np
############################################
# Các tham số cần thiết cho Genetic Algorithm
############################################
POP_SIZE = 100 # Số lượng cá thể trong quần thể
MAX_GEN = 200 # Số lượng thế hệ tối đa
ALPHA = 1 # Hệ số phạt khi mảnh hàng lấn quá
BETA = 1 # Hệ số phạt của tỉ lệ diện tích sử dụng
MUTANT_PROB = 0.2 # Xác suất đột biến
M = 0.01 # Xác suất đột biến của mỗi gen (loại đột biến số 1)
CR = 0.8 # Xác suất lai ghép
ELITE = 10 # Số lượng cá thể tốt nhất được giữ lại
TOURM_SIZE = 50 # Kích thước giải đấu
LOCAL = True # Sử dụng lai ghép cục bộ hay không
LOCAL_PROB = 0.6 # Xác suất lai ghép cục bộ
EXPLORE_GEN = 25 # Số thế hệ để bắt đầu lai ghép cục bộ
DIS_THRESHOLD = 20 # Ngưỡng khoảng cách để chọn cá thể lai cục bộ
DIFF_STOCK_DISTANCE = 2 # Khoảng cách giữa 2 mảnh hàng khác nhau
DIFF_ROTATE_DISTANCE = 1 # Khoảng cách giữa 2 mảnh hàng cùng loại nhưng khác hướng
RD_SEED = 1 # Seed cho random
############################################
############################################ 

class Policy2311186_2310853_2310974_2311063_2310965(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.mode = "greedy"
            self.list_prods = []  # Danh sách sản phẩm
            self.total_products = 0  # Tổng số sản phẩm
            self.stocks = []  # Danh sách kho, chỉ quản lý trong class
            self.placements = []  # Lưu tất cả các hành động sắp xếp
            self.current_action_idx = 0  # Chỉ số hành động hiện tại
            self.initialized = False  # Đánh dấu xem đã chuẩn bị dữ liệu chưa
            self.stockused = []
            self.prod_area = 0
        elif policy_id == 2:
            self.mode = "genetic"
            self.solution = []
            self.population = []
            rd.seed(RD_SEED)

    def get_action(self, observation, info):
        
        # Code cho Greedy First-Fit
        if self.mode == "greedy":
            if not self.initialized:
                self.prepare_placements(observation)
                self.initialized = True
                self.current_action_idx = 1
                self.total_products = sum(prod["quantity"] for prod in observation["products"])
                return self.placements[0]

            if self.current_action_idx < len(self.placements):
                action = self.placements[self.current_action_idx]
                self.current_action_idx += 1
                total_remaining_products = sum(prod["quantity"] for prod in observation["products"])
                print(f"-------------------" )
                print(f"Số sản phẩm còn lại: {total_remaining_products-1}")

                if total_remaining_products == 1:
                    last_stock_idx = action["stock_idx"]
                    self.waste(last_stock_idx)
                    pause = input("Press Enter to continue...")
                    self.list_prods = []  # Danh sách sản phẩm
                    self.stocks = []  # Danh sách kho, chỉ quản lý trong class
                    self.placements = []  # Lưu tất cả các hành động sắp xếp
                    self.current_action_idx = 0  # Chỉ số hành động hiện tại
                    self.initialized = False  # Đánh dấu xem đã chuẩn bị dữ liệu chưa
                return action
            
            # Nếu hết hành động, trả về giá trị mặc định
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        # Code cho Genetic Algorithm
        elif self.mode == "genetic":
            if not self.solution:
                # Lấy thông tin về các mảnh hàng và kho
                self.stocks = observation["stocks"]
                self.stocks_size = [self._get_stock_size_(s) for s in self.stocks]
                self.products = observation["products"]
                self.products_size = [p["size"] for p in self.products for _ in range(p["quantity"])]
                # Tùy chọn kích thước quần thể và số thế hệ tối đa
                self.max_gen = MAX_GEN
                self.pop_size = POP_SIZE
                print(f"Numb of products: {self.products_size.__len__()} and numb of stocks: {self.stocks.__len__()}")
                pause = input("Press Enter to continue...")
                # Tính diện tích của các mảnh hàng
                self.area = sum(w * h for w, h in self.stocks_size)
                # Tìm lời giải
                self.solution = self.solve() 
                print(self.solution)
            # Trả về hành động dựa trên lời giải
            stock_idx, prod_size, position = self.solution.pop(0)
            prod_size = self.products_size.pop(0)
            return {"stock_idx": stock_idx, "size": prod_size, "position": position}
        
    #########################################################
    ######### Các hàm cần thiết cho Greedy First-Fit ########
    #########################################################
    def _mark_stock_area(self, stock, position, prod_size):
        """
        Đánh dấu vùng trong kho đã được sử dụng bằng 0.
        """
        x, y = position
        prod_w, prod_h = prod_size
        stock[x : x + prod_w, y : y + prod_h] = 0  # Đánh dấu vùng đã đặt là 0   

    def waste(self,stock_idx):
        waste = 0
        area = 0
        prodarea = 0
        for i in range(stock_idx+1):
            stock = self.stocks[i]
            stock_w, stock_h = self._get_stock_size_(stock)
            area += stock_w * stock_h
            for x in range(stock_w):
                for y in range(stock_h):
                    if stock[x, y] == -1:
                        waste += 1
        print ("tổng diện tích sản phẩm:",self.prod_area)
        print ("tỉ lệ lãng phí sau khi đặt hết :",waste, "/" ,area)

    def _find_empty_regions(self, stock, prod_size):
        """
        Tìm tất cả các vùng trống trong kho có thể đặt sản phẩm.
        """
        stock_width = np.sum(np.any(stock != -2, axis=1))
        stock_height = np.sum(np.any(stock != -2, axis=0))
        empty_regions = []
        prod_w, prod_h = prod_size

        for x in range(stock_width - prod_w + 1):
            for y in range(stock_height - prod_h + 1):
                if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                    empty_regions.append((x, y))
        return empty_regions

    def _find_edge_regions(self, stock, prod_size):
        """
        Tìm tất cả các góc trống trong kho có thể đặt sản phẩm.
        Một góc được định nghĩa:
        - Kề ít nhất hai số 0 trong ma trận (lân cận trực tiếp).
        - Hoặc nằm ở rìa (hàng đầu tiên hoặc cuối).
        """
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        edge_regions = []

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if stock[x, y] != -1:  # Bỏ qua vị trí không trống
                    continue
                neighbors = 0

                if x > 0 and stock[x - 1, y] == 0:  
                    neighbors += 1
                if y > 0 and stock[x, y - 1] == 0: 
                    neighbors += 1
                if x < stock_w - 1 and stock[x + 1, y] == 0: 
                    neighbors += 1
                if y < stock_h - 1 and stock[x, y + 1] == 0:  
                    neighbors += 1

                if neighbors >= 1 or x == 0 and y == 0 :  
                    if x + prod_w <= stock_w and y + prod_h <= stock_h:  # Kiểm tra vừa sản phẩm
                        if np.all(stock[x : x + prod_w, y : y + prod_h] == -1):  # Vùng đủ trống
                            edge_regions.append((x, y))

        return edge_regions

    def prepare_placements(self, observation):
        """
        Chuẩn bị danh sách các hành động sắp xếp sản phẩm vào kho.
        """
        # Lưu lại danh sách sản phẩm và sao chép kho
        self.list_prods = [prod.copy() for prod in observation["products"]]

        # Sắp xếp sản phẩm theo diện tích giảm dần
        self.list_prods = sorted(
            self.list_prods, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True
        )
        stocks_with_index = [(i, stock.copy()) for i, stock in enumerate(observation["stocks"])]

        # Sắp xếp theo tiêu chí và giữ index gốc
        stocks_with_index = sorted(
            stocks_with_index,
            key=lambda item: np.sum(np.any(item[1] != -2, axis=1) * np.any(item[1] != -2, axis=0)),
            reverse=True
        )

        # Tách lại danh sách self.stocks và lưu lại thông tin index gốc
        self.stocks = [stock for _, stock in stocks_with_index]
        self.stock_indices = [index for index, _ in stocks_with_index]
        for stock_idx, stock in enumerate(self.stocks):
            for prod in self.list_prods:
                while prod["quantity"] > 0:
                    empty_regions = self._find_edge_regions(stock, prod["size"])
                    if not empty_regions:
                        break
                    for x, y in empty_regions:
                        if self._can_place_(stock, (x, y), prod["size"]):
                            print(f"Đặt sản phẩm {prod['size']} vào vị trí ({x}, {y}) trong kho {self.stock_indices[stock_idx]}")
                            self.placements.append(
                                {
                                    "stock_idx": self.stock_indices[stock_idx],  # Dùng index gốc
                                    "size": prod["size"],
                                    "position": (x, y),
                                }
                            )
                            self._mark_stock_area(stock, (x, y), prod["size"])
                            prod["quantity"] -= 1
                            placed = True

        for stock_idx, stock in enumerate(self.stocks):
            for prod in self.list_prods:
                while prod["quantity"] > 0:
                    rotated_size = prod["size"][::-1]
                    empty_regions = self._find_edge_regions(stock, rotated_size)
                    if not empty_regions:
                        break
                    for x, y in empty_regions:
                        if self._can_place_(stock, (x, y), rotated_size):
                            print(f"Đã xoay và đặt sản phẩm {rotated_size} vào vị trí ({x}, {y}) trong kho {self.stock_indices[stock_idx]}")
                            self.placements.append(
                                {
                                    "stock_idx": self.stock_indices[stock_idx],  # Dùng index gốc
                                    "size": rotated_size,
                                    "position": (x, y),
                                }
                            )
                            self._mark_stock_area(stock, (x, y), rotated_size)
                            prod["quantity"] -= 1
                            placed = True
                    if placed:
                        break
    
    #########################################################
    ############ Các hàm cần thiết cho Genetic Algorithm ####
    #########################################################
    def solve(self):
        """
        Hàm giải bài toán bằng thuật toán di truyền.
        """
        self.init_population(self.pop_size)  # Khởi tạo quần thể với 100 cá thể
        self.population_fitness = [self.fitness(particle) for particle in self.population]  # Đánh giá fitness của quần thể
        self.best_particle_index = np.argmax(self.population_fitness)  # Chỉ số của cá thể tốt nhất
        best_particle = self.evolve(self.max_gen)  # Tiến hóa qua các thế hệ
        # Trả về cá thể tốt nhất dưới dạng thông tin
        return self._decode_(best_particle)

    def _decode_(self, particle):
        """
        Giải mã một cá thể từ dạng mã hóa sang dạng thông tin.
        
        Input: 
        - particle: [(stock_idx, rotate), ...]

        Output: 
        - [(stock_idx, prod_size, position), ...]
        """
        temp_stocks = [s.copy() for s in self.stocks]
        encoded_particle = []
        for i, prod in enumerate(particle):
            stock_idx, rotate  = prod
            prod_size = self.products_size[i].copy()
            if rotate:
                prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
            position = self._find_best_position(temp_stocks[stock_idx], stock_idx, prod_size)
            if position is None:
                print(f"Product {i} cannot be placed in stock {stock_idx}")
                pause = input("Press Enter to continue...")
            temp_stocks[stock_idx] = self._put_in_stock(temp_stocks[stock_idx], position, prod_size)
            encoded_particle.append((stock_idx, prod_size, position))
        return encoded_particle

    def init_population(self, pop_size):
        """
        Khởi tạo quần thể ngẫu nhiên.
        Tạo các particle dạng [(stock_idx, False), ...] và đảm bảo hợp lệ.

        Input: 
        - pop_size: Số lượng cá thể trong quần thể.
        """
        print("Initializing population...")
        self.population = []
        
        # Duyệt cho đến khi đạt số lượng yêu cầu của quần thể
        while len(self.population) < pop_size:
            # Tạo particle trống với shape bằng số lượng sản phẩm
            particle = [(None, False)] * len(self.products_size)
            
            # Sao chép danh sách stock và sản phẩm để sử dụng tạm
            temp_stocks = [s.copy() for s in self.stocks]
            temp_products_size = list(enumerate(self.products_size))  # (index, product_size)
            
            # Shuffle danh sách sản phẩm để đảm bảo khởi tạo ngẫu nhiên
            rd.shuffle(temp_products_size)
            
            # Tham lam đặt từng sản phẩm
            while temp_products_size:
                prod_idx, prod_size = temp_products_size[0]
                stock_idx = rd.randint(0, len(temp_stocks) - 1)
                # Cập nhật particle
                particle[prod_idx] = (stock_idx, False)
                # Xóa sản phẩm đã đặt
                temp_products_size.pop(0)

            # Nếu particle hợp lệ, thêm vào quần thể
            self.population.append(particle)
            print(f"Particle {len(self.population)} initialized")

        print("Population initialized")


    def _find_best_position(self, stock, stock_idx, prod_size):
        """
        Hàm tìm vị trí đầu tiên có thể đặt được sản phẩm vào kho.
        Duyệt từ trái sang phải, từ trên xuống dưới.

        Input: 
            - stock: Lưới của stock (ma trận 2D)
            - stock_idx: Chỉ số stock
            - prod_size: Kích thước sản phẩm.
        
        Output:
            - Vị trí đặt sản phẩm (x, y) hoặc None nếu không tìm thấy vị trí nào.
        """
        stock_w, stock_h = self.stocks_size[stock_idx]
        prod_w, prod_h = prod_size
        # Duyệt qua các vị trí có thể đặt được
        for x in range(0, stock_w - prod_w + 1):
            for y in range(0, stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None

    def _put_in_stock(self, stock, position, size):
        """
        Đặt một sản phẩm vào kho.

        Input:
            - stock: Lưới stock
            - position: Vị trí đặt
            - size: Kích thước sản phẩm

        Output: 
            - Stock sau khi đặt sản phẩm vào.
        """
        pos_x, pos_y = position
        prod_w, prod_h = size
        stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] = 0
        return stock

    def sumUsed(self, particle):
        """
        Tính tổng diện tích các mảnh đã sử dụng.

        Input:
        - particle: Một cá thể, là danh sách các mảnh, mỗi mảnh là (stock_idx, rotate).

        Output:
        - Tổng diện tích các stock đã sử dụng.
        """
        sum_in_use = 0
        stock_used = set()
        for i, prod in enumerate(particle):
            stock, _  = prod
            if stock not in stock_used:
                w, h = self.stocks_size[stock]
                sum_in_use += w * h
                stock_used.add(stock)
        return sum_in_use
    
    def fitness(self, particle):
        """
        Hàm đánh giá fitness của một cá thể. Phần thưởng dựa trên diện tích sử dụng và số lượng mảnh hàng lấn quá stock.
        
        Input: 
        - particle: Một cá thể, là danh sách các ma trận, mỗi ma trận là (stock_idx, rotate).
        
        Output: 
        - Giá trị fitness của cá thể.
        """
        # Tính số lượng mảnh hàng lấn quá
        overload_prod = 0
        temp_stocks = [s.copy() for s in self.stocks]
        for i, prod in enumerate(particle):
            stock_idx, rotate  = prod
            prod_size = self.products_size[i].copy()
            if rotate:
                prod_size[0], prod_size[1] = prod_size[1], prod_size[0]
            position = self._find_best_position(temp_stocks[stock_idx], stock_idx, prod_size)
            if position is None:
                overload_prod += 1
                continue
            temp_stocks[stock_idx] = self._put_in_stock(temp_stocks[stock_idx], position, prod_size)
        # Tính tỉ lệ diện tích sử dụng
        used_area = self.sumUsed(particle)
        
        return -ALPHA*overload_prod -BETA*(used_area / self.area)

    def _select_local_(self, best_particle, distance_threshold=1000, k=50):
        """
        Chọn các cá thể nằm trong một khoảng distance nhất định so với cá thể tốt nhất
        
        Input:
        - best_particle: Cá thể tốt nhất hiện tại
        - distance_threshold: Ngưỡng khoảng cách để chọn các cá thể
        - k: Số lượng cá thể cần chọn

        Ouput:
        - Danh sách k cá thể được chọn ngẫu nhiên
        """
        # Lưu trữ các cá thể nằm trong ngưỡng khoảng cách
        nearby_particles = []
    
        for i, particle in enumerate(self.population):
            total_distance = 0
            for (stock1, rot1), (stock2, rot2) in zip(particle, best_particle):
                # Tính khoảng cách giữa các điểm
                if (stock1 != stock2):
                    total_distance += DIFF_STOCK_DISTANCE
                if (rot1 != rot2):
                    total_distance += DIFF_ROTATE_DISTANCE
                
            # Nếu tổng khoảng cách nhỏ hơn ngưỡng, thêm vào danh sách
            if total_distance <= distance_threshold:
                nearby_particles.append(i)
        
        # Nếu không đủ k cá thể, bổ sung ngẫu nhiên
        if len(nearby_particles) < k:
            # Lấy thêm từ quần thể ban đầu
            additional_particles = rd.sample(
                [i for i, _ in enumerate(self.population) if i not in nearby_particles], 
                k - len(nearby_particles)
            )
            nearby_particles.extend(additional_particles)
        
        return nearby_particles

    def tournament_selection(self, population, fitness_values, k=3):
        """
        Chọn cá thể sử dụng phương pháp Tournament Selection.
        
        Input:
        - population: Danh sách các cá thể trong quần thể.
        - fitness_values: Giá trị fitness của từng cá thể.
        - k: Số lượng cá thể trong giải đấu (kích thước tournament).

        Ouput:
        - best_candidate: Cá thể tốt nhất được chọn.
        """
        # Chọn ngẫu nhiên k cá thể
        candidates = rd.sample(range(len(population)), k)
        
        # Tìm cá thể tốt nhất trong nhóm
        best_candidate = max(candidates, key=lambda idx: fitness_values[idx]) # Chỉ số của cá thể tốt nhất
        
        return population[best_candidate]

    def mutate(self, particle, M = 0.01):
        """
        Đột biến một cá thể (mỗi gen)

        Cách đột biến: 
            80%: Tại mỗi gen, có xác suất M rằng gen đó bị đột biến. Thay đổi stock và rotate cho 1 product
            20%: Thay đổi stock cho tất cả product trong 1 stock

        Input:
        - particle: Cá thể cần đột biến
        - M: Xác suất đột biến của mỗi gen

        Output:
        - particle: Cá thể sau khi đột biến
        """
        
        if rd.random() < 0.8:
            for i in range(len(particle)):
                if rd.random() < M:
                    # Thay đổi stock và rotate cho 1 product
                    stock = rd.randint(0, len(self.stocks) - 1)
                    particle[i] = (stock, rd.choice([True, False]))
        else:
            # Thay đổi stock cho tất cả product trong 1 stock
            stock_src = rd.randint(0, len(self.stocks) - 1)
            stock_des = rd.randint(0, len(self.stocks) - 1)
            particle = [(stock_des, p[1]) if p[0] == stock_src else p for p in particle]
        
        return particle

    def crossover(self, parent1, parent2):
        """
        Lai ghép 2 cá thể.
        Có 3 kiểu lai ghép:
        - Single-point crossover
        - Two-point crossover

        Input:
        - parent1, parent2: Cá thể cha mẹ

        Output:
        - offspring1, offspring2: Cá thể con sau khi lai ghép
        """
        if rd.random() > CR:
            return parent1, parent2
        
        offspring1 = []
        offspring2 = []

        # Chọn kiểu lai
        cross_type = rd.choice(["single", "two"])
        if cross_type == "single":
            # Single-point crossover
            cross_point = rd.randint(0, len(parent1) - 1)
            offspring1 = parent1[:cross_point] + parent2[cross_point:]
            offspring2 = parent2[:cross_point] + parent1[cross_point:]
        elif cross_type == "two":
            # Two-point crossover
            cross_point1 = rd.randint(0, len(parent1) - 1)
            cross_point2 = rd.randint(cross_point1, len(parent1) - 1)
            offspring1 = parent1[:cross_point1] + parent2[cross_point1:cross_point2] + parent1[cross_point2:]
            offspring2 = parent2[:cross_point1] + parent1[cross_point1:cross_point2] + parent2[cross_point2:]

        return offspring1, offspring2

    def evolve(self, max_gen):
        """
        Tiến hóa quần thể qua các thế hệ.

        Input: 
        - max_gen: Số lượng thế hệ tối đa.

        Output:
        - particle: Cá thể tốt nhất sau khi tiến hóa.
        """
        print("Start evolving...")
        best_fit = self.population_fitness[self.best_particle_index]
        print(f"Begin best fitness: {-best_fit}")
        for gen in range(max_gen):
            next_population = []
            next_population_fitness = []
            # Tiến hóa từng cá thể
            while next_population.__len__() < self.population.__len__() - ELITE:
                # Chọn 2 cá thể cha mẹ
                # Ở 25 thế hệ đầu tiên, sử dụng lai ghép toàn cục
                # Sau 25 thế hệ, sử dụng lai ghép cục bộ với tỉ lệ LOCAL_PROB (nếu LOCAL = True)
                if LOCAL == True and gen > EXPLORE_GEN and rd.random() < LOCAL_PROB:
                    nearby_particles_indices = self._select_local_(self.population[self.best_particle_index], distance_threshold=DIS_THRESHOLD, k=2*TOURM_SIZE)
                else:
                    nearby_particles_indices = range(self.population.__len__())
                # Chọn cha mẹ bằng Tournament Selection
                parent1_indices = self.tournament_selection(nearby_particles_indices, [self.population_fitness[p] for p in nearby_particles_indices], k=TOURM_SIZE)
                parent2_indices = self.tournament_selection(nearby_particles_indices, [self.population_fitness[p] for p in nearby_particles_indices], k=TOURM_SIZE)
                parent1 = self.population[parent1_indices]
                parent2 = self.population[parent2_indices]
                # Lai ghép
                if rd.random() < CR:
                    offspring1, offspring2 = self.crossover(parent1, parent2)
                    next_population.append(offspring1)
                    next_population.append(offspring2)
                else:
                    # Nếu không lai ghép, giữ nguyên cha mẹ
                    next_population.append(parent1)
                    next_population.append(parent2)
                # Đột biến
                if rd.random() < MUTANT_PROB:
                    parent1 = self.mutate(parent1, M)
                    next_population.append(parent1)
                if rd.random() < MUTANT_PROB:
                    parent2 = self.mutate(parent2, M)
                    next_population.append(parent2)

            # Đánh giá fitness của quần thể mới
            next_population_fitness = [self.fitness(particle) for particle in next_population]
            # Duy trì đúng số lượng cá thể mới
            next_population_filtered = sorted(zip(next_population, next_population_fitness), key=lambda x: x[1], reverse=True)[:self.population.__len__() - ELITE]
            next_population, next_population_fitness = zip(*next_population_filtered)
            next_population = list(next_population)
            next_population_fitness = list(next_population_fitness)

            # Duy trì những cá thể tốt nhất
            elites = sorted(zip(self.population, self.population_fitness), key=lambda x: x[1], reverse=True)[:ELITE]
            for elite, fit in elites:
                next_population.append(elite)
                next_population_fitness.append(fit)
                    
            # Cập nhật quần thể
            self.population = next_population.copy()
            self.population_fitness = next_population_fitness.copy()
            self.best_particle_index = np.argmax(self.population_fitness)
            best_fit = self.population_fitness[self.best_particle_index]
            # Lưu lại giai pháp tốt nhất
            print(f"Generation {gen + 1}: Best solution = {-best_fit}")
            with open("85Prods_res.txt", "a") as f:
                f.write(f"{gen + 1} {-best_fit}\n")
            # Dừng sớm nếu tìm được giải pháp tốt
            if best_fit > -0.01:
                break
        # Trả về cá thể tốt nhất
        self.best_particle_index = np.argmax(self.population_fitness)
        print(f"End best fitness: {-best_fit}")
        pause = input("Press Enter to continue...")
        return self.population[self.best_particle_index]
