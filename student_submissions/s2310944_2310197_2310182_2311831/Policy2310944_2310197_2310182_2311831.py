from policy import Policy
import random
import numpy as np

class Policy2310944_2310197_2310182_2311831(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.best_fit(observation, info)
        elif self.policy_id == 2:
            return self.genetic_algorithm(observation)

    def best_fit(self, observation, info):
        list_prods = observation["products"]
        
        # Khởi tạo các giá trị tạm thời cho kích thước sản phẩm, chỉ mục kho và vị trí
        best_fit_stock_idx = -1
        best_fit_pos = None
        best_fit_size = None
        min_wasted_space = float('inf')
        
        # Duyệt qua mỗi sản phẩm để tìm sản phẩm có số lượng > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                
                # Tìm kho tốt nhất để đặt sản phẩm
                for stock_idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    
                    # Thử đặt sản phẩm theo hướng hiện tại của kho
                    if stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                        # Tìm tất cả các vị trí có thể đặt sản phẩm
                        for x in range(stock_w - prod_size[0] + 1):
                            for y in range(stock_h - prod_size[1] + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    # Tính toán không gian lãng phí sau khi đặt sản phẩm
                                    wasted_space = (stock_w * stock_h) - (prod_size[0] * prod_size[1])
                                    if wasted_space < min_wasted_space:
                                        min_wasted_space = wasted_space
                                        best_fit_stock_idx = stock_idx
                                        best_fit_pos = (x, y)
                                        best_fit_size = prod_size

                    # Thử xoay sản phẩm 90 độ (nếu có thể)
                    if stock_w >= prod_size[1] and stock_h >= prod_size[0]:
                        for x in range(stock_w - prod_size[1] + 1):
                            for y in range(stock_h - prod_size[0] + 1):
                                if self._can_place_(stock, (x, y), prod_size[::-1]):
                                    wasted_space = (stock_w * stock_h) - (prod_size[1] * prod_size[0])
                                    if wasted_space < min_wasted_space:
                                        min_wasted_space = wasted_space
                                        best_fit_stock_idx = stock_idx
                                        best_fit_pos = (x, y)
                                        best_fit_size = prod_size[::-1]

        if best_fit_stock_idx == -1:
            return None  # Không tìm thấy vị trí hợp lệ để đặt sản phẩm
        
        return {"stock_idx": best_fit_stock_idx, "size": best_fit_size, "position": best_fit_pos}

    def genetic_algorithm(self, observation):
        """
        Triển khai thuật toán Genetic Algorithm
        """
        population_size = 20
        generations = 50

        self.population_size = population_size
        self.generations = generations

        products = sorted(
            [prod for prod in observation["products"] if prod["quantity"] > 0],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True,
        )

        placed_products = []  # Theo dõi các sản phẩm đã được đặt

        for product in products:
            # Sử dụng kết hợp giữa kích thước và số lượng làm mã nhận diện duy nhất
            product_key = tuple(product["size"])  # Ví dụ sử dụng kích thước làm khóa

            if product_key in placed_products:
                continue  # Bỏ qua sản phẩm đã được đặt

            prod_size = product["size"]
            best_x, best_y, best_stock_idx, best_rotation = None, None, None, None
            best_waste = float("inf")

            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)

                for rotation in [(prod_size[0], prod_size[1]), (prod_size[1], prod_size[0])]:
                    rotated_w, rotated_h = rotation
                    if rotated_w > stock_w or rotated_h > stock_h:
                        continue

                    for x in range(stock_w - rotated_w + 1):
                        for y in range(stock_h - rotated_h + 1):
                            if self._can_place_(stock, (x, y), (rotated_w, rotated_h)):
                                remaining_w = stock_w - (x + rotated_w)
                                remaining_h = stock_h - (y + rotated_h)
                                waste = remaining_w * remaining_h

                                if waste < best_waste:
                                    best_x, best_y = x, y
                                    best_waste = waste
                                    best_stock_idx = stock_idx
                                    best_rotation = rotation

            if best_x is not None and best_y is not None:
                # Sau khi đặt, thêm kích thước của sản phẩm vào placed_products
                placed_products.append(product_key)

                # Tính toán tỷ lệ sử dụng
                stock = observation["stocks"][best_stock_idx]
                stock_area = stock.shape[0] * stock.shape[1]
                used_area = best_rotation[0] * best_rotation[1]
                utilization_ratio = used_area / stock_area

                
                return {
                    "stock_idx": best_stock_idx,
                    "size": best_rotation,
                    "position": (best_x, best_y),
                }

        return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}

    def _run_genetic_algorithm(self, observation):
        # Khởi tạo dân số
        population = [self._random_solution(observation) for _ in range(self.population_size)]
        best_fitness = float("-inf")
        stagnation_counter = 0

        for gen in range(self.generations):
            # Đánh giá fitness
            fitness_scores = [self._evaluate_fitness(solution, observation) for solution in population]

            if not fitness_scores:
                raise ValueError("Fitness scores are empty. Check population and evaluation logic.")

            current_best = max(fitness_scores)
            if current_best > best_fitness:
                best_fitness = current_best
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= 10:
                print(f"Early stopping at generation {gen}")
                break

            # Chọn bố mẹ và tạo ra thế hệ tiếp theo
            parents = self._select_parents(population, fitness_scores)
            children = self._crossover(parents)
            population = [self._mutate(child, observation) for child in children]

            # Giữ kích thước dân số
            while len(population) < self.population_size:
                population.append(self._random_solution(observation))

        best_index = np.argmax(fitness_scores)
        return population[best_index]

    def _random_solution(self, observation):
        products = list(observation["products"])  # Đảm bảo danh sách có thể thay đổi
        random.shuffle(products)
        return products

    def _evaluate_fitness(self, solution, observation):
        total_waste = 0
        for product in solution:
            product_size = product["size"]
            stock_waste = self._calculate_waste(observation, product_size)
            total_waste += stock_waste

        return -total_waste  # Fitness là giá trị âm của tổng lượng waste

    def _calculate_waste(self, observation, product_size):
        """
        Tính toán lượng waste (diện tích kho chưa sử dụng) khi đặt sản phẩm.
        """
        total_waste = 0
        for stock in observation["stocks"]:
            stock_w, stock_h = self._get_stock_size_(stock)
            product_w, product_h = product_size

            # Kiểm tra xem sản phẩm có thể fit vào kho không
            if stock_w >= product_w and stock_h >= product_h:
                # Tính toán waste: chiều rộng và chiều cao còn lại
                remaining_w = stock_w - product_w
                remaining_h = stock_h - product_h
                total_waste += remaining_w * remaining_h

        return total_waste

    def _select_parents(self, population, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        return random.choices(population, probabilities, k=len(population) // 2)

    def _crossover(self, parents):
        children = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1, parent2 = parents[i], parents[i + 1]
                split = random.randint(1, len(parent1) - 1)
                child1 = parent1[:split] + parent2[split:]
                child2 = parent2[:split] + parent1[split:]
                children.extend([child1, child2])
        return children

    def _mutate(self, solution, observation):
        if len(solution) > 1:
            idx1, idx2 = random.sample(range(len(solution)), 2)
            solution[idx1], solution[idx2] = solution[idx2], solution[idx1]
        return solution

    def _apply_best_fit(self, observation, solution):
        for product in solution:
            product_size = product["size"]
            for stock_idx, stock in enumerate(observation["stocks"]):
                stock_w, stock_h = self._get_stock_size_(stock)
                if stock_w >= product_size[0] and stock_h >= product_size[1]:
                    for x in range(stock_w - product_size[0] + 1):
                        for y in range(stock_h - product_size[1] + 1):
                            if self._can_place_(stock, (x, y), product_size):
                                return {"stock_idx": stock_idx, "size": product_size, "position": (x, y)}

        return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}
