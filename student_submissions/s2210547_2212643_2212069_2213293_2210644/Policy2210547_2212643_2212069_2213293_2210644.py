import random
from policy import Policy

class Policy2210547_2212643_2212069_2213293_2210644(Policy):
    def __init__(self,policy_id, population_size=50, generations=100, mutation_rate=0.01, epsilon=0.1):
       self.policy_id = policy_id
       if self.policy_id == 1:
        """
        Khởi tạo chính sách Best Fit Decreasing với tham số epsilon.
        """
        self.epsilon = epsilon  # Sai số gần đúng
        self.actions = []  # Danh sách các hành động sẽ thực hiện

       if self.policy_id == 2:
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate




    def get_action(self, observation, info):
        if self.policy_id == 1:
            """
            Trả về hành động tiếp theo dựa trên danh sách hành động đã sinh ra.
            Nếu danh sách hành động trống, tạo mới bằng thuật toán Best Fit Decreasing.
            """
            if not self.actions:
                self.actions = self.generate_all_actions(observation, info)

            # Trả về hành động đầu tiên trong danh sách
            if self.actions:
                print(self.actions[0])
                return self.actions.pop(0)
            else:
                return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}
        
        if self.policy_id == 2:
            # Khởi tạo quần thể
            population = self._initialize_population(observation)

            for generation in range(self.generations):
                # Đánh giá mức độ phù hợp của mỗi cá thể
                fitness_scores = [self._evaluate(individual, observation) for individual in population]

                # Chọn lọc các cá thể tốt nhất
                selected_individuals = self._selection(population, fitness_scores)

                # Lai ghép để tạo ra thế hệ mới
                next_generation = self._crossover(selected_individuals)

                # Đột biến để duy trì sự đa dạng
                population = self._mutation(next_generation)

            # Chọn cá thể tốt nhất từ quần thể cuối cùng
            best_individual = max(population, key=lambda ind: self._evaluate(ind, observation))
            stock_idx, pos_x, pos_y, prod_size = best_individual

            return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def generate_all_actions(self, observation, info):
        """
        Sinh ra toàn bộ danh sách hành động bằng thuật toán Best Fit Decreasing.
        """
        products = observation["products"]
        stocks = observation["stocks"]

        # Phân loại sản phẩm
        wide_prods, narrow_prods = self.classify_items(products, stocks)

        # Gom nhóm sản phẩm nhỏ
        avg_stock_width = sum(self._get_stock_size_(s)[0] for s in stocks) / len(stocks)
        grouped_narrow_prods = self.group_small_items(narrow_prods, avg_stock_width)

        # Tạo danh sách hành động
        actions = []

        # Đóng gói sản phẩm lớn bằng Best Fit Decreasing
        actions += self.best_fit_decreasing(wide_prods, stocks)

        # Lấp đầy không gian trống với sản phẩm nhỏ
        for stock_idx, stock in enumerate(stocks):
            actions += self.fill_remaining_space(stock, grouped_narrow_prods, stock_idx)

        return actions

    def classify_items(self, products, stocks):
        """
        Phân loại sản phẩm thành wide và narrow dựa trên ngưỡng epsilon.
        """
        avg_stock_width = sum(self._get_stock_size_(s)[0] for s in stocks) / len(stocks)
        threshold = self.epsilon / (2 + self.epsilon) * avg_stock_width

        wide_prods = [p for p in products if p["size"][0] > threshold]
        narrow_prods = [p for p in products if p["size"][0] <= threshold]

        return wide_prods, narrow_prods

    def group_small_items(self, narrow_prods, max_width):
        """
        Gom nhóm sản phẩm nhỏ thành các khối lớn hơn để đóng gói hiệu quả.
        """
        grouped_items = []
        current_width, max_height = 0, 0

        for prod in narrow_prods:
            prod_w, prod_h = prod["size"]
            if current_width + prod_w > max_width:
                grouped_items.append({"size": (current_width, max_height)})
                current_width, max_height = 0, 0
            current_width += prod_w
            max_height = max(max_height, prod_h)

        if current_width > 0:
            grouped_items.append({"size": (current_width, max_height)})

        return grouped_items

    def best_fit_decreasing(self, items, stocks):
        """
        Đóng gói sản phẩm lớn theo chiến lược Best Fit Decreasing.
        """
        actions = []
        items = sorted(items, key=lambda x: x["size"][0] * x["size"][1], reverse=True)  # Sắp xếp theo diện tích giảm dần

        for item in items:
            if item["quantity"] <= 0: continue

            best_fit_idx = -1
            best_fit_gap = float('inf')
            pos_x, pos_y = None, None

            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                item_w, item_h = item["size"]
                
                if stock_w < item_w or stock_h < item_h:
                    continue

                for x in range(stock_w - item_w + 1):
                    for y in range(stock_h - item_h + 1):
                        if self._can_place_(stock, (x, y), item["size"]):
                            # Tính khoảng trống còn lại sau khi đóng gói
                            remaining_gap = (stock_w - x - item_w) * (stock_h - y - item_h)
                            if remaining_gap < best_fit_gap:
                                best_fit_gap = remaining_gap
                                best_fit_idx = stock_idx
                                pos_x, pos_y = x, y
                            break
                    if pos_x is not None and pos_y is not None:
                        break

            if best_fit_idx != -1 and pos_x is not None and pos_y is not None:
                actions.append({
                    "stock_idx": best_fit_idx,
                    "size": item["size"],
                    "position": (pos_x, pos_y),
                })

        return actions

    def fill_remaining_space(self, stock, small_items, stock_idx):
        """
        Lấp đầy khoảng trống bằng các sản phẩm nhỏ.
        """
        actions = []
        stock_w, stock_h = self._get_stock_size_(stock)

        for item in small_items:
            item_w, item_h = item["size"]

            if stock_w < item_w or stock_h < item_h:
                continue

            pos_x, pos_y = None, None
            for x in range(stock_w - item_w + 1):
                for y in range(stock_h - item_h + 1):
                    if self._can_place_(stock, (x, y), item["size"]):
                        actions.append({
                            "stock_idx": stock_idx,
                            "size": item["size"],
                            "position": (x, y),
                        })
                        break
        return actions


        

    def _initialize_population(self, observation):
        population = []
        for _ in range(self.population_size):

            # random choice a stock
            stock_idx = random.randint(0, len(observation["stocks"]) - 1)
            stock = observation["stocks"][stock_idx]

            # Random choice a stock position
            stock_w, stock_h = self._get_stock_size_(stock)
            prod = random.choice(observation["products"])

            prod_size = prod["size"]

            pos_x = random.randint(0, stock_w - prod_size[0])
            pos_y = random.randint(0, stock_h - prod_size[1])

            population.append((stock_idx, pos_x, pos_y, prod_size))

        return population
    




    def _evaluate(self, individual, observation):

        stock_idx, pos_x, pos_y, prod_size = individual
        stock = observation["stocks"][stock_idx]

        if self._can_place_(stock, (pos_x, pos_y), prod_size):
            return 1  # Fitness score, có thể cải thiện bằng cách tính toán khác
        
        return 0

    def _selection(self, population, fitness_scores):
        #https://www.w3schools.com/python/ref_random_choices.asp
        selected = random.choices(population, weights=fitness_scores, k=self.population_size)
        return selected

    def _crossover(self, selected_individuals):
        next_generation = []
        
        for _ in range(self.population_size // 2):

            parent1 = random.choice(selected_individuals)
            parent2 = random.choice(selected_individuals)

            child1, child2 = self._crossover_individuals(parent1, parent2)

            next_generation.extend([child1, child2])
        return next_generation

    def _crossover_individuals(self, parent1, parent2):

        stock_idx1, pos_x1, pos_y1, prod_size1 = parent1
        stock_idx2, pos_x2, pos_y2, prod_size2 = parent2

        child1 = (stock_idx1, pos_x2, pos_y1, prod_size1)
        child2 = (stock_idx2, pos_x1, pos_y2, prod_size2)

        return child1, child2

    def _mutation(self, population):
        for i in range(len(population)):
            if random.random() < self.mutation_rate:
                stock_idx, pos_x, pos_y, prod_size = population[i]
                pos_x += random.randint(-1, 1)
                pos_y += random.randint(-1, 1)
                population[i] = (stock_idx, pos_x, pos_y, prod_size)
        return population


    # Student code here
    # You can add more functions if needed
