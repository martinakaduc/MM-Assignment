from policy import Policy
import random
import numpy as np

class Policy2353273_2352918_2352323_2353148_2352500(Policy):
    def __init__(self, policy_id=2, population_size=50, generations=10, alpha=0.5, beta=0.5):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        # Choose the algorithm based on the policy_id
        self.policy_id = policy_id
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.alpha = alpha  
        self.beta = beta
        

    def get_action(self, observation, info):
        
        # Execute the policy based on the policy_id
        if self.policy_id == 2:
            return self.greedy_action(observation, info)
        elif self.policy_id == 1:
            return self.genetic_action(observation, info)

    def greedy_action(self, observation, info):
        """
        Greedy Heuristic Algorithm:
        - Chọn stock có thể cắt sản phẩm với lãng phí thấp nhất.
        - Chọn sản phẩm có yêu cầu lớn nhất để tối ưu hóa sử dụng stock.
        """
        list_prods = observation["products"]
        
        # Kiểm tra nếu list_prods là tuple và chuyển thành list nếu cần
        if isinstance(list_prods, tuple):
            list_prods = list(list_prods)

        best_stock_idx = -1
        best_pos_x, best_pos_y = None, None
        best_prod_size = [0, 0]
        min_waste = float('inf')

        # Sắp xếp các sản phẩm theo yêu cầu giảm dần
        list_prods.sort(key=lambda x: x["quantity"], reverse=True)

        # Duyệt qua tất cả các sản phẩm có số lượng lớn hơn 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Duyệt qua tất cả các stock
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    waste = float('inf')

                    # Kiểm tra cả hai cách cắt: (width, height) và (height, width)
                    for orientation in [(prod_w, prod_h), (prod_h, prod_w)]:
                        pw, ph = orientation
                        if stock_w >= pw and stock_h >= ph:
                            pos_x, pos_y = None, None
                            for x in range(stock_w - pw + 1):
                                for y in range(stock_h - ph + 1):
                                    if self._can_place_(stock, (x, y), orientation):
                                        pos_x, pos_y = x, y
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break

                            # Tính toán lãng phí cho stock này
                            if pos_x is not None and pos_y is not None:
                                stock_waste = (stock_w * stock_h) - (pw * ph)
                                # Chọn stock có lãng phí thấp nhất
                                if stock_waste < min_waste:
                                    min_waste = stock_waste
                                    best_stock_idx = i
                                    best_pos_x, best_pos_y = pos_x, pos_y
                                    best_prod_size = orientation
                                    waste = stock_waste

        # Trả về kết quả vị trí cắt và stock được chọn
        return {"stock_idx": best_stock_idx, "size": best_prod_size, "position": (best_pos_x, best_pos_y)}

    def _get_stock_dimensions(self, stock):
        """Tính toán kích thước thực tế (w, h) của kho."""
        width = np.sum(np.any(stock != -2, axis=1))
        height = np.sum(np.any(stock != -2, axis=0))
        return width, height

    def _can_place_product(self, stock, position, prod_size):
        """Kiểm tra xem có thể đặt sản phẩm ở vị trí chỉ định hay không."""
        stock_w, stock_h = self._get_stock_dimensions(stock)
        prod_w, prod_h = prod_size
        x, y = position
        return 0 <= x <= stock_w - prod_w and 0 <= y <= stock_h - prod_h

    def _initialize_population(self, observation):
        """Khởi tạo quần thể ban đầu với các cá thể ngẫu nhiên."""
        return [self._generate_random_solution(observation) for _ in range(self.population_size)]

    def _generate_random_solution(self, observation):
        """Sinh ngẫu nhiên một cá thể (chromosome)."""
        chromosome = []
        for product in observation["products"]:
            if product["quantity"] > 0:
                stock_idx = random.randint(0, len(observation["stocks"]) - 1)
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_dimensions(stock)
                position = (random.randint(0, stock_w - product["size"][0]),
                            random.randint(0, stock_h - product["size"][1]))
                chromosome.append({"stock_idx": stock_idx, "position": position, "prod_size": product["size"]})
        return chromosome

    def _evaluate_population(self, observation):
        """Tính toán điểm fitness của từng cá thể trong quần thể."""
        fitness_scores = []
        for chromosome in self.population:
            used_stocks = set()
            total_waste = 0

            for gene in chromosome:
                stock_idx, position, prod_size = gene["stock_idx"], gene["position"], gene["prod_size"]
                stock = observation["stocks"][stock_idx]
                stock_w, stock_h = self._get_stock_dimensions(stock)

                if self._can_place_product(stock, position, prod_size):
                    used_stocks.add(stock_idx)
                else:
                    waste_area = (stock_w * stock_h) - (prod_size[0] * prod_size[1])
                    total_waste += waste_area

            num_used_stocks = len(used_stocks)
            fitness = self.alpha / max(1, num_used_stocks) + self.beta / (1 + total_waste)
            fitness_scores.append(fitness)
        return fitness_scores

    def _select_parents(self, fitness_scores):
        """Lựa chọn cha mẹ dựa trên điểm fitness."""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(self.population), random.choice(self.population)

        probabilities = [score / total_fitness for score in fitness_scores]
        return random.choices(self.population, probabilities, k=2)

    def _crossover(self, parent1, parent2):
        """Lai ghép hai cha mẹ để tạo ra hai con."""
        if len(parent1) == 1 or len(parent2) == 1:
            return parent1, parent2
        point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        return (parent1[:point] + parent2[point:], parent2[:point] + parent1[point:])

    def _mutate(self, chromosome, observation):
        """Thực hiện đột biến trên một cá thể."""
        if not chromosome:
            return chromosome
        point = random.randint(0, len(chromosome) - 1)
        chromosome[point]["stock_idx"] = random.randint(0, len(observation["stocks"]) - 1)
        return chromosome

    def _create_new_generation(self, fitness_scores, observation):
        """Tạo thế hệ mới từ quần thể hiện tại."""
        # Giữ lại các cá thể tốt nhất (elitism)
        sorted_population = [x for _, x in sorted(zip(fitness_scores, self.population), key=lambda x: x[0], reverse=True)]
        new_population = sorted_population[:int(self.population_size * 0.2)]

        # Tạo các cá thể mới thông qua lai ghép và đột biến
        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parents(fitness_scores)
            child1, child2 = self._crossover(parent1, parent2)
            new_population.append(self._mutate(child1, observation))
            new_population.append(self._mutate(child2, observation))

        self.population = new_population[:self.population_size]

    def run(self, observation):
        """Chạy thuật toán qua nhiều thế hệ và trả về lời giải tốt nhất."""
        self.population = self._initialize_population(observation)

        for _ in range(self.generations):
            fitness_scores = self._evaluate_population(observation)
            self._create_new_generation(fitness_scores, observation)

        best_fitness = max(fitness_scores)
        best_solution = self.population[fitness_scores.index(best_fitness)]
        return best_solution

    def genetic_action(self, observation, info):
        """Chọn hành động tốt nhất từ lời giải."""
        best_solution = self.run(observation)
        best_gene = best_solution[0]
        return {
            "stock_idx": best_gene["stock_idx"],
            "position": best_gene["position"],
            "size": best_gene["prod_size"]
        }   