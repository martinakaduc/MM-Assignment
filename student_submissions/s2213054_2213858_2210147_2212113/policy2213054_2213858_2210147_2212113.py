import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from math import floor, ceil
from copy import deepcopy
from random import randint, shuffle, random, choice, choices
import time 
from policy import Policy

class Policy2213054_2213858_2210147_2212113(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        
        if policy_id == 1:
            self.policy = ColumnGeneration()
        elif policy_id == 2:
            self.policy = GeneticAlgorithm()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

class ColumnGeneration(Policy):
    def __init__(self):
        super().__init__()
        self.patterns = []
        self.init = False
        self.num_prods = None

    def get_action(self, observation, info):
        products = observation["products"]
        stocks = observation["stocks"]
        demand = np.array([prod["quantity"] for prod in products])
        sizes = [prod["size"] for prod in products]
        num_prods = len(products)

        # Khởi tạo lại mẫu nếu cần thiết
        if not self.init or self.num_prods != num_prods:
            self.init_patterns(num_prods, sizes, stocks)
            self.init = True
            self.num_prods = num_prods

        # Vòng lặp tìm kiếm mẫu tối ưu
        while True:
            c = np.ones(len(self.patterns))
            A = np.array(self.patterns).T
            b = demand

            res = linprog(c, A_ub=-A, b_ub=-b, bounds=(0, None), method='highs')

            if res.status != 0:
                break

            dual_prices = self._get_dual_prices(res)

            if dual_prices is None:
                break

            new_pattern = self.solve_pricing_problem(dual_prices, sizes, stocks)

            if new_pattern is None or self._is_pattern_duplicate(new_pattern):
                break

            self.patterns.append(new_pattern)

        # Chọn mẫu tốt nhất và chuyển đổi thành hành động
        best_pattern = self.select_best_pattern(self.patterns, demand)
        return self.pattern_to_action(best_pattern, sizes, stocks)

    def _get_dual_prices(self, res):
        """Trả về dual prices từ kết quả tối ưu"""
        return res.ineqlin.marginals if hasattr(res.ineqlin, 'marginals') else None

    def _is_pattern_duplicate(self, new_pattern):
        """Kiểm tra nếu mẫu mới đã có trong patterns"""
        return any(np.array_equal(new_pattern, p) for p in self.patterns)

    def init_patterns(self, num_prods, sizes, stocks):
        """Khởi tạo các patterns ban đầu dựa trên sản phẩm và cổ phiếu"""
        self.patterns = []
        for stock in stocks:
            stock_size = self._get_stock_size_(stock)
            for i in range(num_prods):
                if self._can_accommodate(stock_size, sizes[i]):
                    pattern = np.zeros(num_prods, dtype=int)
                    pattern[i] = 1
                    self.patterns.append(pattern)
        self.patterns = list({tuple(p): p for p in self.patterns}.values())

    def _can_accommodate(self, stock_size, prod_size):
        """Kiểm tra xem sản phẩm có thể được đóng gói vào cổ phiếu hay không"""
        return stock_size[0] >= prod_size[0] and stock_size[1] >= prod_size[1]

    def solve_pricing_problem(self, dual_prices, sizes, stocks):
        """Giải bài toán định giá cho từng cổ phiếu"""
        best_pattern, best_reduced_cost = None, -1

        for stock in stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            if stock_w <= 0 or stock_h <= 0:
                continue

            dp = np.zeros((stock_h + 1, stock_w + 1))

            for i, (w, h) in enumerate(sizes):
                if w > stock_w or h > stock_h or dual_prices[i] <= 0:
                    continue
                for x in range(stock_w, w - 1, -1):
                    for y in range(stock_h, h - 1, -1):
                        dp[y][x] = max(dp[y][x], dp[y - h][x - w] + dual_prices[i])

            pattern, reduced_cost = self._extract_pattern_from_dp(dp, dual_prices, sizes, stock_w, stock_h)
            if reduced_cost > best_reduced_cost:
                best_reduced_cost = reduced_cost
                best_pattern = pattern

        return best_pattern if best_reduced_cost > 1e-6 else None

    def _extract_pattern_from_dp(self, dp, dual_prices, sizes, stock_w, stock_h):
        """Trích xuất pattern từ bảng DP sau khi giải bài toán định giá"""
        n = len(sizes)
        pattern = np.zeros(n, dtype=int)
        w, h = stock_w, stock_h

        for i in range(n - 1, -1, -1):
            item_w, item_h = sizes[i]
            
            # Kiểm tra để tránh truy cập chỉ số âm trong mảng dp
            if w - item_w >= 0 and h - item_h >= 0:
                if dp[h][w] == dp[h - item_h][w - item_w] + dual_prices[i]:
                    pattern[i] = 1
                    w -= item_w
                    h -= item_h

        reduced_cost = np.dot(pattern, dual_prices) - 1
        return pattern, reduced_cost

    def select_best_pattern(self, patterns, demand):
        """Chọn mẫu tốt nhất dựa trên độ phủ và chi phí"""
        best_pattern, best_coverage, best_cost = None, -1, float('inf')

        for pattern in patterns:
            coverage = np.sum(np.minimum(pattern, demand))
            cost = np.sum(pattern)
            if coverage > best_coverage or (coverage == best_coverage and cost < best_cost):
                best_coverage = coverage
                best_cost = cost
                best_pattern = pattern

        return best_pattern

    def pattern_to_action(self, pattern, sizes, stocks):
        """Chuyển đổi pattern thành hành động cụ thể"""
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
        """Tìm vị trí thấp nhất bên trái để đặt sản phẩm"""
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        if stock_w < prod_w or stock_h < prod_h:
            return None

        for y in range(stock_h - prod_h + 1):
            for x in range(stock_w - prod_w + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    return (x, y)
        return None

class GeneticAlgorithm(Policy):
    def __init__(self, population_size=300, penalty=2, mutation_rate=0.1):
        """
        Hàm khởi tạo lớp GeneticAlgorithm để triển khai thuật toán di truyền.

        Tham số:
        - population_size (int): Số lượng cá thể (chromosome) trong quần thể.
        - penalty (float): Hệ số phạt áp dụng cho nhu cầu chưa được đáp ứng.
        - mutation_rate (float): Xác suất xảy ra đột biến trong mỗi bước tiến hóa.
        """
        self.MAX_ITERATIONS = 150
        self.POPULATION_SIZE = population_size
        self.stock_length = 0
        self.stock_width = 0
        self.length_arr = []
        self.width_arr = []
        self.demand_arr = []
        self.N = None
        self.penalty = penalty
        self.mutation_rate = mutation_rate

    # Phần tạo mẫu (Generate Efficient Patterns)
    def generate_efficient_patterns(self):
        patterns = []
        stack = [([0] * self.N, 0, 0)]  # start with an empty pattern

        while stack:
            current_pattern, length_used, width_used = stack.pop()

            for i in range(self.N):
                max_repeat = min(
                    (self.stock_length - length_used) // self.length_arr[i],
                    (self.stock_width - width_used) // self.width_arr[i],
                    self.demand_arr[i]
                )
                if max_repeat > 0:
                    new_pattern = current_pattern.copy()
                    new_pattern[i] += max_repeat
                    patterns.append(new_pattern)
                    stack.append((new_pattern, length_used + max_repeat * self.length_arr[i],
                                  width_used + max_repeat * self.width_arr[i]))

        return patterns

    def calculate_max_pattern_repetition(self, patterns_arr):
        result = []
        for pattern in patterns_arr:
            maxRep = 0
            for i in range(len(pattern)):
                if pattern[i] > 0:
                    neededRep = ceil(self.demand_arr[i] / pattern[i])
                    if neededRep > maxRep:
                        maxRep = neededRep
            result.append(maxRep)
        return result

    def initialize_population(self, max_repeat_arr):
        init_population = []
        for _ in range(self.POPULATION_SIZE):
            chromosome = []
            for i in np.argsort(-np.array(self.length_arr) * np.array(self.width_arr)):
                chromosome.append(i)
                chromosome.append(randint(1, max_repeat_arr[i]))
            init_population.append(chromosome)
        return init_population

    # Phần đánh giá fitness
    def evaluate_fitness(self, chromosome, patterns_arr):
        P = self.penalty
        unsupplied_sum = 0
        provided = [0] * self.N
        total_unused_area = 0

        if self.stock_length == 0 or self.stock_width == 0:
            raise ValueError("Stock dimensions (length or width) are not properly initialized.")

        stock_area = self.stock_length * self.stock_width
        if stock_area == 0:
            stock_area = 1

        for i in range(0, len(chromosome), 2):
            pattern_index = chromosome[i]
            repetition = chromosome[i + 1]
            pattern = patterns_arr[pattern_index]

            for j in range(len(pattern)):
                provided[j] += pattern[j] * repetition

            pattern_area = sum(
                pattern[j] * self.length_arr[j] * self.width_arr[j] for j in range(len(pattern))
            )
            total_unused_area += stock_area - pattern_area * repetition

        for i in range(self.N):
            unsupplied = max(0, self.demand_arr[i] - provided[i])
            unsupplied_sum += unsupplied * self.length_arr[i] * self.width_arr[i]

        fitness = (
            0.7 * (1 - total_unused_area / stock_area)
            - 0.3 * (P * unsupplied_sum / sum(self.demand_arr))
        )

        return fitness

    # Phần lai ghép và đột biến
    @staticmethod
    def crossover(parent1, parent2):
        child = []
        for i in range(len(parent1)):
            if random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    @staticmethod
    def mutate(chromosome, mutation_rate, max_repeat_arr):
        mutated_chromosome = chromosome[:]
        for i in range(0, len(chromosome), 2):  # Xét từng cặp (pattern_index, repetition)
            if random() < mutation_rate and i + 1 < len(chromosome):
                pattern_index = mutated_chromosome[i]
                mutated_chromosome[i + 1] = randint(1, max_repeat_arr[pattern_index])
        return mutated_chromosome

    # Phần chọn cha mẹ và tạo quần thể mới
    @staticmethod
    def select_parents1(population, fitness_scores):
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return choice(population)
        probabilities = [fitness / total_fitness for fitness in fitness_scores]
        return choices(population, probabilities)[0]

    @staticmethod
    def select_parents2(population, fitness_scores, tournament_size=5):
        indices = choices(range(len(population)), k=tournament_size)
        tournament = [population[i] for i in indices]
        tournament_scores = [fitness_scores[i] for i in indices]
        best_index = tournament_scores.index(max(tournament_scores))
        return tournament[best_index]

    def select_new_population(self, population, fitness_scores, patterns_arr, mutation_rate, max_repeat_arr, selection_type="tournament"):
        new_population = []
        for _ in range(len(population) // 2):
            if selection_type == "tournament":
                parent1 = self.select_parents1(population, fitness_scores)
                parent2 = self.select_parents1(population, fitness_scores)
            elif selection_type == "roulette":
                parent1 = self.select_parents2(population, fitness_scores)
                parent2 = self.select_parents2(population, fitness_scores)

            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)

            child1 = self.mutate(child1, mutation_rate, max_repeat_arr)
            child2 = self.mutate(child2, mutation_rate, max_repeat_arr)

            new_population.extend([child1, child2])

        return new_population

    # Phần chạy thuật toán di truyền
    def run(self, population, patterns_arr, max_repeat_arr, problem_path, queue=None):
        start_time = time.time()
        best_results = []
        num_iters_same_result = 0
        last_result = float('inf')

        for count in range(self.MAX_ITERATIONS):
            fitness_pairs = [(ch, self.evaluate_fitness(ch, patterns_arr)) for ch in population]
            fitness_pairs.sort(key=lambda x: x[1], reverse=True)

            best_solution, best_fitness = fitness_pairs[0]
            best_results.append(best_fitness)

            if abs(best_fitness - last_result) < 1e-5:
                num_iters_same_result += 1
            else:
                num_iters_same_result = 0
            last_result = best_fitness

            if num_iters_same_result >= 100 or best_fitness == 1:
                break

            next_generation = [fitness_pairs[i][0] for i in range(3)]

            while len(next_generation) < self.POPULATION_SIZE:
                parent1 = self.select_parents1([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])
                parent2 = self.select_parents1([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])

                child1 = self.mutate(self.crossover(parent1, parent2), self.mutation_rate, max_repeat_arr)
                child2 = self.mutate(self.crossover(parent2, parent1), self.mutation_rate, max_repeat_arr)
                next_generation.extend([child1, child2])

            population = deepcopy(next_generation[:self.POPULATION_SIZE])

            if queue is not None:
                queue.put((count, best_solution, best_fitness, time.time() - start_time))

        end_time = time.time()
        return best_solution, best_fitness, best_results, end_time - start_time

    # Phần đưa ra hành động dựa trên quan sát
    def get_action(self, observation, info):
        list_prods = observation["products"]
        stocks = observation["stocks"]

        if not stocks or not list_prods:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        self.length_arr = [prod["size"][0] for prod in list_prods if prod["quantity"] > 0]
        self.width_arr = [prod["size"][1] for prod in list_prods if prod["quantity"] > 0]
        self.demand_arr = [prod["quantity"] for prod in list_prods if prod["quantity"] > 0]
        self.N = len(self.length_arr)

        if self.N == 0:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        first_stock = stocks[0]
        self.stock_length, self.stock_width = self._get_stock_size_(first_stock)

        patterns_arr = self.generate_efficient_patterns()
        max_repeat_arr = self.calculate_max_pattern_repetition(patterns_arr)
        population = self.initialize_population(max_repeat_arr)

        best_solution, _, _, _ = self.run(population, patterns_arr, max_repeat_arr, None)

        for i in range(0, len(best_solution), 2):
            pattern_index = best_solution[i]
            repetition = best_solution[i + 1]
            pattern = patterns_arr[pattern_index]

            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                for x in range(stock_w):
                    for y in range(stock_h):
                        if pattern_index >= len(self.length_arr):
                            continue
                        prod_size = (self.length_arr[pattern_index], self.width_arr[pattern_index])

                        if self._can_place_(stock, (x, y), prod_size):
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (x, y)
                            }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
