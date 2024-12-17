from policy import Policy
from copy import deepcopy
from random import randint, random, choice, choices
import time
import numpy as np
import gymnasium as gym
from math import ceil


class GeneticOptimizer:
    """
    A Genetic Algorithm optimizer for various chromosome patterns.
    It requires:
    - A fitness function
    - Crossover and mutation operators
    - Selection methods
    """

    def __init__(self, population_size, max_iterations):
        self.population_size = population_size
        self.max_iterations = max_iterations

    def run(self, population, fitness_func, crossover_func, mutate_func, select_func1, select_func2, queue=None):
        """
        Run the GA optimization loop.

        Parameters:
        - population: initial population of chromosomes
        - fitness_func: function(chromosome) -> fitness_value
        - crossover_func: function(parent1, parent2) -> child
        - mutate_func: function(chromosome) -> mutated_chromosome
        - select_func1: selection method 1 (e.g., roulette)
        - select_func2: selection method 2 (e.g., tournament)

        Returns:
        - best_solution, best_fitness, fitness_history, elapsed_time
        """
        start_time = time.time()
        last_best = float('-inf')  # Initialize to negative infinity for maximization
        stable_count = 0
        fitness_history = []

        for iteration in range(self.max_iterations):
            # Đánh giá population
            scored = [(ch, fitness_func(ch)) for ch in population]
            scored.sort(key=lambda x: x[1], reverse=True)  # Sort descending by fitness

            best_sol, best_fit = scored[0]
            fitness_history.append(best_fit)

            if abs(best_fit - last_best) < 1e-5:
                stable_count += 1
            else:
                stable_count = 0
            last_best = best_fit

            # Early stop
            if stable_count >= 100 or best_fit >= 1.0:
                break

            # Elitism: giữ lại top 3 cá thể tốt nhất
            next_gen = [scored[i][0] for i in range(min(3, len(scored)))]
            pop_chroms = [s[0] for s in scored]
            pop_fits = [s[1] for s in scored]

            # Sinh thế hệ mới với chiến lược sinh sản nâng cao
            while len(next_gen) < self.population_size:
                # Chọn phương pháp chọn lọc ngẫu nhiên
                if random() < 0.5:
                    p1 = select_func1(pop_chroms, pop_fits)
                    p2 = select_func2(pop_chroms, pop_fits)
                else:
                    p1 = select_func2(pop_chroms, pop_fits)
                    p2 = select_func2(pop_chroms, pop_fits)

                # Lai ghép và đột biến để tạo 2 con mới
                c1 = mutate_func(crossover_func(p1, p2))
                c2 = mutate_func(crossover_func(p2, p1))

                # Thêm các con mới vào thế hệ tiếp theo
                next_gen.extend([c1, c2])

                # Nếu số lượng quần thể chưa đủ, thêm các cá thể từ quần thể cũ
                while len(next_gen) < self.population_size:
                    # Chọn ngẫu nhiên các cá thể từ quần thể cũ để bổ sung
                    # Ưu tiên chọn các cá thể có fitness cao hơn
                    additional_chrom = select_func1(pop_chroms, pop_fits)
                    next_gen.append(additional_chrom)

            # Cập nhật quần thể
            population = deepcopy(next_gen[:self.population_size])

            if queue is not None:
                queue.put((iteration, best_sol, best_fit, time.time() - start_time))

        end_time = time.time()
        return best_sol, best_fit, fitness_history, end_time - start_time

    @staticmethod
    def roulette_selection(population, fitness_scores):
        total_fit = sum(fitness_scores)
        if total_fit <= 0:
            return choice(population)
        probs = [f / total_fit for f in fitness_scores]
        return choices(population, weights=probs, k=1)[0]

    @staticmethod
    def tournament_selection(population, fitness_scores, k=5):
        selected = choices(list(zip(population, fitness_scores)), k=k)
        return max(selected, key=lambda x: x[1])[0]


class PatternGenerator:
    """
    A class to generate patterns for the cutting stock problem and calculate related parameters.
    """

    def __init__(self, stock_length, stock_width, length_arr, width_arr, demand_arr):
        self.stock_length = stock_length
        self.stock_width = stock_width
        self.lengths = length_arr
        self.widths = width_arr
        self.demands = demand_arr
        self.num_items = len(self.lengths)

    def gen_complex_patterns(self):
        patterns = []
        stack = [([0] * self.num_items, 0, 0)]

        def calculate_max_rep(length, width, used_len, used_wid, pattern_i, demand_i):
            """Helper function to calculate maximum repetitions for a piece"""
            return min(
                (self.stock_length - used_len) // length,
                (self.stock_width - used_wid) // width,
                demand_i - pattern_i
            )

        while stack:
            pattern, used_len, used_wid = stack.pop()

            # Sort pieces by area (largest first)
            piece_indices = sorted(range(self.num_items),
                                   key=lambda i: self.lengths[i] * self.widths[i],
                                   reverse=True)

            for i in piece_indices:
                # Check normal orientation
                max_rep = calculate_max_rep(
                    self.lengths[i], self.widths[i],
                    used_len, used_wid,
                    pattern[i], self.demands[i]
                )

                # Check rotated orientation if piece isn't square
                if self.lengths[i] != self.widths[i]:
                    max_rep_rotated = calculate_max_rep(
                        self.widths[i], self.lengths[i],
                        used_len, used_wid,
                        pattern[i], self.demands[i]
                    )
                    max_rep = max(max_rep, max_rep_rotated)

                if max_rep > 0:
                    for rep in range(1, max_rep + 1):
                        new_pat = pattern.copy()
                        new_pat[i] += rep
                        patterns.append(new_pat)

                        # Add to stack for further exploration
                        stack.append(
                            (new_pat,
                             used_len + rep * self.lengths[i],
                             used_wid + rep * self.widths[i])
                        )

        return patterns

    def max_repetition_ga(self, patterns):
        """Calculate maximum repetition for GA patterns."""
        res = []
        for pat in patterns:
            max_rep = 0
            for i, val in enumerate(pat):
                if val > 0:
                    needed = ceil(self.demands[i] / val)
                    if needed > max_rep:
                        max_rep = needed
            res.append(max_rep)
        return res


class GeneticPolicy(Policy):
    def __init__(self, population_size=300, max_iterations=2000, penalty=2, mutation_rate=0.1):
        super().__init__()

        self.stockLength = 0
        self.stockWidth = 0
        self.lengthArr = []
        self.widthArr = []
        self.demandArr = []
        self.optimizer = GeneticOptimizer(
            population_size=population_size,
            max_iterations=max_iterations,
        )
        self.penalty = penalty
        self.mutation_rate = mutation_rate

    def calculate_fitness(self, chromosome, patterns):
        """Hàm fitness đơn giản hóa, tập trung vào 2 yếu tố chính"""
        if self.stockLength <= 0 or self.stockWidth <= 0:
            return float('-inf')

        # Tính số lượng miếng đã cắt được
        provided = np.zeros(len(self.lengthArr))
        total_used_area = 0.0

        # Duyệt qua các pattern được sử dụng
        for i in range(0, len(chromosome), 2):
            pat_idx = chromosome[i]
            rep = chromosome[i + 1]
            if 0 <= pat_idx < len(patterns):
                pattern = patterns[pat_idx]
                provided += np.array(pattern) * rep
                # Tính diện tích đã sử dụng
                pattern_area = sum(pat * l * w for pat, l, w in
                                   zip(pattern, self.lengthArr, self.widthArr))
                total_used_area += pattern_area * rep

        demand_ratio = min(1.0, min(provided / np.maximum(1, self.demandArr)))

        usage_ratio = total_used_area / (self.stockLength * self.stockWidth)

        return 0.75 * demand_ratio + 0.25 * usage_ratio

    def get_action(self, observation, info):
        products = observation.get("products", [])
        stocks = observation.get("stocks", [])

        if not products or not stocks:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        self.lengthArr = [p["size"][0] for p in products if p["quantity"] > 0]
        self.widthArr = [p["size"][1] for p in products if p["quantity"] > 0]
        self.demandArr = [p["quantity"] for p in products if p["quantity"] > 0]

        if len(self.lengthArr) == 0:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        self.stockLength, self.stockWidth = self._get_stock_size_(stocks[0])

        # GA Policy
        gen = PatternGenerator(self.stockLength, self.stockWidth, self.lengthArr, self.widthArr, self.demandArr)
        patterns = gen.gen_complex_patterns()
        if not patterns:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        max_rep = gen.max_repetition_ga(patterns)

        # Initialize population
        item_sizes = np.array(self.lengthArr) * np.array(self.widthArr)
        order = list(np.argsort(-item_sizes))
        population = []
        for _ in range(self.optimizer.population_size):
            chromosome = []
            for idx in order:
                chromosome.append(idx)
                chromosome.append(randint(1, max_rep[idx]))
            population.append(chromosome)

        # Run GA
        fitness_func = lambda ch: self.calculate_fitness(ch, patterns)
        # crossover_func = self._ga_crossover
        crossover_func = self.enhanced_crossover
        # mutate_func = lambda ch: self._ga_mutate(ch, max_rep)
        mutate_func = lambda ch: self.enhanced_mutate(ch, max_rep)
        select_func1 = GeneticOptimizer.roulette_selection
        select_func2 = GeneticOptimizer.tournament_selection

        best_solution, best_fitness, fitness_history, elapsed_time = self.optimizer.run(
            population, fitness_func, crossover_func, mutate_func, select_func1, select_func2
        )
        # print("Best solution: ", best_solution, "Best fitness: ", best_fitness)

        # Try to place best solution
        for i in range(0, len(best_solution), 2):
            pat_idx = best_solution[i]
            for stock_idx, stock in enumerate(stocks):
                sw, sh = self._get_stock_size_(stock)
                for x in range(sw):
                    for y in range(sh):
                        if pat_idx < len(self.lengthArr):
                            prod_size = (self.lengthArr[pat_idx], self.widthArr[pat_idx])
                            if self._can_place_(stock, (x, y), prod_size):
                                return {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": (x, y)
                                }
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    def _can_place_(self, stock, position, size):
        pos_x, pos_y = position
        width, height = size
        stock_w, stock_h = self._get_stock_size_(stock)

        # Add explicit boundary check
        if pos_x + width > stock_w or pos_y + height > stock_h:
            return False

        return np.all(stock[pos_x:pos_x + width, pos_y:pos_y + height] == -1)

    @staticmethod
    def enhanced_crossover(p1, p2):
        """Uniform crossover that preserves pattern-repetition pairs"""
        child = []
        for i in range(0, len(p1), 2):
            # Keep pattern and repetition together
            if random() < 0.5:
                child.extend([p1[i], p1[i + 1]])
            else:
                child.extend([p2[i], p2[i + 1]])
        return child

    def enhanced_mutate(self, chromosome, max_repeat_arr):
        """Strategic mutation with occasional pattern changes"""
        mutated = chromosome.copy()
        for i in range(0, len(mutated), 2):
            if random() < self.mutation_rate:
                idx = mutated[i]
                if 0 <= idx < len(max_repeat_arr):
                    # Small chance to change pattern
                    if random() < 0.1:
                        new_idx = np.random.randint(0, len(max_repeat_arr))
                        mutated[i] = new_idx
                        mutated[i + 1] = randint(1, max_repeat_arr[new_idx])
                    else:
                        # Usually just adjust repetition
                        current = mutated[i + 1]
                        max_rep = max_repeat_arr[idx]
                        # Allow small adjustments more often than large ones
                        delta = randint(-2, 2)
                        mutated[i + 1] = max(1, min(max_rep, current + delta))
        return mutated
