from policy import Policy
from math import floor, ceil
from copy import deepcopy
from random import randint, shuffle, random, choice, choices
import time  # Ensure 'time' is imported
class Policy2310139_2310090_2310191_2310242_2310423(Policy):
    def __init__(self, populationSize = 300, penalty = 2, mutationRate = 0.1):
        # Student code here
        """
        Hàm khởi tạo lớp GeneticPolicy để triển khai thuật toán di truyền.

        Tham số:
        - populationSize (int): Số lượng cá thể (chromosome) trong quần thể.
        - penalty (float): Hệ số phạt áp dụng cho nhu cầu chưa được đáp ứng.
        - mutationRate (float): Xác suất xảy ra đột biến trong mỗi bước tiến hóa.
        """
        self.MAX_ITERATIONS = 2000 # Số vòng lặp tối đa cho thuật toán
        self.POPULATION_SIZE = populationSize
        self.stockLength = None  # Chiều dài của kho nguyên liệu
        self.stockWidth = None # Chiều rộng của kho nguyên liệu
        self.lengthArr = [] # Mảng chứa chiều dài của các mẫu
        self.widthArr = [] # Mảng chứa chiều rộng của các mẫu
        self.demandArr = [] # Mảng chứa số lượng yêu cầu các mẫu
        self.N = None # Số lượng mẫu
        self.penalty = penalty
        self.mutationRate = mutationRate

   
    def generate_efficient_patterns(self):
        result = []
        availableArr = [
                min(
                        floor(self.stockLength / self.lengthArr[0]),
                        floor(self.stockWidth / self.widthArr[0]),
                        self.demandArr[0]
                )
        ]

        for n in range(1, self.N):
                s = sum(availableArr[j] * self.lengthArr[j] for j in range(n))
                availableArr.append(
                        min(
                                floor((self.stockLength - s) / self.lengthArr[n]),
                                floor((self.stockWidth - s) / self.widthArr[n]),
                                self.demandArr[n]
                        )
                )

        while True:
                result.append(availableArr.copy())
                for j in range(len(availableArr) - 1, -1, -1):
                        if availableArr[j] > 0:
                                availableArr[j] -= 1
                                for k in range(j + 1, self.N):
                                        s = sum(availableArr[m] * self.lengthArr[m] for m in range(k))
                                        availableArr[k] = min(
                                                floor((self.stockLength - s) / self.lengthArr[k]),
                                                floor((self.stockWidth - s) / self.widthArr[k]),
                                                self.demandArr[k]
                                        )
                                break
                else:
                        break
        return result



    def calculate_max_pattern_repetition(self, patternsArr):
        """
            Tính số lần tối đa mỗi mẫu (pattern) có thể được sử dụng.

            Tham số:
            - patternsArr: Danh sách các mẫu.

            Kết quả:
            - Trả về danh sách số lần tối đa mỗi mẫu có thể lặp lại.
            """
        result = [] # Danh sách lưu trữ số lần lặp tối đa cho từng mẫu
        for pattern in patternsArr:
                maxRep = 0
                for i in range(len(pattern)):
                    if pattern[i] > 0:
                        # Tính số lần cần lặp để đáp ứng nhu cầu
                        neededRep = ceil(self.demandArr[i] / pattern[i])
                        if neededRep > maxRep:
                            maxRep = neededRep
                result.append(maxRep)
        return result


    def initialize_population(self, maxRepeatArr):
        initPopulation = []
        for _ in range(self.POPULATION_SIZE):
            chromosome = []
            indices = list(range(len(maxRepeatArr)))
            shuffle(indices)
            for idx in indices:
                chromosome.append(idx)
                chromosome.append(randint(1, maxRepeatArr[idx]))
            initPopulation.append(chromosome)
        return initPopulation


    def evaluate_fitness(self, chromosome, patterns_arr):
        P = self.penalty
        unsupplied_sum = 0
        provided = [0] * self.N
        total_unused_area = 0  # Track unused area

        for i in range(0, len(chromosome), 2):
            pattern_index = chromosome[i]
            repetition = chromosome[i + 1]
            pattern = patterns_arr[pattern_index]

            for j in range(len(pattern)):
                provided[j] += pattern[j] * repetition

            # Simulate stock usage for unused area calculation
            pattern_area = sum(
                pattern[j] * self.lengthArr[j] * self.widthArr[j] for j in range(len(pattern))
            )
            total_unused_area += self.stockLength * self.stockWidth - pattern_area * repetition

        for i in range(self.N):
            unsupplied = max(0, self.demandArr[i] - provided[i])
            unsupplied_sum += unsupplied * self.lengthArr[i] * self.widthArr[i]

        fitness = (
            0.7 * (1 - total_unused_area / (self.stockLength * self.stockWidth))  # Prioritize material usage
            - 0.3 * (P * unsupplied_sum / sum(self.demandArr))  # Penalize unsupplied products proportionally
        )

        return fitness



    def run(self, population, patterns_arr, max_repeat_arr, problem_path, queue=None):
        """
        Run the Genetic Algorithm to solve the 2D cutting stock problem.

        Parameters:
        - population: Initial population of chromosomes.
        - patterns_arr: List of feasible patterns.
        - max_repeat_arr: Maximum repetitions for each pattern.
        - problem_path: Problem instance file path (not used directly here).
        - queue: Optional queue for communication during execution.

        Returns:
        - Best solution, its fitness, fitness history, and execution time.
        """
        start_time = time.time()
        best_results = []
        num_iters_same_result = 0
        last_result = float('inf')

        for count in range(self.MAX_ITERATIONS):
            # Evaluate fitness for the population
            fitness_pairs = [(ch, self.evaluate_fitness(ch, patterns_arr)) for ch in population]
            fitness_pairs.sort(key=lambda x: x[1], reverse=True)

            # Track the best result
            best_solution, best_fitness = fitness_pairs[0]
            best_results.append(best_fitness)

            # Convergence check
            if abs(best_fitness - last_result) < 1e-5:
                num_iters_same_result += 1
            else:
                num_iters_same_result = 0
            last_result = best_fitness

            # Early termination if converged
            if num_iters_same_result >= 100 or best_fitness == 1:
                break

            # Preserve top 3 (elitism)
            next_generation = [fitness_pairs[i][0] for i in range(3)]

            # Create new population
            while len(next_generation) < self.POPULATION_SIZE:
                if random() < 0.5:
                    parent1 = self.select_parents1([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])
                    parent2 = self.select_parents1([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])
                else:
                    parent1 = self.select_parents2([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])
                    parent2 = self.select_parents2([fp[0] for fp in fitness_pairs], [fp[1] for fp in fitness_pairs])

                # Generate offspring with crossover and appropriate mutation
                if count < self.MAX_ITERATIONS * 0.5:
                    child1 = self.mutate2(self.crossover(parent1, parent2), max_repeat_arr, patterns_arr)
                    child2 = self.mutate2(self.crossover(parent2, parent1), max_repeat_arr, patterns_arr)
                else:
                    child1 = self.mutate(self.crossover(parent1, parent2), max_repeat_arr)
                    child2 = self.mutate(self.crossover(parent2, parent1), max_repeat_arr)

                next_generation.extend([child1, child2])

            # Update population for the next iteration
            population = deepcopy(next_generation[:self.POPULATION_SIZE])

            # Communicate progress if using a queue
            if queue is not None:
                queue.put((count, best_solution, best_fitness, time.time() - start_time))

        end_time = time.time()

        return best_solution, best_fitness, best_results, end_time - start_time


    @staticmethod
    # def mutate(chromosome, mutation_rate, max_repeat_arr):
    #     """
    #     Thực hiện đột biến ngẫu nhiên trên một cá thể.

    #     Tham số:
    #     - chromosome: Cá thể cần đột biến.
    #     - mutation_rate: Xác suất đột biến.
    #     - max_repeat_arr: Giới hạn số lần lặp tối đa của từng mẫu.

    #     Kết quả:
    #     - Trả về cá thể sau khi đột biến.
    #     """
    #     mutated_chromosome = chromosome[:]
    #     for i in range(0, len(chromosome), 2):  # Xét từng cặp (pattern_index, repetition)
    #         if random() < mutation_rate and i + 1 < len(chromosome) :
    #             # Thay đổi số lần lặp trong giới hạn
    #             pattern_index = mutated_chromosome[i]
    #             mutated_chromosome[i+1] = randint(1, max_repeat_arr[pattern_index])
    #     return mutated_chromosome
    
    def select_new_population(self,population, fitness_scores, patterns_arr, mutation_rate, max_repeat_arr, selection_type="tournament"):
        """
        Tạo quần thể mới bằng cách chọn lọc, lai ghép và đột biến.

        Tham số:
        - population: Quần thể hiện tại.
        - fitness_scores: Điểm fitness của các cá thể trong quần thể.
        - patterns_arr: Danh sách các mẫu khả thi.
        - mutation_rate: Xác suất đột biến.
        - max_repeat_arr: Số lần lặp tối đa cho từng mẫu.
        - selection_type: Phương pháp chọn lọc ('tournament' hoặc 'roulette').

        Kết quả:
        - Trả về quần thể mới.
        """
        new_population = []
        for _ in range(len(population) // 2):  # Số cặp cha mẹ
            # Chọn cha mẹ
            if selection_type == "tournament":
                parent1 = self.select_parents1(population, fitness_scores)
                parent2 = self.select_parents1(population, fitness_scores)
            elif selection_type == "roulette":
                parent1 = self.select_parents2(population, fitness_scores)
                parent2 = self.select_parents2(population, fitness_scores)

            # Lai ghép
            child1 = self.crossover(parent1, parent2)
            child2 = self.crossover(parent2, parent1)

            # Đột biến
            child1 = self.mutate(child1, mutation_rate, max_repeat_arr)
            child2 = self.mutate(child2, mutation_rate, max_repeat_arr)

            # Thêm cá thể con vào quần thể mới
            new_population.extend([child1, child2])

        return new_population


    # def get_action(self, observation, info):
    #     """
    #     Genetic algorithm to decide the best action.
    #     """
    #     list_prods = observation["products"]
    #     stocks = observation["stocks"]

    #     # Default action
    #     prod_size = [0, 0]
    #     stock_idx = -1
    #     pos_x, pos_y = 0, 0

    #     # Iterate over products
    #     for prod in list_prods:
    #         if prod["quantity"] > 0:
    #             prod_size = prod["size"]

    #             # Apply a genetic-like heuristic to find placement
    #             best_fitness = float("inf")
    #             best_action = None

    #             for i, stock in enumerate(stocks):
    #                 stock_w, stock_h = self._get_stock_size_(stock)
    #                 prod_w, prod_h = prod_size

    #                 if stock_w < prod_w or stock_h < prod_h:
    #                     continue

    #                 # Check all possible positions
    #                 for x in range(stock_w - prod_w + 1):
    #                     for y in range(stock_h - prod_h + 1):
    #                         if self._can_place_(stock, (x, y), prod_size):
    #                             # Fitness function: Minimize unused area
    #                             unused_area = stock_w * stock_h - prod_w * prod_h
    #                             # fitness = unused_area + abs(stock_w - prod_w) + abs(stock_h - prod_h)
    #                             if unused_area < best_fitness:
    #                                 best_fitness = unused_area
    #                                 best_action = {"stock_idx": i, "size": prod_size, "position": (x, y)}

    #             if best_action:
    #                 stock_idx = best_action["stock_idx"]
    #                 pos_x, pos_y = best_action["position"]
    #                 break

    #     return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
    
    def get_action(self, observation, info):
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        # Sort products by size (largest first)
        list_prods = sorted(list_prods, key=lambda prod: prod["size"][0] * prod["size"][1], reverse=True)

        # Pick a product that has quantity > 0
        for prod in list_prods:
            if prod["quantity"] > 0:
                prod_size = prod["size"]

                # Loop through all stocks
                for i, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    if stock_w < prod_w or stock_h < prod_h:
                        continue

                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break

                    if pos_x is not None and pos_y is not None:
                        stock_idx = i
                        break

                if pos_x is not None and pos_y is not None:
                    break

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
    # def get_action(self, observation, info):
    #     list_prods = observation["products"]
    #     stocks = observation["stocks"]

    #     prod_to_cut = None
    #     best_stock_idx = -1
    #     best_position = None
    #     best_unused_area = float("inf")

    #     # Prioritize products by demand (highest first) and size (largest first)
    #     list_prods = sorted(
    #         list_prods,
    #         key=lambda prod: (-prod["quantity"], -prod["size"][0] * prod["size"][1])
    #     )

    #     for prod in list_prods:
    #         if prod["quantity"] <= 0:
    #             continue

    #         prod_size = prod["size"]
    #         prod_w, prod_h = prod_size

    #         for stock_idx, stock in enumerate(stocks):
    #             stock_w, stock_h = self._get_stock_size_(stock)

    #             # Skip stock if product doesn't fit
    #             if stock_w < prod_w or stock_h < prod_h:
    #                 continue

    #             # Try to place product in stock
    #             for x in range(stock_w - prod_w + 1):
    #                 for y in range(stock_h - prod_h + 1):
    #                     if self._can_place_(stock, (x, y), prod_size):
    #                         unused_area = stock_w * stock_h - prod_w * prod_h
    #                         # Optimize: prioritize better placement (lowest unused area)
    #                         if unused_area < best_unused_area:
    #                             best_unused_area = unused_area
    #                             prod_to_cut = prod
    #                             best_stock_idx = stock_idx
    #                             best_position = (x, y)

    #     # If no valid action found, return a no-op
    #     if prod_to_cut is None:
    #         return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    #     # Return the best action found
    #     return {
    #         "stock_idx": best_stock_idx,
    #         "size": prod_to_cut["size"],
    #         "position": best_position,
    #     }
    
    
    
    
    # def get_action(self, observation, info):
    #     list_prods = observation["products"]
    #     stocks = observation["stocks"]

    #     prod_size = [0, 0]
    #     stock_idx = -1
    #     pos_x, pos_y = 0, 0

    #     for prod in list_prods:
    #         if prod["quantity"] > 0:
    #             prod_size = prod["size"]
    #             prod_w, prod_h = prod_size
    #             best_fitness = float("inf")
    #             best_action = None

    #             for i, stock in enumerate(stocks):
    #                 stock_w, stock_h = self._get_stock_size_(stock)

    #                 if stock_w < prod_w or stock_h < prod_h:
    #                     continue

    #                 for x in range(stock_w - prod_w + 1):
    #                     for y in range(stock_h - prod_h + 1):
    #                         if self._can_place_(stock, (x, y), prod_size):
    #                             unused_area = stock_w * stock_h - prod_w * prod_h
    #                             if unused_area < best_fitness:
    #                                 best_fitness = unused_area
    #                                 best_action = {"stock_idx": i, "size": prod_size, "position": (x, y)}

    #             if best_action:
    #                 stock_idx = best_action["stock_idx"]
    #                 pos_x, pos_y = best_action["position"]
    #                 break

    #     return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    
    
    # def _initialize_free_regions(self, stock):
    #     width, height = self._get_stock_size_(stock)
    #     return [(0, 0, width, height)]  # Entire stock is initially free

    # def _update_free_regions(self, regions, used_region, prod_size):
    #     x, y, rw, rh = used_region
    #     pw, ph = prod_size

    #     # Subdivide the remaining free space after placing the product
    #     new_regions = []
    #     if rw > pw:
    #         new_regions.append((x + pw, y, rw - pw, rh))
    #     if rh > ph:
    #         new_regions.append((x, y + ph, rw, rh - ph))

    #     # Include unaffected regions
    #     return [region for region in regions if region != used_region] + new_regions


    
    # def get_action(self, observation, info):
    #     list_prods = observation["products"]
    #     stock_idx = -1
    #     pos_x, pos_y = None, None
    #     prod_size = [0, 0]

    #     # Maintain a list of free spaces for each stock
    #     free_regions = [self._initialize_free_regions(stock) for stock in observation["stocks"]]

    #     for prod in list_prods:
    #         if prod["quantity"] > 0:
    #             prod_size = prod["size"]
    #             prod_w, prod_h = prod_size

    #             # Find best fit region across all stocks
    #             for i, regions in enumerate(free_regions):
    #                 for region in regions:
    #                     x, y, rw, rh = region
    #                     if rw >= prod_w and rh >= prod_h:
    #                         stock_idx = i
    #                         pos_x, pos_y = x, y

    #                         # Update free regions
    #                         free_regions[i] = self._update_free_regions(
    #                             regions, region, prod_size
    #                         )
    #                         break
    #                 if pos_x is not None and pos_y is not None:
    #                     break

    #             if stock_idx != -1:
    #                 break

    #         return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

   
  # Student code here
  # You can add more functions if needed