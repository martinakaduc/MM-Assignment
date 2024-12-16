import numpy as np
from random import randint, random, sample
from copy import deepcopy
from math import ceil
from policy import Policy

class Policy2350030(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = GeneticAlgorithmPolicy()
        elif policy_id == 2:
            self.policy = SkylineAlgorithmPolicy()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

class GeneticAlgorithmPolicy(Policy):
    def __init__(self):
        self.MAX_ITER = 10
        self.pop_size = 100
        self.generations = 100
        self.mutation_rate = 0.01
        self.elite_size = 10
        self.population = []
        self.best_solution = None
        self.lengthArr = []
        self.widthArr = []
        self.demandArr = []
        self.N = 0

    def initialize_population(self, maxRepeatArr):
        sorted_indices = np.argsort(-np.array(self.lengthArr) * np.array(self.widthArr))
        return [
            [i, randint(1, maxRepeatArr[i])]
            for _ in range(self.pop_size)
            for i in sorted_indices
        ]

    def _can_place_(self, stock, position, prod_size, rotated=False):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size if not rotated else prod_size[::-1]
        return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)

    def calculate_fitness(self, chromosome, patterns):
        return sum(
            np.sum(patterns[pattern_index]) * weight
            for pattern_index, weight in zip(chromosome[::2], chromosome[1::2])
            if pattern_index < len(patterns)
        )

    def generate_efficient_patterns(self, stockLength, stockWidth):
        patterns = []
        stack = [([0] * self.N, 0, 0)]
        while stack:
            current_pattern, length_used, width_used = stack.pop()
            for i in range(self.N):
                if i >= len(self.lengthArr) or i >= len(self.widthArr) or i >= len(self.demandArr):
                    continue
                max_repeat = min(
                    (stockLength - length_used) // self.lengthArr[i],
                    (stockWidth - width_used) // self.widthArr[i],
                    self.demandArr[i],
                )
                if max_repeat > 0:
                    new_pattern = current_pattern.copy()
                    new_pattern[i] += max_repeat
                    patterns.append(new_pattern)
                    stack.append(
                        (
                            new_pattern,
                            length_used + max_repeat * self.lengthArr[i],
                            width_used + max_repeat * self.widthArr[i],
                        )
                    )
        return patterns

    def max_pattern_exist(self, patterns):
        return [
            max(ceil(self.demandArr[i] / pattern[i]) for i in range(len(pattern)) if pattern[i] > 0)
            for pattern in patterns
        ]

    def crossover(self, parent1, parent2):
        crossover_point = randint(0, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(self, chromosome, mutation_rate):
        for i in range(len(chromosome)):
            if random() < mutation_rate:
                swap_idx = randint(0, len(chromosome) - 1)
                chromosome[i], chromosome[swap_idx] = chromosome[swap_idx], chromosome[i]
        return chromosome

    def select_parents(self, fitness_s, population):
        total_fitness = np.sum(fitness_s)
        pick = np.random.rand() * total_fitness
        cumulative_fitness = np.cumsum(fitness_s)
        parent_index = np.searchsorted(cumulative_fitness, pick)
        return population[parent_index]

    def evolve(self, population, new_population, patterns, fitness_s, mutation_rate, max_repeat_arr):
        while len(new_population) < self.pop_size:
            parent1 = self.select_parents(fitness_s, population)
            parent2 = self.select_parents(fitness_s, population)
            while parent1 == parent2:
                parent2 = self.select_parents(fitness_s, population)
            child = self.crossover(parent1, parent2)
            self.mutate(child, mutation_rate)
            new_population.append(child)
        return new_population

    def run_genetic_algorithm(self, patterns, population, max_repeat_arr):
        best_results = []
        for _ in range(self.MAX_ITER):
            fitness_pairs = [
                (ch, self.calculate_fitness(ch, patterns)) for ch in self.population
            ]
            fitness_pairs.sort(key=lambda x: x[1], reverse=True)
            new_population = deepcopy([sc[0] for sc in fitness_pairs[: self.elite_size]])
            best_solution, best_fitness = fitness_pairs[0]
            best_results.append(best_fitness)
            next_gen = self.evolve(
                population,
                new_population,
                patterns,
                [sc[1] for sc in fitness_pairs],
                self.mutation_rate,
                max_repeat_arr,
            )
            self.population = deepcopy(next_gen[: self.pop_size])
        return best_solution, best_fitness, best_results

    def create_new_pop(self, population):
        return [
            self.mutate(self.crossover(*sample(population, 2)), self.mutation_rate)
            for _ in range(self.pop_size)
        ]

    def get_action(self, observation, info): 
        list_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        stocks = observation["stocks"]

        if not list_prods or not stocks:
            return self._get_empty_action()

        self.lengthArr = [prod["size"][0] for prod in list_prods if prod["quantity"] > 0]
        self.widthArr = [prod["size"][1] for prod in list_prods if prod["quantity"] > 0]
        self.demandArr = [prod["quantity"] for prod in list_prods if prod["quantity"] > 0]
        self.N = len(self.lengthArr)

        if self.N == 0:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

        for stock_idx, stock in enumerate(stocks):
            stock_Length, stock_Width = self._get_stock_size_(stock)
            patterns = self.generate_efficient_patterns(stock_Length, stock_Width)

            for pattern_index, pattern in enumerate(patterns):
                for x in range(stock_Width):
                    for y in range(stock_Length):
                        if pattern_index >= len(self.lengthArr):
                            continue
                        prod_size = (self.lengthArr[pattern_index], self.widthArr[pattern_index])
                        if self._can_place_(stock, (x, y), prod_size):
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (x, y),
                                "rotated": False,
                            }
                        elif self._can_place_(stock, (x, y), prod_size, rotated=True):
                            return {
                                "stock_idx": stock_idx,
                                "size": (prod_size[1], prod_size[0]),
                                "position": (x, y),
                                "rotated": True,
                            }

        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    @staticmethod
    def _get_empty_action():
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

from policy import Policy

class SkylineAlgorithmPolicy(Policy):
    def __init__(self):
        self.skyline = []
        self.stock_width = 0
        self.stock_height = 0
        self.height_cache = {}

    def generate_efficient_patterns(self, stock_length, stock_width):
        patterns = []
        for i in range(self.N):
            pattern = [0] * self.N
            pattern[i] = 1
            patterns.append(pattern)
        return patterns

    def get_action(self, observation, info):
        list_prods = sorted(observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        stocks = observation["stocks"]
        if not list_prods or not stocks:
            return self._get_empty_action()
        self.lengthArr = [prod["size"][0] for prod in list_prods if prod["quantity"] > 0]
        self.widthArr = [prod["size"][1] for prod in list_prods if prod["quantity"] > 0]
        self.demandArr = [prod["quantity"] for prod in list_prods if prod["quantity"] > 0]
        self.N = len(self.lengthArr)

        if self.N == 0:
            return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
        
        for stock_idx, stock in enumerate(stocks):
            stock_Length, stock_Width = self._get_stock_size_(stock)
            patterns = self.generate_efficient_patterns(stock_Length, stock_Width)
            for pattern_index, pattern in enumerate(patterns):
                for x in range(stock_Width):
                    for y in range(stock_Length):
                        if pattern_index >= len(self.lengthArr):
                            continue
                        prod_size = (self.lengthArr[pattern_index], self.widthArr[pattern_index])
                        if self._can_place_(stock, (x, y), prod_size):
                            return {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (x, y),
                                "rotated": False,
                            }
                        elif self._can_place_(stock, (x, y), prod_size, rotated=True):
                            return {
                                "stock_idx": stock_idx,
                                "size": (prod_size[1], prod_size[0]),
                                "position": (x, y),
                                "rotated": True,
                            }
        return self._get_empty_action()
    
    def _can_place_(self, stock, position, prod_size, rotated=False):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size if not rotated else prod_size[::-1]
        return np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)

    def calculate_used_area(self):
        used_area = 0
        for i in range(1, len(self.skyline)):
            width = self.skyline[i][0] - self.skyline[i - 1][0]
            height = self.skyline[i - 1][1]
            used_area += width * height
        return used_area

    def _initialize_new_stock(self, stock):
        self.stock_width, self.stock_height = self._get_stock_size_(stock)
        self.skyline = [(0, 0)]
        self.height_cache.clear()

    def _try_place_product(self, prod):
        width, height = prod["size"]
        orientations = [(width, height, False), (height, width, True)]
        best_position = None
        best_rotation = False
        min_waste = float('inf')
        for w, h, rot in orientations:
            pos = self.add_rectangle(w, h)
            if pos:
                waste = self.calculate_used_area()
                if waste < min_waste:
                    min_waste = waste
                    best_position = pos
                    best_rotation = rot
        if best_position:
            return {
                "size": [height, width] if best_rotation else [width, height],
                "position": best_position,
                "rotated": best_rotation
            }
        return None
    
    def add_rectangle(self, rect_width, rect_height):
        if rect_width > self.stock_width or rect_height > self.stock_height:
            return None
        pos = self.find_position(rect_width, rect_height)
        if pos is not None:
            self.update_skyline(pos[0], rect_width, pos[1] + rect_height)
        return pos

    def find_position(self, rect_width, rect_height):
        if not self.skyline:
            return (0, 0) if self._fits_in_stock(rect_width, rect_height) else None
        return self._find_best_position(rect_width, rect_height)

    def _get_cached_height(self, x, width):
        if (x, width) not in self.height_cache:
            self.height_cache[(x, width)] = self._calculate_height(x, width)
        return self.height_cache[(x, width)]

    def _calculate_height(self, x, width):
        max_height = 0
        for i in range(x, x + width):
            max_height = max(max_height, self._get_height_at(i))
        return max_height

    def _get_height_at(self, x):
        for i in range(len(self.skyline) - 1, -1, -1):
            if self.skyline[i][0] <= x:
                return self.skyline[i][1]
        return 0

    def _fits_in_stock(self, width, height):
        return width <= self.stock_width and height <= self.stock_height

    def _find_best_position(self, rect_width, rect_height):
        best_x = -1
        best_y = float('inf')
        for i in range(len(self.skyline)):
            x = self.skyline[i][0]
            y = self._get_cached_height(x, rect_width)
            if y + rect_height <= self.stock_height and y < best_y:
                best_x = x
                best_y = y
        return (best_x, best_y) if best_x != -1 else None

    def update_skyline(self, x, width, height):
        new_skyline = []
        for i in range(len(self.skyline)):
            if self.skyline[i][0] < x:
                new_skyline.append(self.skyline[i])
            elif self.skyline[i][0] >= x + width:
                new_skyline.append((x + width, height))
                new_skyline.extend(self.skyline[i:])
                break
        self.skyline = new_skyline

    def _get_stock_size_(self, stock):
        return stock.shape[1], stock.shape[0]

    def _get_empty_action(self):
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}