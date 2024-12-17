import numpy as np
import random
from policy import Policy
import numpy as np
from copy import deepcopy
import time
import tracemalloc
class GA_Optimizer:

    def __init__(self,
                 stock_width,
                 stock_height,
                 w_arr,
                 h_arr,
                 d_arr,
                 population_size=20,
                 generations=50,
                 mutation_rate=0.1):
        self.stock_width = stock_width
        self.stock_height = stock_height
        self.w_arr = w_arr
        self.h_arr = h_arr
        self.d_arr = d_arr
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            chromosome = []
            for w, h in zip(self.w_arr, self.h_arr):
                while True:
                    x = random.randint(0, self.stock_width - w)
                    y = random.randint(0, self.stock_height - h)
                    if is_position_available(
                            np.full((self.stock_height, self.stock_width), -1),
                            x, y, w, h):
                        chromosome.append((x, y, w, h))
                        break
            population.append(chromosome)
        return population

    def evaluate_fitness(self, chromosome):
        used_area = 0
        used_positions = []

        for x, y, width, height in chromosome:
            if x + width > self.stock_width or y + height > self.stock_height:
                continue

            overlapping = False
            for px, py, pw, ph in used_positions:
                if not (x + width <= px or px + pw <= x or y + height <= py
                        or py + ph <= y):
                    overlapping = True
                    break

            if not overlapping:
                used_positions.append((x, y, width, height))
                used_area += width * height

        total_area = self.stock_width * self.stock_height
        return used_area / total_area if total_area > 0 else 0

    def selection(self, population, fitness_scores):
        probabilities = [f / sum(fitness_scores) for f in fitness_scores]
        selected = random.choices(population, probabilities, k=2)
        return selected

    def crossover(self, parent1, parent2):
        """
        Lai ghép hai cá thể để tạo thế hệ mới.
        """
        # Kiểm tra nếu parent1 hoặc parent2 có ít hơn 2 phần tử
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1, parent2  # Trả về chính cha mẹ nếu không thể lai ghép

        # Lai ghép tại điểm ngẫu nhiên
        split_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                x = random.randint(0, self.stock_width - self.w_arr[i])
                y = random.randint(0, self.stock_height - self.h_arr[i])
                chromosome[i] = (x, y, self.w_arr[i], self.h_arr[i])
        return chromosome

    def run(self):
        population = self.initialize_population()

        for generation in range(self.generations):
            fitness_scores = [self.evaluate_fitness(ch) for ch in population]

            if max(fitness_scores) == 0:
                return None, 0

            new_population = []
            for _ in range(self.population_size // 2):
                parent1, parent2 = self.selection(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend(
                    [self.mutate(child1),
                     self.mutate(child2)])

            population = new_population

        best_index = np.argmax(
            [self.evaluate_fitness(ch) for ch in population])
        return population[best_index], fitness_scores[best_index]

def is_position_available(stock, x, y, width, height):
    for i in range(y, y + height):
        for j in range(x, x + width):
            if stock[i, j] != -1:
                return False
    return True

def place_product_in_stock(stock, x, y, product_width, product_height):
    """
    Đặt sản phẩm vào kho và cập nhật trạng thái các ô tương ứng.

    Args:
        stock (numpy.ndarray): Mảng đại diện cho kho.
        x (int): Tọa độ x của vị trí bắt đầu đặt sản phẩm.
        y (int): Tọa độ y của vị trí bắt đầu đặt sản phẩm.
        product_width (int): Chiều rộng của sản phẩm.
        product_height (int): Chiều cao của sản phẩm.
    """
    for row in range(y, y + product_height):
        for col in range(x, x + product_width):
            stock[row, col] = 1  # Đánh dấu vùng này là đã sử dụng

class Policy2353281thuhai:

    def sort_stocks_by_remaining_area(self, stocks):
        return sorted(enumerate(stocks), key=lambda x: np.sum(x[1] == -1))

    def get_action(self, observation, info=None):
        stocks = observation["stocks"]
        products = observation["products"]

        for product in products:
            if product["quantity"] <= 0:
                continue

            product_width, product_height = product["size"]
            sorted_stocks = self.sort_stocks_by_remaining_area(stocks)

            for stock_idx, stock in sorted_stocks:
                stock_width, stock_height = stock.shape
                remaining_area = np.sum(stock == -1)

                if product_width * product_height > remaining_area:
                    continue

                for y in range(stock_height - product_height + 1):
                    for x in range(stock_width - product_width + 1):
                        if is_position_available(stock, x, y, product_width,
                                                 product_height):
                            place_product_in_stock(stock, x, y, product_width,
                                                   product_height)
                            product["quantity"] -= 1
                            print(
                                f"stock_idx: {stock_idx}  size {product_width}x{product_height} quantity: {product['quantity']}"
                            )
                            return {
                                "stock_idx": stock_idx,
                                "size": [product_width, product_height],
                                "position": (x, y)
                            }

        print("No valid action found.")
        return {"stock_idx": None, "size": None, "position": None}
        
class Policy2353281(Policy):

    def __init__(self):
        # Student code here
        self.idx_stock = []
        self._stocks = []
        self.num_stocks = 0
        self.list_prod = []
        self.list_action = []
        self.num_prod = [0]
        self.prod_area_left = 0
        self.prod_area = 0

    def ___stock_area___(self, stock):
        x, y = self._get_stock_size_(stock)
        return x * y

    def ___csize___(self, stock):
        return np.max(np.sum(np.any(stock > -1, axis=1))), np.max(
            np.sum(np.any(stock > -1, axis=0)))

    def get_action(self, observation, info):
        if (info["filled_ratio"] == 0):
            self.__init__()
            self._stocks = deepcopy(observation["stocks"])
            self.num_stocks = deepcopy(len(self._stocks))
            for prod in observation["products"]:
                self.prod_area_left += np.prod(prod["size"]) * prod["quantity"]
            self.idx_stock = sorted(enumerate(observation["stocks"]),
                                    key=lambda x: self.___stock_area___(x[1]),
                                    reverse=True)
            self.list_prod = sorted(
                enumerate(deepcopy(observation["products"])),
                key=lambda x: x[1]["size"][0] * x[1]["size"][1],
                reverse=True)
            self.render_action()
            self.idx_stock.reverse()
            self.num_prod.reverse()
            end = np.sum(self.num_prod)
            idx = 0
            for d in self.num_prod:
                start = end - d
                x, y = self.list_action[end - 1]["size"]
                while True:
                    i, stock = self.idx_stock[idx]
                    w, h = self._get_stock_size_(stock)
                    if x <= w and y <= h:
                        for j in range(start, end):
                            self.list_action[j]["action"]["stock_idx"] = i
                        end -= d
                        idx += 1
                        break
                    idx += 1
            self.list_action.reverse()
        return self.list_action.pop()["action"]

    def place_prod(self, stock, prod):
        pos_x, pos_y = None, None
        minx, miny = 0, 0
        cx, cy = self.___csize___(stock)
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_size = prod["size"]
        prod_w, prod_h = prod_size
        if stock_w < prod_w or stock_h < prod_h:
            return pos_x, pos_y, minx, miny
        if self.prod_area < stock_h * stock_w * 0.75:
            min_S = -1
            for x in range(stock_w - prod_w + 1):
                if x > cx: continue
                for y in range(stock_h - prod_h + 1):
                    if y > cy: continue
                    if self._can_place_(stock, (x, y), prod_size):
                        mx = max(cx, x + prod_w)
                        my = max(cy, y + prod_h)
                        new_S = mx * my
                        f = stock_w / stock_h
                        if (new_S < min_S or min_S == -1) and (
                            (cx == 0 and cy == 0) or
                            (mx / my <= cx / cy and cx / cy >= f) or
                            (mx / my >= cx / cy and cx / cy <= f)) and (
                                x == 0 or stock[x - 1][y]
                                >= 0) and (y == 0 or stock[x][y - 1] >= 0):
                            pos_x, pos_y, minx, miny, min_S = x, y, mx, my, new_S
            if pos_x is not None: return pos_x, pos_y, minx, miny
        min_S = -1
        for x in range(stock_w - prod_w + 1):
            if x > cx: continue
            for y in range(stock_h - prod_h + 1):
                if y > cy: continue
                if self._can_place_(stock, (x, y), prod_size):
                    mx = max(cx, x + prod_w)
                    my = max(cy, y + prod_h)
                    new_S = mx * my
                    if (new_S < min_S or min_S == -1):
                        pos_x, pos_y, minx, miny, min_S = x, y, mx, my, new_S
        return pos_x, pos_y, minx, miny

    def render_action(self):
        end = True
        for i, _ in self.idx_stock:
            stock_idx = i
            stock = self._stocks[i]
            self.prod_area = self.prod_area_left
            for j, prod in self.list_prod:
                while prod["quantity"] > 0:
                    prod_size = prod["size"]
                    pos_x, pos_y, mx, my = self.place_prod(stock, prod)
                    if pos_x is not None:
                        self.num_prod[-1] += 1
                        action = {
                            "size": (mx, my),
                            "action": {
                                "stock_idx": stock_idx,
                                "size": prod_size,
                                "position": (pos_x, pos_y)
                            }
                        }
                        self.list_action.append(action)
                        stock[pos_x:pos_x + prod_size[0],
                              pos_y:pos_y + prod_size[1]] = j + 1
                        prod["quantity"] -= 1
                        self.prod_area_left -= np.prod(prod_size)
                        end = all([
                            product[1]["quantity"] == 0
                            for product in self.list_prod
                        ])
                        if end: break
                    else: break
            if end: break
            self.num_prod.append(0)


class BFPolicy(Policy):

    def __init__(self):
        self.c_stock = 0
        self.idx_stock = []
        self.list_prod = []
        self.prod_area_left = 0
        self.prod_area = 0
        self.m = True

    def ___stock_area___(self, stock):
        x, y = self._get_stock_size_(stock)
        return x * y

    def ___csize___(self, stock):
        return np.max(np.sum(np.any(stock > -1, axis=1))), np.max(
            np.sum(np.any(stock > -1, axis=0)))

    def get_action(self, observation, info):
        if (info["filled_ratio"] == 0):
            self.__init__()
            for prod in observation["products"]:
                self.prod_area_left += np.prod(prod["size"]) * prod["quantity"]
            self.prod_area = self.prod_area_left
            self.idx_stock = sorted(enumerate(observation["stocks"]),
                                    key=lambda x: self.___stock_area___(x[1]),
                                    reverse=True)
            self.list_prod = sorted(observation["products"],
                                    key=lambda x: x["size"][0] * x["size"][1],
                                    reverse=True)
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = None, None
        c_stock = -1
        for i, _ in self.idx_stock:
            c_stock += 1
            if c_stock < self.c_stock: continue
            stock = observation["stocks"][i]
            stock_w, stock_h = self._get_stock_size_(stock)
            if self.prod_area < stock_h * stock_w * 0.75 and c_stock < len(
                    self.idx_stock) - 1 and self.m:
                self.c_stock += 1
                continue
            cx, cy = self.___csize___(stock)
            # c_prod=-1

            min_dS = -1
            min_size = [0, 0]
            for prod in self.list_prod:
                # c_prod+=1
                # if c_prod<self.c_prod: continue
                if prod["quantity"] > 0:
                    min_size = prod["size"]
                    prod_w, prod_h = min_size
                    if stock_w < prod_w or stock_h < prod_h: continue
                    for x in range(stock_w - prod_w + 1):
                        if x > cx: continue
                        for y in range(stock_h - prod_h + 1):
                            if y > cy: continue
                            if self._can_place_(stock, (x, y), min_size):
                                new_S = max(cy, y + prod_h) * max(
                                    cx, x + prod_w)
                                dS = (new_S - cx * cy) / (prod_h * prod_w)
                                if dS < min_dS or min_dS == -1:
                                    prod_size = min_size
                                    pos_x = x
                                    pos_y = y
                                    min_dS = dS
            if pos_x is not None:
                stock_idx = i
                self.prod_area_left -= np.prod(prod_size)
                break
            if self.c_stock == len(self.idx_stock) - 1:
                self.c_stock -= 2
                self.m = False
            self.c_stock += 1
            self.prod_area = self.prod_area_left
            # self.c_prod=0
        return {
            "stock_idx": stock_idx,
            "size": prod_size,
            "position": (pos_x, pos_y)
        }


class FFPolicy(Policy):

    def __init__(self):
        self.c_prod = 0
        self.c_stock = 0
        self.idx_stock = []
        self.list_prod = []
        self.prod_area_left = 0
        self.m = True

    def ___stock_area___(self, stock):
        x, y = self._get_stock_size_(stock)
        return x * y

    def ___csize___(self, stock):
        return np.max(np.sum(np.any(stock > -1, axis=1))), np.max(
            np.sum(np.any(stock > -1, axis=0)))

    def get_action(self, observation, info):
        if (info["filled_ratio"] == 0):
            self.__init__()
            for prod in observation["products"]:
                self.prod_area_left += np.prod(prod["size"]) * prod["quantity"]
            self.prod_area = self.prod_area_left
            self.idx_stock = sorted(enumerate(observation["stocks"]),
                                    key=lambda x: self.___stock_area___(x[1]),
                                    reverse=True)
            self.list_prod = sorted(observation["products"],
                                    key=lambda x: x["size"][0] * x["size"][1],
                                    reverse=True)
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = None, None
        c_stock = -1
        for i, _ in self.idx_stock:
            c_stock += 1
            if c_stock < self.c_stock: continue
            stock = observation["stocks"][i]
            stock_w, stock_h = self._get_stock_size_(stock)
            if self.prod_area < stock_h * stock_w * 0.75 and c_stock < len(
                    self.idx_stock) - 1 and self.m:
                self.c_stock += 1
                continue
            c_prod = -1
            cx, cy = self.___csize___(stock)
            for prod in self.list_prod:
                c_prod += 1
                if c_prod < self.c_prod: continue
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size
                    if stock_w < prod_w or stock_h < prod_h: continue
                    for x in range(stock_w - prod_w + 1):
                        if x > cx: continue
                        for y in range(stock_h - prod_h + 1):
                            if y > cy: continue
                            if self._can_place_(stock, (x, y), prod_size):
                                pos_x = x
                                pos_y = y
                                break
                        if pos_x is not None:
                            break
                    if pos_x is not None:
                        break
                self.c_prod += 1
            if pos_x is not None:
                stock_idx = i
                self.prod_area_left -= np.prod(prod_size)
                break
            if self.c_stock == len(self.idx_stock) - 1:
                self.c_stock -= 2
                self.m = False
            self.c_stock += 1
            self.c_prod = 0
            self.prod_area = self.prod_area_left
        return {
            "stock_idx": stock_idx,
            "size": prod_size,
            "position": (pos_x, pos_y)
        }


def evaluate_algorithm(algorithm, *args, **kwargs):

    # memory usage
    tracemalloc.start()

    # time measure
    start_time = time.perf_counter()
    result = algorithm(*args, **kwargs)
    end_time = time.perf_counter()

    # stop
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # dict to store output
    evaluation = {
        "runtime_seconds": end_time - start_time,
        "current_memory_bytes": current,
        "peak_memory_bytes": peak,
        "result": result
    }

    return evaluation
