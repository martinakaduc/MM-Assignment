import random
import numpy as np
from scipy.optimize import linprog
from policy import Policy

class Policy2310373_2013452_2311958_2312137_2313045(Policy):
    def __init__(self, policy_id = 1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

        # Parameters initialization
        if policy_id == 1:
            # GA parameters
            self.policy_id = policy_id
            self.population_size = 50
            self.generations = 20
            self.mutation_rate = 0.5
        elif policy_id == 2:
            # CG parameters
            pass

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.get_action_1(observation, info)
        elif self.policy_id == 2:
            return self.get_action_2(observation, info)
        return None
    
    ## get_action for policy 1
    def get_action_1(self, observation, info):
        # Initialize results
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        
        # Pick a product that has quantity > 0
        products = observation["products"]
        product_to_place = None
        for prod in products:
            if prod["quantity"] > 0:
                product_to_place = prod
                break
        
        # Algorithm
        if product_to_place != None:
            prod_size = product_to_place["size"]
            
            # Initialize population
            population = self.initialize_population(observation, prod_size)
            
            # Run GA for a number of generations
            for _ in range(self.generations):
                # Fitness score
                scores = [self.fitness(chrom, observation) for chrom in population]
                
                # Select
                population = self.select(scores, population)
                
                # Crossover
                offspring = []
                while len(population) + len(offspring) < self.population_size:
                    parent1, parent2 = random.sample(population, 2)
                    child = self.crossover(parent1, parent2)
                    # Mutate
                    child = self.mutate(child, observation)
                    offspring.append(child)
                    
                population += offspring
                
            # Pick the best solution after GA finishes
            scores = [self.fitness(chrom, observation) for chrom in population]
            best_idx = np.argmax(scores)
            res = population[best_idx]
            
            # Get the stock index and product position
            stock_idx, prod_size, (pos_x, pos_y) = res
        
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    ## Support function for policy 1
    # Check if the stock is unfilled
    def is_unfilled(self, stock):
        return np.all(stock >= 0)
    
    # Count number of unfilled stocks
    def num_unfilled(self, stocks):
        return sum(self.is_unfilled(s) for s in stocks)
    
    # Initialize population 
    def initialize_population(self, observation, prod_size):
        population = []
        stocks = observation["stocks"]
        attempts = self.population_size * 10 # Try more to ensure feasibility
        
        for _ in range(attempts):
            # Random choose a stock
            stock_idx = random.randint(0, len(stocks) - 1)
            stock = stocks[stock_idx]
            
            # Random choose a position
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size
            
            if stock_w >= prod_w and stock_h >= prod_h:
                pos_x = random.randint(0, stock_w - prod_w)
                pos_y = random.randint(0, stock_h - prod_h)
                if self._can_place_(stock, (pos_x, pos_y), prod_size):
                    population.append((stock_idx, prod_size, (pos_x, pos_y)))
                    if len(population) == self.population_size:
                        break
            if stock_w >= prod_h and stock_h >= prod_w:
                pos_x = random.randint(0, stock_w - prod_h)
                pos_y = random.randint(0, stock_h - prod_w)
                if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                    population.append((stock_idx, prod_size[::-1], (pos_x, pos_y)))
                    if len(population) == self.population_size:
                        break
        
        # If not enough feasible solution, fill with dummy (infeasible) solutions
        while len(population) < self.population_size:
            population.append((-1, [0, 0], (0, 0)))
        
        return population
    
    # Fitness score
    def fitness(self, chrom, observation):
        stock_idx, prod_size, (pos_x, pos_y) = chrom
        
        # Dummy
        if stock_idx < 0:
            return 0
        
        # Check feasibility
        stocks = observation["stocks"]
        stock = stocks[stock_idx]
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        if stock_idx >= len(stocks):
            return 0
        if pos_x + prod_w > stock_w or pos_y + prod_h > stock_h:
            return 0
        if not self._can_place_(stock, (pos_x, pos_y), prod_size):
            return 0
        
        # Number of unfilled stock before placement
        unfilled_before = self.num_unfilled(stocks)
        unfilled_after = unfilled_before - 1 if self.is_unfilled(stock) else unfilled_before
        
        # Logistic params
        midpoint = len(stocks) // 2
        steepness = 1.0
        penalty = 0.5
        
        # Compute fitness score
        score = 1.0 / (1.0 + np.exp(-steepness * (unfilled_after - midpoint)))
        
        # Add penalty
        if self.is_unfilled(stock):
            score *= penalty
            
        return score
    
    # Select
    def select(self, scores, population):
        # Sort population by fitness score descending
        sorted_pop = [p for _, p in sorted(zip(scores, population), key = lambda x: x[0], reverse = True)]
        
        # Keep top half (at least 2)
        return sorted_pop[: max(2, self.population_size // 2)]
    
    # Crossover
    def crossover(self, parent1, parent2):
        stock_idx1, prod_size1, position1 = parent1
        stock_idx2, prod_size2, position2 = parent2
        
        # Randomly choose stock_idx from one parent and position from another
        if random.random() < 0.5:
            child = (stock_idx1, prod_size2, position2)
        else:
            child = (stock_idx2, prod_size1, position1)
        
        # If the child is not feasible, we might return it anyway and rely on mutation or selection
        return child
    
    def mutate(self, chrom, observation):
        # Mutations occur only at a certain rate
        if random.random() <= self.mutation_rate:
            stock_idx, prod_size, (pos_x, pos_y) = chrom
            stocks = observation["stocks"]
            if len(stocks) == 0:
                return chrom
            
            # Try changing stock or position
            if random.random() < 0.5:
                stock_idx = random.randint(0, len(stocks) - 1)
                
            stock = stocks[stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size
            
            if stock_w >= prod_w and stock_h >= prod_h:
                pos_x = random.randint(0, stock_w - prod_w)
                pos_y = random.randint(0, stock_h - prod_h)
                chrom = (stock_idx, prod_size, (pos_x, pos_y))
            else:
                pos_x = random.randint(0, stock_w - prod_h)
                pos_y = random.randint(0, stock_h - prod_w)
                chrom =  (stock_idx, prod_size[::-1], (pos_x, pos_y))
            
        return chrom
    
    ## get_action for policy 2 
    def get_action_2(self, observation, info):
        patterns, _ = self._column_generation(observation)
        stocks = observation["stocks"]
        action = None

        for stock_idx, pattern in patterns:
            stock = stocks[stock_idx]
            
            for prod_idx, count in enumerate(pattern):
                while count > 0:  
                    prod_size = observation["products"][prod_idx]["size"]
                    placed = False
                    for x in range(stock.shape[0] - prod_size[0] + 1):
                        for y in range(stock.shape[1] - prod_size[1] + 1):
                            position = (x, y)
                            if self._can_place_(stock, position, prod_size, allow_rotation=True):
                                count -= 1  
                                placed = True
                                action = {
                                    "stock_idx": stock_idx,
                                    "size": prod_size,
                                    "position": position
                                }
                                break  
                        if placed:
                            break  
                    if not placed:
                        break  
            if action is not None:
                return action
    
    ## Support function for policy 2
    def cutting_generation(self, products, stocks):
        patterns = []
        for stock_idx, (stock_w, stock_h) in enumerate(stocks):
            for prod_idx, product in enumerate(products):
                prod_w, prod_h = product["size"]
                if prod_w <= stock_w and prod_h <= stock_h:
                    max_x = stock_w // prod_w
                    max_y = stock_h // prod_h
                    max_x_rot = stock_w // prod_h
                    max_y_rot = stock_h // prod_w
                    max_count = max_x * max_y
                    max_count_rot = max_x_rot * max_y_rot
                    if max_count < max_count_rot:
                        max_count = max_count_rot
                    max_count = min(max_count, product["quantity"])
                    if max_count > 0:
                        pattern = [0] * len(products)
                        pattern[prod_idx] = max_count
                        patterns.append((stock_idx, pattern))
        return patterns
    
    def master_problem(self, patterns, demands):
        """
        Hàm giải bài toán master problem bằng `linprog`.
        """
        if not patterns:
            return None, float("inf")
        
        num_patterns = len(patterns)
        num_products = len(demands)

        c = np.ones(num_patterns)  

        # Ma trận ràng buộc A và vector vế phải b
        A_eq = np.zeros((num_products, num_patterns))
        for j, (_, pattern) in enumerate(patterns):
            for i in range(num_products):
                A_eq[i, j] = pattern[i]
        b_eq = np.array(demands)

        # Sử dụng `linprog` 
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='simplex')
        if result.success:
            return result.x, result.fun  
        else:
            return None, float("inf")
        
    def _column_generation(self, observation):
        """
        Thực hiện column generation cho bài toán cutting stock.
        """
        # Lấy thông tin sản phẩm và kho
        products = observation["products"]
        stocks = [self._get_stock_size_(stock) for stock in observation["stocks"]]
        patterns = []
        demands = [product["quantity"] for product in products]
        # Khởi tạo các pattern ban đầu (identity matrix cho mỗi sản phẩm)
        for prod_idx, product in enumerate(products):
            if product["quantity"] > 0:
                pattern = [1 if idx == prod_idx else 0 for idx, _ in enumerate(products)]
                patterns.append((0, pattern))  # Chi phí 0 cho các pattern ban đầu

        # Lặp để tìm nghiệm tối ưu
        while True:
            # Giải bài toán master problem
            solution, objective_value = self.master_problem(patterns, demands)
            if solution is None:
                break

            # Tạo các pattern mới từ hàm `cutting_generation`
            new_patterns = self.cutting_generation(products, stocks)
            new_patterns = [p for p in new_patterns if p not in patterns]

            # Nếu không tạo được pattern mới, dừng lặp
            if not new_patterns:
                break

            # Thêm các pattern mới vào danh sách pattern
            patterns.extend(new_patterns)

        return patterns, objective_value
    
    def _can_place_(self, stock, position, prod_size, allow_rotation=False):
        """
        Kiểm tra xem sản phẩm có thể được đặt tại vị trí trên tấm vật liệu không, 
        ưu tiên xoay sản phẩm nếu cắt được nhiều sản phẩm hơn.
        """
        pos_x, pos_y = position
        prod_w, prod_h = prod_size
        stock_h, stock_w = stock.shape

        # Kiểm tra không xoay
        can_place_no_rotation = (
            pos_x + prod_w <= stock_h and
            pos_y + prod_h <= stock_w and
            np.all(stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] == -1)
        )

        # Nếu không cho phép xoay, chỉ kiểm tra trường hợp không xoay
        if not allow_rotation:
            return can_place_no_rotation

        # Kiểm tra xoay
        can_place_rotation = (
            pos_x + prod_h <= stock_h and
            pos_y + prod_w <= stock_w and
            np.all(stock[pos_x:pos_x + prod_h, pos_y:pos_y + prod_w] == -1)
        )

        # Tính số lượng sản phẩm có thể cắt được
        count_no_rotation = (stock_h // prod_h) * (stock_w // prod_w) if can_place_no_rotation else 0
        count_rotation = (stock_h // prod_w) * (stock_w // prod_h) if can_place_rotation else 0

        # So sánh 
        if count_rotation > count_no_rotation and can_place_rotation:
            prod_size[0], prod_size[1] = prod_h, prod_w  # Cập nhật kích thước sản phẩm khi xoay
            return True

        return can_place_no_rotation