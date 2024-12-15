from policy import Policy
import random
import numpy as np
from scipy.optimize import linprog



class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        self.policy_id = policy_id
        if policy_id == 1:
            self.population_size = 50  
            self.mutation_rate = 0.01
            self.generations = 30
            self.output = []

            
        elif policy_id == 2:
            pass

            
    def initial_population(self, observation):
        """
        Khởi tạo quần thể:
        1. Sắp xếp sản phẩm theo kích thước giảm dần.
        2. Chọn một stock ngẫu nhiên.
        3. Lấp đầy càng nhiều sản phẩm càng tốt vào stock đó, bao gồm cả thử xoay sản phẩm.
        4. Chuyển sang stock ngẫu nhiên khác khi không thể lấp đầy thêm.
        """
        population = []
        count = 0

        while count < self.population_size:
            obs_copy = {
                "products": [prod.copy() for prod in observation["products"]],
                "stocks": [stock.copy() for stock in observation["stocks"]]
            }
            individual = []  
            list_products = obs_copy["products"]
            stocks = obs_copy["stocks"]

            sorted_products = sorted(
                list_products, 
                key=lambda p: p["size"][0] * p["size"][1], 
                reverse=True
            )

            stock_indices = list(range(len(stocks)))
            random.shuffle(stock_indices)

            for stock_idx in stock_indices:
                stock = stocks[stock_idx]
                stock_w, stock_h = self._get_stock_size_(stock)

                for product in sorted_products:
                    quantity = product["quantity"]
                    size = product["size"]

                    while quantity > 0:
                        placed = False

                        for x in range(stock_w - size[0] + 1):
                            for y in range(stock_h - size[1] + 1):
                                if self._can_place_(stock, (x, y), size):
                                    action = {
                                        "stock_idx": stock_idx,
                                        "size": size,
                                        "position": (x, y),
                                    }
                                    individual.append(action)
                                    stock[x:x + size[0], y:y + size[1]] = 1
                                    quantity -= 1
                                    product["quantity"] = quantity
                                    placed = True
                                    break
                            if placed:
                                break

                        if not placed:
                            rotated_size = size[::-1] 
                            for x in range(stock_w - rotated_size[0] + 1):
                                for y in range(stock_h - rotated_size[1] + 1):
                                    if self._can_place_(stock, (x, y), rotated_size):
                                        action = {
                                            "stock_idx": stock_idx,
                                            "size": rotated_size,
                                            "position": (x, y),
                                        }
                                        individual.append(action)
                                        stock[x:x + rotated_size[0], y:y + rotated_size[1]] = 1
                                        quantity -= 1
                                        product["quantity"] = quantity
                                        placed = True
                                        break
                            if placed:
                                break

                        if not placed:
                            break

                if all(prod["quantity"] == 0 for prod in sorted_products):
                    break

            if self.check_individual_validity(individual, observation):
                population.append(individual)
                count += 1

        return population

    def fitness(self, individual, observation):
        """
        Tính toán fitness của một cá thể:
        - Tối ưu hóa diện tích lấp đầy trong các stock được sử dụng.
        - Công thức fitness:
            fitness = (tổng diện tích lấp đầy) / (tổng diện tích có sẵn trong các stock được sử dụng).
        """
        obs_copy = {
            "products": [prod.copy() for prod in observation["products"]],
            "stocks": [stock.copy() for stock in observation["stocks"]]
        }

        total_filled_area = 0  
        total_available_area = 0  
        stock_usage = [np.zeros_like(stock, dtype=int) for stock in obs_copy["stocks"]]  # Đánh dấu vùng đã sử dụng

        for action in individual:
            stock_idx = action["stock_idx"]
            size = action["size"]
            position = action["position"]

            x, y = position
            w, h = size

            stock_usage[stock_idx][x:x + w, y:y + h] = 1
            total_filled_area += w * h  

        for stock, stock_used in zip(obs_copy["stocks"], stock_usage):
            if np.sum(stock_used) > 0:  
                stock_w, stock_h = self._get_stock_size_(stock)
                total_available_area += stock_w * stock_h

        if total_available_area > 0:
            fitness = total_filled_area / total_available_area
        else:
            fitness = 0  

        return fitness

    def selection(self, population, fitness_values, num_selected):
        """
        Chọn lọc các cá thể tốt nhất bằng Tournament Selection.
        """
        selected = []
        for _ in range(num_selected):
            tournament = random.sample(list(zip(population, fitness_values)), k=3)
            best = max(tournament, key=lambda x: x[1])[0]
            selected.append(best)
        return selected

    def crossover(self, parent1, parent2):
        """
        Lai ghép hai cá thể để tạo ra hai cá thể con.
        """
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        
        offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
        offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
        return offspring1, offspring2

    def mutation(self, individual, observation):
        """
        Đột biến: Thay đổi một hành động trong cá thể, bao gồm thử xoay sản phẩm.
        """
        if random.random() < self.mutation_rate:
            mutation_point = random.randint(0, len(individual) - 1)
            action = individual[mutation_point]
            stock_idx = action["stock_idx"]
            size = action["size"]

            stock = observation["stocks"][stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)

            for _ in range(10):
                x = random.randint(0, stock_w - size[0])
                y = random.randint(0, stock_h - size[1])
                if self._can_place_(stock, (x, y), size):
                    action["position"] = (x, y)
                    break

                rotated_size = size[::-1]
                if self._can_place_(stock, (x, y), rotated_size):
                    action["position"] = (x, y)
                    action["size"] = rotated_size
                    break
        return individual

    def check_individual_validity(self, individual, observation):
        """
        Kiểm tra tính hợp lệ của một cá thể.
        """
        obs_copy = {
            "products": [prod.copy() for prod in observation["products"]],
            "stocks": [stock.copy() for stock in observation["stocks"]]
        }

        for action in individual:
            stock_idx = action["stock_idx"]
            size = action["size"]
            position = action["position"]
            x, y = position

            if not self._can_place_(obs_copy["stocks"][stock_idx], (x, y), size):
                return False
            obs_copy["stocks"][stock_idx][x:x + size[0], y:y + size[1]] = 1

        return True

    def evolve(self, observation):
        """
        Tiến hóa để tìm giải pháp tốt nhất. Ghi nhớ kết quả sau khi tiến hóa.
        """

        self.population = self.initial_population(observation)
        best_individual = None
        best_fitness = -float("inf")

        for generation in range(self.generations):
            fitness_values = [self.fitness(individual, observation) for individual in self.population]

            max_fitness = max(fitness_values)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = self.population[fitness_values.index(max_fitness)]

            
            if generation > 5 and max_fitness == best_fitness:
                break
            
            selected_population = self.selection(self.population, fitness_values, self.population_size // 2)

            new_population = []
            for i in range(0, len(selected_population), 2):
                if i + 1 < len(selected_population):
                    offspring1, offspring2 = self.crossover(selected_population[i], selected_population[i + 1])
                    new_population.extend([offspring1, offspring2])
            
            mutated_population = [self.mutation(ind, observation) for ind in new_population]
            mutated_population = [ind for ind in mutated_population if ind is not None]

            while len(mutated_population) < self.population_size:
                mutated_population.append(random.choice(selected_population))
            self.population = mutated_population[:self.population_size]


        self.output = best_individual
        
    def get_action(self, observation, info):
        if self.policy_id == 1:
            if not self.output:
                self.evolve(observation)  

            if self.output:
                action = self.output.pop(0)
                return {
                    "stock_idx": action["stock_idx"],
                    "size": action["size"],
                    "position": action["position"],
                }
        elif self.policy_id == 2:
            products = observation["products"]
            stocks = observation["stocks"]                       

            product_sizes = [prod["size"] for prod in products]    
            product_quantities = [prod["quantity"] for prod in products] 
            stock_shape = stocks[0].shape                          

            patterns = self.initialize_patterns(products, stock_shape) 
            if patterns.size == 0:                                    
                pattern = np.zeros(len(products), dtype=int)         
                for i, p in enumerate(products):                     
                    if p["size"][0] <= stock_shape[0] and p["size"][1] <= stock_shape[1]:
                        # [12] Nếu sản phẩm i vừa với tấm
                        pattern[i] = 1                                 
                        break                                      
                patterns = pattern.reshape(-1, 1)                      

            costs = np.ones(patterns.shape[1])                       

            max_iterations = 300                                     
            no_improvement_limit = 50                               
            no_improvement_count = 0                                  
            best_solution = None                                      
            best_patterns = None                                     
            best_obj = np.inf                                         

            for iteration in range(max_iterations):                    
                res = linprog(                                          
                    c=costs,
                    A_eq=patterns,
                    b_eq=product_quantities,
                    method='highs',
                    bounds=(0, None)
                )

                if not res.success:                                    
                    break                                            

                current_obj = res.fun                                  
                if current_obj < best_obj:                           
                    best_obj = current_obj                              
                    best_solution = res.x                               
                    best_patterns = patterns.copy()                    
                    no_improvement_count = 0                           
                else:
                    no_improvement_count += 1                          

                dual_prices = res.eqlin.marginals                       
                if dual_prices is None:                                
                    break                                               

                # [38] Giải bài toán con: tìm pattern mới bằng knapsack 2D dựa trên dual_prices
                new_pattern, reduced_cost = self.solve_knapsack_dp(product_sizes, dual_prices, stock_shape)

                if reduced_cost >= -1e-6:                              
                    if no_improvement_count >= no_improvement_limit:   
                        break                                          
                else:
                    # [42] Nếu tìm được pattern mới với reduced_cost < 0, thêm pattern này vào RMP
                    patterns = np.column_stack((patterns, new_pattern)) 
                    costs = np.append(costs, 1)                       

            # [45] Sau khi kết thúc vòng lặp (do không cải thiện hoặc đủ pattern), lấy nghiệm tốt nhất
            if best_solution is not None and len(best_solution) > 0:    
                solution = best_solution.round().astype(int)           
                chosen_pattern_idx = np.argmax(solution)                
                pattern = best_patterns[:, chosen_pattern_idx]          
                action = self.construct_action(products, stocks, pattern)
                if action is not None:                                 
                    return action                                       

            # [53] Nếu không tìm được hành động tốt hơn, trả về hành động mặc định
            return {"stock_idx": 0, "size": products[0]["size"], "position": (0, 0)}

    def solve_knapsack_dp(self, product_sizes, dual_prices, stock_shape):
        # [54] Bài toán con: knapsack 2D để tìm pattern mới
        stock_w, stock_h = stock_shape                            
        num_products = len(product_sizes)                          

        dp = np.zeros((stock_w + 1, stock_h + 1))                 
        trace = [[[] for _ in range(stock_h + 1)] for _ in range(stock_w + 1)]

        for i, (w, h) in enumerate(product_sizes):                
            value = dual_prices[i]                                 
            if w <= stock_w and h <= stock_h:                     
                for W in range(stock_w, w-1, -1):                 
                    for H in range(stock_h, h-1, -1):              
                        # [64] Kiểm tra nếu đặt sản phẩm i vào vùng (W,H) cải thiện dp không
                        if dp[W - w][H - h] + value > dp[W][H]:
                            dp[W][H] = dp[W - w][H - h] + value     
                            trace[W][H] = trace[W - w][H - h] + [i] 

        # [67] Tìm nghiệm tốt nhất từ dp
        best_val = 0
        best_w = 0
        best_h = 0
        for W in range(stock_w + 1):                           
            for H in range(stock_h + 1):
                if dp[W][H] > best_val:
                    best_val = dp[W][H]
                    best_w = W
                    best_h = H

        selected_products = trace[best_w][best_h]                  
        pattern = np.zeros(num_products, dtype=int)                
        for idx in selected_products:                             
            pattern[idx] += 1

        reduced_cost = 1 - sum(dual_prices[i]*pattern[i] for i in range(num_products)) # [72] Tính reduced cost = 1 - ∑(dual_prices[i]*pattern[i])
        return pattern, reduced_cost                                # [73] Trả về pattern mới và reduced_cost

    def construct_action(self, products, stocks, pattern):
        # [74] Từ pattern, chọn ra sản phẩm và đặt lên tấm
        for stock_idx, stock in enumerate(stocks):                 
            for prod_idx, quantity in enumerate(pattern):           
                if quantity > 0:                                   
                    product_size = products[prod_idx]["size"]      
                    pos_x, pos_y = self.find_position(stock, product_size) 
                    if pos_x is not None:                         
                        return {
                            "stock_idx": stock_idx,
                            "size": product_size,
                            "position": (pos_x, pos_y),
                        }                                          
        return None                                               

    def find_position(self, stock, product_size):
        # [83] Tìm vị trí trống trên tấm để đặt sản phẩm
        stock_w, stock_h = stock.shape                             
        prod_w, prod_h = product_size                              

        for x in range(stock_w - prod_w + 1):                      
            for y in range(stock_h - prod_h + 1):                   
                if np.all(stock[x:x+prod_w, y:y+prod_h] == -1):    
                    return x, y                                     
        return None, None                                          

    def initialize_patterns(self, products, stock_shape):
        # [91] Khởi tạo một tập pattern ban đầu
        stock_w, stock_h = stock_shape                           
        patterns = []                                             

        # [94] Tạo pattern đơn cho mỗi sản phẩm (chỉ chứa 1 loại sản phẩm)
        for i, prod in enumerate(products):
            w, h = prod["size"]                                     
            q = prod["quantity"]                                    
            if w <= stock_w and h <= stock_h:                       
                pattern = np.zeros(len(products), dtype=int)       
                max_fit_w = stock_w // w                           
                max_fit_h = stock_h // h                            
                max_fit = max_fit_w * max_fit_h                    
                used = min(q, max_fit)                              
                if used > 0:
                    pattern[i] = used                              
                    patterns.append(pattern)                        

        # [105] Nếu ít pattern quá, tạo pattern hỗn hợp
        if len(patterns) < 2:
            mixed = self.create_mixed_pattern(products, stock_shape) 
            if mixed is not None:
                patterns.append(mixed)                             

        if len(patterns) == 0:                                     
            return np.array([])                                    

        return np.array(patterns).T                               

    def create_mixed_pattern(self, products, stock_shape):
        # [111] Tạo một pattern hỗn hợp gồm nhiều loại sản phẩm
        stock_w, stock_h = stock_shape                            
        space = np.full((stock_w, stock_h), -1)                     
        pattern = np.zeros(len(products), dtype=int)               

        # [115] Sắp xếp sản phẩm theo diện tích giảm dần để đặt những sản phẩm lớn trước
        sorted_prods = sorted(enumerate(products),
                              key=lambda x: x[1]["size"][0]*x[1]["size"][1],
                              reverse=True)
        for i, prod in sorted_prods:                                
            w, h = prod["size"]                                   
            pos = self.find_position(space, (w, h))               
            if pos is not None and pos[0] is not None and pos[1] is not None:
                x, y = pos
                space[x:x+w, y:y+h] = i
                pattern[i] += 1
                                  

        if np.any(pattern > 0):                                     
            return pattern                                         
        return None                                              
