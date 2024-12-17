import random
from policy import Policy


class Policy2012741_2012500_2213379_2353213_2214062(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.best_fit_decreasing(observation, info)
        elif self.policy_id == 2:
            return self.genetic_algorithm(observation, info)
        
    def best_fit_decreasing(self, observation, info):
        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        
        # Sắp xếp sản phẩm theo diện tích giảm dần
        list_sorted_products = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True)

        # Duyệt qua từng sản phẩm
        for prod in list_sorted_products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                min_waste = float('inf') 

                # Tìm tấm tốt nhất để đặt sản phẩm
                for idx, stock in enumerate(observation["stocks"]):
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # Kiểm tra nếu kích thước tấm không đủ chứa sản phẩm
                    if (stock_w < prod_w or stock_h < prod_h) and (stock_w < prod_h or stock_h < prod_w):
                        continue
                    # Xoay sản phẩm nếu không vừa tấm
                    if (stock_w < prod_w or stock_h < prod_h):  
                        prod_w, prod_h = prod_h, prod_w

                    feasible = 0 # Biến để kiểm tra xem đã tìm được vị trí đặt hay chưa
                    # Duyệt qua tất cả các vị trí trên tấm
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            # Nếu tìm được vị trí phù hợp
                            if self._can_place_(stock, (x, y), prod_size):
                                # Tính lượng không gian dư thừa khi đặt prod
                                remaining_space = self.remaining_space(stock) - prod_w * prod_h
                                if remaining_space < min_waste:
                                    min_waste = remaining_space
                                    stock_idx = idx
                                    pos_x, pos_y = x, y
                                feasible = 1
                                break  
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                # Tính lượng không gian dư thừa
                                remaining_space = self.remaining_space(stock) - prod_w * prod_h
                                if remaining_space < min_waste:
                                    min_waste = remaining_space
                                    stock_idx = idx
                                    pos_x, pos_y = x, y
                                    prod_size = prod_size[::-1]
                                feasible = 1
                                break  
                        if feasible == 1:
                            break
                if stock_idx >= 0:
                    break                
       
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    # Hàm tính diện tích phần dư của tấm 
    def remaining_space(self, stock):
        # Tính diện tích của tấm
        stock_w, stock_h = self._get_stock_size_(stock)
        total_area = stock_w * stock_h

        # Kiểm tra và tính diện tích đã chiếm dụng (có sản phẩm đã được cắt)
        used_area = 0
        for x in range(stock_w):
            for y in range(stock_h):
                if stock[x, y] != -1:  # Kiểm tra nếu ô đã bị chiếm dụng
                    used_area += 1

        # Tính diện tích thừa còn lại
        return total_area - used_area 
    

    def genetic_algorithm(self, observation, info):
        # Khởi tạo quần thể ban đầu
        population = self.initialize_population(observation, pop_size=50)
        num_generations = 100

        # Tiến hành qua nhiều thế hệ
        for generation in range(num_generations):
            # Đánh giá fitness
            fitness_scores = [(self.fitness(individual, observation), individual) for individual in population]
            fitness_scores.sort(key=lambda x: x[0], reverse=True)  # Sắp xếp theo fitness giảm dần

            # Lựa chọn (Selection): Giữ lại các cá thể tốt nhất
            selected = [ind for _, ind in fitness_scores[:10]]

            # Lai ghép (Crossover) và đột biến (Mutation)
            next_generation = []
            while len(next_generation) < len(population):
                parent1, parent2 = random.sample(selected, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child, observation)
                next_generation.append(child)

            # Cập nhật quần thể
            population = next_generation

        # Trả về giải pháp tốt nhất
        best_solution = max(population, key=lambda ind: self.fitness(ind, observation))
        best_solution_sorted = sorted(
            best_solution, 
            key=lambda ind: self.remaining_space(observation["stocks"][ind["stock_idx"]]) - (ind["size"][0] * ind["size"][1]), 
            reverse=False
        )
        return self.decode_solution(best_solution_sorted)

    # Tạo quần thể các lời giải ban đầu.
    def initialize_population(self, observation, pop_size):
        # Khởi tạo quần thể ngẫu nhiên.
        population = []
        for _ in range(pop_size):
            solution = []
            for prod in observation["products"]:
                for _ in range(prod["quantity"]):
                    prod_size = prod["size"]
                    stock_idx, pos_x, pos_y = self._random_(prod_size, observation)
                    solution.append({"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)})
            population.append(solution)
        return population
    
    # Tính điểm fitness cho một cá thể dựa trên diện tích đã sử dụng.
    def fitness(self, individual, observation):
        filled_area = 0
        used_stocks = set()  # Lưu các tấm đã được sử dụng
        for action in individual:
            stock_idx = action["stock_idx"]
            if stock_idx == -1:  # Nếu sản phẩm không được đặt lên tấm nào
                continue
            used_stocks.add(stock_idx)
            filled_area += action["size"][0] * action["size"][1]
        total_area = sum(
            self._get_stock_size_(observation["stocks"][idx])[0] *
            self._get_stock_size_(observation["stocks"][idx])[1]
            for idx in used_stocks
        )
        if total_area == 0:
            return 0
        return filled_area / total_area
    
    #  Lai ghép hai cá thể cha mẹ để tạo ra cá thể con.
    def crossover(self, parent1, parent2):
        split = len(parent1) // 2
        child = parent1[:split] + parent2[split:]
        return child
    
    # ĐỘt biến. Tăng tính đa dạng của quần thể bằng cách thay đổi một phần ngẫu nhiên của cá thể.
    def mutate(self, individual, observation):
        if random.random() < 0.1:  # Tỉ lệ đột biến  10% >=0 && <1
            idx = random.randint(0, len(individual) - 1)
            action = individual[idx]
            action["stock_idx"], pos_x, pos_y = self._random_(action["size"], observation)
            action["position"] = (pos_x, pos_y)
            individual[idx] = action
        return individual
    
    # 
    def decode_solution(self, solution):
        if solution:     
            best_action = solution[0]
            return {"stock_idx": best_action["stock_idx"], "size": best_action["size"], "position": best_action["position"]}
        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}


    # hàm random tấm vật liệu và vị trí để đặt sản phẩm
    def _random_(self, prod_size, observation):
        stock_idx = -1
        pos_x, pos_y = 0, 0
        for _ in range(100):
            idx = random.randint(0, len(observation["stocks"]) - 1)
            stock = observation["stocks"][idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size
            if stock_w >= prod_w and stock_h >= prod_h:
                pos_x = random.randint(0, stock_w - prod_w)
                pos_y = random.randint(0, stock_h - prod_h)
                if self._can_place_(stock, (pos_x, pos_y), prod_size):
                    stock_idx = idx
                    break

            if stock_w >= prod_h and stock_h >= prod_w:
                pos_x = random.randint(0, stock_w - prod_h)
                pos_y = random.randint(0, stock_h - prod_w)
                if self._can_place_(stock, (pos_x, pos_y), prod_size[::-1]):
                    prod_size = prod_size[::-1]
                    stock_idx = idx
                    break
        return stock_idx, pos_x, pos_y