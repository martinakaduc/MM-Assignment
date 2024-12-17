from policy import Policy
import copy
import numpy as np
class Policy2313306_2311444_2311399(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.check = True
            self.S = [0.0] * 100
            self.data = []
            self.stocks = []
        elif policy_id == 2:
            self.policy_id = policy_id
            self.best_solution = None
            self.obs_copy = None
            self.alpha = 1   # hệ số alpha (ảnh hưởng của pheromone)
            self.beta = 1     # hệ số beta (ảnh hưởng của heuristic)
            self.step = None
            self.pheromone_matrix = None 
            self.hs_bayhoi = 0.1
            self.max_iterations = 3
            self.num_ants = 1
            self.needed = 0


####
##  Algorithm 1: First Fit Decreasing (FFD)
####
    def precompute(self, observation):
        """
        Tính toán trước các hành động và lưu vào `self.data`.
        """
        list_prods = copy.deepcopy(observation["products"])
        self.stock = copy.deepcopy(observation["stocks"])
        
        indexed_stocks = list(enumerate(self.stock))
        indexed_prod = list(enumerate(list_prods))
        
        sorted_stocks = sorted(
            indexed_stocks,
            key=lambda item: self._get_stock_size_(item[1])[0] * self._get_stock_size_(item[1])[1],
            reverse=True
        )

        sorted_prod =  sorted(
            indexed_prod,
            key=lambda item: item[1]["size"][0] * item[1]["size"][1],
            reverse=True
        )
        # Cập nhật diện tích các stock
        for stock_id, stock in sorted_stocks:
            stock_w, stock_h = self._get_stock_size_(stock)
            self.S[stock_id] = stock_w * stock_h

        for i, prod in sorted_prod:
            prod_size = prod["size"]
            quantity = prod["quantity"]

            if quantity <= 0:
                continue
            
            for stock_id, stock in sorted_stocks:
                stock_w, stock_h = self._get_stock_size_(stock)
                
                
                if stock_w * stock_h < prod_size[0] * prod_size[1]:
                    break
                if self.S[stock_id] < prod_size[0] * prod_size[1]:
                    continue

                # Tìm vị trí đặt sản phẩm không xoay
                while quantity > 0:
                    pos = self.find_placement(stock, prod_size, i)
                    if pos:
                        self.data.append({
                            "stock_idx": stock_id,
                            "size": prod_size,
                            "position": pos
                        })
                        quantity -= 1
                        self.S[stock_id] -= prod_size[0] * prod_size[1]
                    else:
                        break

                # Tìm vị trí đặt sản phẩm xoay 90 độ
                rotated_size = prod_size[::-1]
                while quantity > 0:
                    pos = self.find_placement(stock, rotated_size, i)
                    if pos:
                        self.data.append({
                            "stock_idx": stock_id,
                            "size": rotated_size,
                            "position": pos
                        })
                        quantity -= 1
                        self.S[stock_id] -= rotated_size[0] * rotated_size[1]
                    else:
                        break

        # Nếu không còn hành động khả thi, thêm hành động mặc định
    
    def get_action_1(self, observation, info):
        # Student code here
        
        if self.check == True:
            self.precompute(observation)
            self.check = False
        if self.data:
            res = self.data.pop(0)
            if sum(prod["quantity"] for prod in observation["products"]) == 1: # Nếu số lượng sản phẩm còn lại bằng 1 thì gán check = False
                    self.check = True
                    self.S = [0.0] * 100
                    self.data = []
                    self.stocks = []
            return res
        else:
            return {"stock_idx": -1, "size": [0, 0], "position": (None, None)}

    def find_placement(self, stock, prod_size, prod_idx):
        
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self._can_place_(stock, (x, y), prod_size):
                    for i in range(prod_w):
                        for j in range(prod_h):
                            stock[x + i][y + j] = prod_idx
                    return (x, y)
        return None
    
####
##  Algorithm 2: Ant Colony Optimization (ACO)
####
    def get_action_2(self, observation, info):
        if self.needed == 0:
            self.needed = sum(product["quantity"] for product in observation["products"])
            st = [np.array(copy.deepcopy(stock), dtype=object) for stock in observation["stocks"]]
            pr = [copy.deepcopy(product) for product in observation["products"]]
            self.obs_copy = {"stocks": st, "products": pr}
            self.pheromone_matrix = np.ones((len(st),len(pr)))
            self.step = -1
            self.best_solution = self.aco_cutting(st, pr)
        self.step += 1
        self.needed -=1

        action = {"stock_idx":self.best_solution[self.step]["stock_idx"],"size": self.best_solution[self.step]["size"], "position": self.best_solution[self.step]["position"]}
        return action

    def aco_cutting(self, stocks, products):
        self.pheromone_matrix = np.ones((len(stocks), len(products)))  # khởi tạo ma trận pheromone
        best_solution = None
        best_crit = float('inf') #khởi tạo vô cùng

        for _ in range(self.max_iterations):
            ants = [self.Ant([np.array(copy.deepcopy(st),dtype = object) for st in stocks], copy.deepcopy(products)) for _ in range(self.num_ants)]

            for i,ant in enumerate(ants):
                solution = self.construct_solution(ant)
                crit = self.evaluate_solution(solution, ant.stock)
                if crit < best_crit:
                    best_crit = crit
                    best_solution = solution
        
            self.update_pheromones(ants)
        return best_solution

    class Ant:
        def __init__(self, stocks, products):
            self.stock = stocks  # khởi tạo những thuộc tính cho 1 con kiến
            self.products = products
            self.solution = []

    def construct_solution(self, ant):
        solution = []
        #tính ma trận xác suất
        combined_prob_matrix = self.calc_combined_prob(ant.stock, ant.products)
        while True:
            #chọn cặp để xét
       #     print("he")
            stock_idx, product_idx = self.select_cut(combined_prob_matrix)
            product = ant.products[product_idx]
            if product['quantity'] == 0:
                for i in range (len(ant.stock)):
                    combined_prob_matrix[i][product_idx] = 0
                combined_prob_matrix /= np.sum(combined_prob_matrix)
                continue  # bỏ qua

            position = self.find_valid_position(ant.stock[stock_idx], product["size"])
            position_2 = self.find_valid_position_2(ant.stock[stock_idx],product["size"])
            if position[0] is not None:
                    cut = {"stock_idx": stock_idx, "size": product['size'], "position": position, "prod_idx": product_idx}
                    solution.append(cut)
                    ant.stock[stock_idx][position[0]:position[0]+product["size"][0], position[1]:position[1] + product["size"][1]] = product_idx #cập nhật vùng cắt trên bản sao của stock
                    w, h = self._get_stock_size_(ant.stock[stock_idx])
                    waste = np.sum(ant.stock[stock_idx][0:w, 0:h] == -1)
                    combined_prob_matrix[stock_idx] = self.pheromone_matrix[stock_idx] ** self.alpha * (waste+10000)
                    # nếu kiến gặp 1 cắt 1 khối, tiến hành tăng mạnh xác suất gặp
                    ant.products[product_idx]["quantity"] -= 1
                    combined_prob_matrix /= np.sum(combined_prob_matrix)
            elif position_2[0] is not None:
                cut = {"stock_idx": stock_idx, "size": product['size'][::-1], "position": position_2, "prod_idx": product_idx}
                solution.append(cut)
                ant.stock[stock_idx][position_2[0]:position_2[0]+product["size"][1], position_2[1]:position_2[1] + product["size"][0]] = product_idx #cập nhật vùng cắt trên bản sao của stock
                w, h = self._get_stock_size_(ant.stock[stock_idx])
                waste = np.sum(ant.stock[stock_idx][0:w, 0:h] == -1)
                combined_prob_matrix[stock_idx] = self.pheromone_matrix[stock_idx] ** self.alpha * (waste+10000)
                ant.products[product_idx]["quantity"] -= 1
                combined_prob_matrix /= np.sum(combined_prob_matrix)
            else:
                combined_prob_matrix[stock_idx][product_idx] = 0
                combined_prob_matrix /= np.sum(combined_prob_matrix)
            if len(solution) == self.needed:
                break
        
        ant.solution = solution
        return solution

    def select_cut(self, combined_prob_matrix):
        # chuyển ma trận xác suất về 1 chiều
        flat_probs = combined_prob_matrix.flatten()
        # chọn index
        selected_index = np.random.choice(range(flat_probs.size), p=flat_probs)
        # chuyển về index 2 chiều
        stock_idx = selected_index // combined_prob_matrix.shape[1]
        product_idx = selected_index % combined_prob_matrix.shape[1]
        return stock_idx, product_idx

    def calc_combined_prob(self, stocks, products):
        num_stocks = len(stocks)
        num_products = len(products)
        
        # Initialize the combined probability matrix
        combined_prob_matrix = np.zeros((num_stocks, num_products))
        
        for i in range(num_stocks):
            sw, sh = self._get_stock_size_(stocks[i])
            for j in range(num_products):
                pheromone = self.pheromone_matrix[i][j]
                pw,ph = products[j]["size"]
                heuristic = 0
                if (pw<sw and ph<sh) or (pw<sh and ph<sw):
                    heuristic = products[j]["quantity"]*self.heuristic_value(stocks[i], products[j])  # Define a heuristic for stock-product pair
                combined_prob_matrix[i, j] = pheromone ** self.alpha * heuristic ** self.beta  # Adjust alpha and beta as needed

        # Normalize the combined probability matrix
        combined_prob_matrix /= np.sum(combined_prob_matrix) if np.sum(combined_prob_matrix) > 0 else 1
        
        return combined_prob_matrix

    def heuristic_value(self, stock,product):
        # tính giá trị heuristic dựa trên diện tích còn thừa của stock
        st_w,st_h = self._get_stock_size_(stock)
        prod_w, prod_h = product["size"]
        if ((prod_h<=st_h and prod_w<=st_w) or (prod_h<=st_w and prod_w<=st_h)) : 
              area_waste = np.sum(stock[0:st_w,0:st_h] == -1)
              return float (1/(1+area_waste))  # cang nhieu dien tich thua xac xuat chon cang giam
        return 0

    def find_valid_position(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        if prod_h<=stock_h and prod_w<=stock_w: 
            for x in range(stock_w - prod_w + 1):
                for y in range(stock_h - prod_h + 1):
                    if self._can_place_(stock, ( x, y), prod_size):
                        return (x, y)
        return (None, None)
    def find_valid_position_2(self, stock, prod_size):
        stock_w, stock_h = self._get_stock_size_(stock)
        prod_w, prod_h = prod_size
        if prod_h<=stock_w and prod_w<=stock_h: 
            for x in range(stock_w - prod_h + 1):
                for y in range(stock_h - prod_w + 1):
                    if self._can_place_(stock, ( x, y), prod_size[::-1]):
                        return (x, y)
        return (None, None)
       
    def update_pheromones(self, ants):
        for ant in ants:
            for i, st in  enumerate(ant.stock):
                for j, pr in enumerate(ant.products):
                   self.pheromone_matrix[i, j] += self.pheromone_matrix[i, j] * (1+self.heuristic_value(st,pr))
        self.pheromone_matrix *= (1-self.hs_bayhoi)
        self.pheromone_matrix = np.maximum(self.pheromone_matrix, 1.0)
    # hàm đánh giá: đánh giá thông qua diện tích còn thừa của các stock được sử dụng
    def evaluate_solution(self, solution, stock):
        total_waste = 0
        cnt = 0  
        for st in stock:
            stock_w, stock_h = self._get_stock_size_(st)
            # kiểm tra xem stock có được xài đến chưa
            free = np.all(st[0:stock_w,0:stock_h] == -1)
            res = 0
            if free == False:
                cnt+=1
                res = np.sum(st[0:stock_w,0:stock_h] == -1)
            total_waste += res
        return total_waste
    
    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.get_action_1(observation, info)
        else:
            return self.get_action_2(observation, info)
    # You can add more functions if needed
