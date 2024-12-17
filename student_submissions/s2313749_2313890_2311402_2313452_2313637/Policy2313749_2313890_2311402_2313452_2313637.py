from policy import Policy
import numpy as np
import random
from copy import deepcopy
class Policy2210xxx(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.id=1
            self.dem=0
            self.count=0
            self.best_solution = None
            self.products_list = None
            # Biến kiểm tra xem sản phẩm đã đủ chưa
            self.count_products = 0
            pass
        elif policy_id == 2:
            self.id=2
            self.dem=0
            self.count=0
            self.leng=0
            pass
    def get_action(self, observation, info):
        # Student code here
        if (self.id==2):
         if (self.dem==0):
             self.list_prods = observation["products"]
             self.list_prods=sorted(self.list_prods, key=lambda x: x['size'][0] * x['size'][1],reverse=True)
             for i in self.list_prods:
                 #(i)
                 self.dem=self.dem+i['quantity']
             ls=observation["stocks"]
             self.list_stock=[]
             for i,stock in enumerate(observation["stocks"]):
                 self.list_stock.append({'index':i,'stock': stock,'trim': np.sum(stock == -1)})
             self.list_stock=sorted(self.list_stock,key=lambda x: x['trim'])
         self.dem=self.dem-1
         return self.BestFit( observation, info)
        if (self.id==1):
            products = observation["products"]
            stocks = observation["stocks"]
            output_txt = None
            if self.dem==0:
                for i in products:
                    self.count=self.count+i['quantity']
                    #(i)
                self.best_solution = self.GA_algorithm(products, stocks, population_size=10, num_generations=10)
            for stock_idx, stock in enumerate(self.best_solution):    
                stock_w, stock_h = self._get_stock_size_(stock)

                # Duyệt qua từng ô trong stock
                for i in range(stock_w):
                    for j in range(stock_h):
                        # Nếu ô đó chứa sản phẩm thì trả về action
                        if stock[i, j] >= 0:
                            # Kiểm tra stock[i, j] có phải là sản phẩm không
                            for product in self.products_list:
                                # if product["index"] == stock[i, j]:
                                # Cải tiến cho rotate
                                if product["index"] == stock[i, j] or product["index"] == stock[i, j] -1000:
                                    size = product["size"]
                                    # Cải tiến cho rotate
                                    if product["index"] == stock[i, j] -1000:
                                        size = size[::-1]
                                    w, h = size
                                    # Xóa sản phẩm khỏi best_solution
                                    for x in range(i, i + w):
                                        for y in range(j, j + h):
                                            if x < stock_w and y < stock_h:
                                                stock[x, y] = -1
                                 # Xóa sản phẩm khỏi products_list
                                    self.products_list.remove(product)
                                    # Trả về action
                                    #(products) # In thử
                                    self.dem=self.dem+1
                                    if (self.dem==self.count):
                                        self.dem=0
                                        self.count=0
                                    #(size)
                                    return {"stock_idx": stock_idx, "size": size, "position": (i, j)}
            #(len(self.products_list))
            #(self.count_products)
            return None
    def BestFit(self, observation, info):
         for prod in self.list_prods:
             quantity=prod['quantity']
             if (quantity>0):
                 for stock in self.list_stock:
                     stock_w, stock_h = self._get_stock_size_(stock['stock'])
                     prod_size = prod["size"]
                     prod_w, prod_h = prod['size']

                     prod_hr, prod_wr=prod['size']
                     prod_size_r=np.array([prod_wr,prod_hr])
                     if prod_w*prod_h>stock['trim']:
                         continue
                     if stock_w < prod_w or stock_h < prod_h:
                         if stock_w<prod_wr or stock_h<prod_hr:#Xoay
                             continue
                         else:
                             pass
                     pos_x, pos_y = None, None
                     for x in range(stock_w - min(prod_w,prod_wr) + 1):
                         for y in range(stock_h - min(prod_h,prod_hr) + 1):
                             if (self._can_place_(stock['stock'], (x, y), prod_size)) and(x+prod_w<=stock_w and y+prod_h<=stock_h):
                                 pos_x, pos_y = x, y
                                 break
                             if self._can_place_(stock['stock'], (x, y), prod_size_r) and((x+prod_wr<=stock_w and y+prod_hr<=stock_h)):
                                 pos_x, pos_y = x, y
                                 prod_size=prod_size_r
                                 break
                         if pos_x is not None and pos_y is not None:
                             break
                     if pos_x is not None and pos_y is not None:
                         stock_idx = stock['index']
                         stock['trim']=stock['trim']-prod_w*prod_h
                         for i in range(stock_idx,1,-1):
                             if (self.list_stock[i]['trim']<self.list_stock[i-1]['trim']):
                                 tg=self.list_stock[i]['trim']
                                 self.list_stock[i]['trim']=self.list_stock[i-1]['trim']
                                 self.list_stock[i-1]['trim']=tg
                             else:
                                 break
                         #self.list_stock=sorted(self.list_stock,key=lambda x: x['trim'])
                         break
                 if pos_x is not None and pos_y is not None:
                     break
         return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}
    def initialize_population(self, products, stocks, population_size=10):
        # Lấy danh sách các sản phẩm cần cắt
        products_list = []
        for idx, product in enumerate(products):
            size = product["size"]                                              # np.array([w, h])
            quantity = product["quantity"]
            products_list.extend([{"index": idx, "size": size}] * quantity)     # [{index: idx, size: size}, ...]
        # Dùng deepcopy để không ảnh hưởng đến products_list gốc
        self.products_list = deepcopy(products_list)

        # Khởi tạo quần thể
        population = []
        for seed in range(population_size):
            #Tạo lời giải ngẫu nhiên (danh sách các stock)
            solution = []
            for stock in stocks:
                # Khởi tạo solution có kích thước 100x100
                stock_matrix = np.full((100, 100), -2)
                # Lấy kích thước của stock và chỉnh các ô nằm trong kích thước stock thành -1
                stock_w, stock_h = self._get_stock_size_(stock)
                stock_matrix[:stock_w, :stock_h] = -1
                solution.append(stock_matrix)

            # Đặt các sản phẩm vào các stock

            random.seed(seed)
            random.shuffle(products_list)   # Xáo trộn danh sách sản phẩm

            # Xáo trộn các chỉ số của stock nhưng không làm thay đổi thứ tự của chúng
            stock_indices = list(range(len(solution)))
            random.shuffle(stock_indices)

            for product in products_list:
                index = product["index"]
                size = product["size"]
                placed = False

                # Thử đặt sản phẩm vào các stock
                # for stock_idx, stock in enumerate(solution):
                # for stock_idx in stock_indices:     # Cải tiến
                #     stock = solution[stock_idx]
                #     stock_w, stock_h = self._get_stock_size_(stock)
                #     prod_w, prod_h = size

                #     if stock_w < prod_w or stock_h < prod_h:
                #         continue

                #     for i in range(stock_w - prod_w + 1):
                #         for j in range(stock_h - prod_h + 1):
                #             if self._can_place_(stock, (i, j), size):
                #                 stock[i : i + prod_w, j : j + prod_h] = index
                #                 placed = True
                #                 break
                #         if placed:
                #             break
                #     if placed:
                #         break

                # Cải tiến cho rotate
                for stock_idx in stock_indices:
                    stock = solution[stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = size
                    prod_hr, prod_wr = size
                    prod_size_r = np.array([prod_wr, prod_hr])

                    if stock_w < prod_w or stock_h < prod_h:
                        if stock_w < prod_wr or stock_h < prod_hr:
                            continue
                        else:
                            pass

                    if random.choice([0, 1]) == 0:
                        # Đại diện cho trường hợp không rotate
                        if stock_w > prod_w and stock_h > prod_h:
                            for i in range(stock_w - prod_w + 1):
                                for j in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (i, j), size):
                                        stock[i : i + prod_w, j : j + prod_h] = index
                                        placed = True
                                        self.count_products += 1
                                        break
                                if placed:
                                    break
                            if placed:
                                break
                            # Nếu không đặt được sản phẩm vào stock thì rotate sản phẩm và đặt lại thử
                        if not placed:
                            if stock_w > prod_wr and stock_h > prod_hr:
                                for i in range(stock_w - prod_wr + 1):
                                    for j in range(stock_h - prod_hr + 1):
                                        if self._can_place_(stock, (i, j), prod_size_r):
                                            # Index của sản phẩm bị rotate sẽ là index + độ dài danh sách sản phẩm
                                            stock[i : i + prod_wr, j : j + prod_hr] = index + 1000
                                            self.count_products += 1
                                            placed = True
                                            break
                                    if placed:
                                        break
                                if placed:
                                    break
                    else:
                        # Đại diện cho trường hợp rotate
                        if stock_w > prod_wr and stock_h > prod_hr:
                            for i in range(stock_w - prod_wr + 1):
                                for j in range(stock_h - prod_hr + 1):
                                    if self._can_place_(stock, (i, j), prod_size_r):
                                        # Index của sản phẩm bị rotate sẽ là index + độ dài danh sách sản phẩm
                                        stock[i : i + prod_wr, j : j + prod_hr] = index + 1000
                                        self.count_products += 1
                                        placed = True
                                        break
                                if placed:
                                    break
                            if placed:
                                break
                            # Nếu không đặt được sản phẩm vào stock thì rotate sản phẩm và đặt lại thử
                        if not placed:
                            if stock_w > prod_w and stock_h > prod_h:
                                for i in range(stock_w - prod_w + 1):
                                    for j in range(stock_h - prod_h + 1):
                                        if self._can_place_(stock, (i, j), size):
                                            stock[i : i + prod_w, j : j + prod_h] = index
                                            self.count_products += 1
                                            placed = True
                                            break
                                    if placed:
                                        break
                                if placed:
                                    break


            population.append(solution)
        return population

    def fitness_function(self,  individual):
        # Các thông số cần thiết
        total_used_area = 0

        # Duyệt qua các tấm và nếu tấm đó chứa sản phảm thì cộng thêm diện tích tấm đó vào total_used_area
        for stock in individual:
            if np.sum(stock >= 0) > 0:
                total_used_area += np.sum(stock >= -1)

        # Tính fitness
        fitness = total_used_area
        return fitness

    def selection(self, fitness, population):
        # Sử dụng phương pháp Elitism để chọn 2 cá thể tốt nhất (vì đây là bài toán minimization nên cho fitness càng thấp thì cá thể càng tốt)
        best_individuals = sorted(zip(fitness, population), key=lambda x: x[0])[:2]
        selected_population = [individual for fitness, individual in best_individuals]
        return selected_population

    def repair_solution(self, solution, products):
        # Tạo danh sách các sản phẩm cần cắt
        products_list = []
        for idx, product in enumerate(products):
            size = product["size"]
            quantity = product["quantity"]
            products_list.extend([{"index": idx, "size": size}] * quantity)
        
        # Kiểm tra trong solution, nếu sản phẩm nằm trong solution thì xóa nó khỏi danh sách, còn không thì xóa nó khỏi solution
        for stock in solution:
            used_area = np.zeros_like(stock, dtype=bool)    # False # Mảng đánh dấu các ô đã được sử dụng => Dùng để lọc các ô chứa sản phẩm trong stock của solution
            unwanted_products = False                               # Biến đánh dấu sản phẩm dư thừa => Dùng để xóa sản phẩm dư thừa khỏi solution

            stock_w, stock_h = self._get_stock_size_(stock)         # Lấy kích thước của stock
            for i in range(stock_w):
                for j in range(stock_h):
                    if stock[i, j] >= 0 and not used_area[i, j]:    # Nếu ô đó chứa sản phẩm và chưa được xét
                        product_idx = stock[i, j]                   # Lấy index của sản phẩm
                        # Tìm sản phẩm trong danh sách sản phẩm cần cắt
                        for product in products_list:
                            # if product["index"] == product_idx :
                            # Cải tiến cho rotate
                            if product["index"] == product_idx or product["index"] == product_idx - 1000:
                                size = product["size"]
                                # Cải tiến cho rotate
                                if product["index"] == product_idx - 1000:
                                    size = size[::-1]
                                product_w, product_h = size
                                # Đánh dấu các ô tương ứng với sản phẩm
                                for x in range(i, i + product_w):
                                    for y in range(j, j + product_h):
                                        if x < stock_w and y < stock_h:
                                            used_area[x, y] = True      # Đánh dấu ô đã được sử dụng
                                unwanted_products = False
                                products_list.remove(product)       # Xóa sản phẩm khỏi danh sách
                                break
    # ----------------------------- Cần kiểm tra logic thêm -----------------------------
                            else:
                                unwanted_products = True
                        if unwanted_products:
                            stock[i, j] = -1                # Xóa sản phẩm khỏi solution
                            used_area[i, j] = True          # Đánh dấu ô đã được sử dụng
                            
        # Đặt các sản phẩm còn lại vào solution (bước này có thể xem là mutation)
        random.shuffle(products_list)
        for product in products_list:
            index = product["index"]
            size = product["size"]
            product_w, product_h = size
            placed = False

            # Duyệt qua từng stock trong solution
            for stock in solution:
                # Kiểm tra xem stock có chứa sản phẩm nào không
                if np.sum(stock >= 0) == 0:
                    continue

                stock_w, stock_h = self._get_stock_size_(stock)    # Lấy kích thước của stock

                # if stock_w < product_w or stock_h < product_h:
                #     continue

                # Cải tiến cho rotate
                if stock_w < product_w or stock_h < product_h:
                    if stock_w < product_h or stock_h < product_w:
                        continue
                    else:
                        pass

                # # Kiểm tra xem sản phẩm có thể đặt vào stock không
                # for i in range(stock_w - product_w + 1):
                #     for j in range(stock_h - product_h + 1):
                #         if self._can_place_(stock, (i, j), size):
                #             stock[i : i + product_w, j : j + product_h] = index
                #             placed = True
                #             break
                #     if placed:
                #         break
                # if placed:
                #     break

                # Cải tiến cho rotate
                if random.choice([0, 1]) == 0:
                    if stock_w > product_w and stock_h > product_h:
                        for i in range(stock_w - product_w + 1):
                            for j in range(stock_h - product_h + 1):
                                if self._can_place_(stock, (i, j), size):
                                    stock[i : i + product_w, j : j + product_h] = index
                                    placed = True
                                    break
                            if placed:
                                break
                        if placed:
                            break
                    # Nếu không đặt được sản phẩm vào stock thì rotate sản phẩm và đặt lại thử
                    if not placed:
                        if stock_w > product_h and stock_h > product_w:
                            for i in range(stock_w - product_h + 1):
                                for j in range(stock_h - product_w + 1):
                                    if self._can_place_(stock, (i, j), (product_h, product_w)):
                                        # Index của sản phẩm bị rotate sẽ là index + độ dài danh sách sản phẩm
                                        stock[i : i + product_h, j : j + product_w] = index + 1000
                                        placed = True
                                        break
                                if placed:
                                    break
                        if placed:
                            break
                else:
                    if stock_w > product_h and stock_h > product_w:
                        for i in range(stock_w - product_h + 1):
                            for j in range(stock_h - product_w + 1):
                                if self._can_place_(stock, (i, j), (product_h, product_w)):
                                    # Index của sản phẩm bị rotate sẽ là index + độ dài danh sách sản phẩm
                                    stock[i : i + product_h, j : j + product_w] = index + 1000
                                    placed = True
                                    break
                            if placed:
                                break
                        if placed:
                            break
                        # Nếu không đặt được sản phẩm vào stock thì rotate sản phẩm và đặt lại thử
                    if not placed:
                        if stock_w > product_w and stock_h > product_h:
                            for i in range(stock_w - product_w + 1):
                                for j in range(stock_h - product_h + 1):
                                    if self._can_place_(stock, (i, j), size):
                                        stock[i : i + product_w, j : j + product_h] = index
                                        placed = True
                                        break
                                if placed:
                                    break
                            if placed:
                                break

            # Nếu không đặt được sản phẩm vào các stock đã chứa sản phẩm thì đặt sản phẩm vào stock bất kì trong solution (dùng random)
            if not placed:
                # Duyệt qua từng stock trong solution nhưng theo thứ tự ngẫu nhiên
                stock_indices = list(range(len(solution)))
                random.shuffle(stock_indices)
                for stock_idx in stock_indices:
                    stock = solution[stock_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)

                    # if stock_w < product_w or stock_h < product_h:
                    #     continue

                    # Cải tiến cho rotate
                    if stock_w < product_w or stock_h < product_h:
                        if stock_w < product_h or stock_h < product_w:
                            continue
                        else:
                            pass

                    # for i in range(stock_w - product_w + 1):
                    #     for j in range(stock_h - product_h + 1):
                    #         if self._can_place_(stock, (i, j), size):
                    #             stock[i : i + product_w, j : j + product_h] = index
                    #             placed = True
                    #             break
                    #     if placed:
                    #         break
                    # if placed:
                    #     break

                    # Cải tiến cho rotate
                    if random.choice([0, 1]) == 0:
                        if stock_w > product_w and stock_h > product_h:
                            for i in range(stock_w - product_w + 1):
                                for j in range(stock_h - product_h + 1):
                                    if self._can_place_(stock, (i, j), size):
                                        stock[i : i + product_w, j : j + product_h] = index
                                        placed = True
                                        break
                                if placed:
                                    break
                            if placed:
                                break
                        # Nếu không đặt được sản phẩm vào stock thì rotate sản phẩm và đặt lại thử
                        if not placed:
                            if stock_w > product_h and stock_h > product_w:
                                for i in range(stock_w - product_h + 1):
                                    for j in range(stock_h - product_w + 1):
                                        if self._can_place_(stock, (i, j), (product_h, product_w)):
                                            # Index của sản phẩm bị rotate sẽ là index + độ dài danh sách sản phẩm
                                            stock[i : i + product_h, j : j + product_w] = index + 1000
                                            placed = True
                                            break
                                    if placed:
                                        break
                                if placed:
                                    break
                        if placed:
                            break
                    else:
                        if stock_w > product_h and stock_h > product_w:
                            for i in range(stock_w - product_h + 1):
                                for j in range(stock_h - product_w + 1):
                                    if self._can_place_(stock, (i, j), (product_h, product_w)):
                                        # Index của sản phẩm bị rotate sẽ là index + độ dài danh sách sản phẩm
                                        stock[i : i + product_h, j : j + product_w] = index + 1000
                                        placed = True
                                        break
                                if placed:
                                    break
                            if placed:
                                break
                            # Nếu không đặt được sản phẩm vào stock thì rotate sản phẩm và đặt lại thử
                        if not placed:
                            if stock_w > product_w and stock_h > product_h:
                                for i in range(stock_w - product_w + 1):
                                    for j in range(stock_h - product_h + 1):
                                        if self._can_place_(stock, (i, j), size):
                                            stock[i : i + product_w, j : j + product_h] = index
                                            placed = True
                                            break
                                    if placed:
                                        break
                                if placed:
                                    break

        return solution

    def crossover(self, parent1, parent2, products):
        # Sao chép parent1 và parent2 để không ảnh hưởng đến cá thể gốc
        p1 = deepcopy(parent1)
        p2 = deepcopy(parent2)

        # # Chọn ngẫu nhiên 1 điểm cắt
        # random.seed()
        # cut_point = random.randint(0, len(p1) - 1)

        # # Nối 2 phần của p1 và p2 để tạo ra con
        # child = p1[:cut_point] + p2[cut_point:]

        # Cắt lần lượt random 10 stock hoặc của parent1 hoặc của parent2 để tạo ra con
        child = []
        for i in range(10):
            random.seed()
            choice = random.choice([0, 1])
            if choice == 0:
                # Sao chép stock thứ 10*i đến 10*i+10 của parent1
                child.extend(p1[10*i:10*i+10])
            else:
                # Sao chép stock thứ 10*i đến 10*i+10 của parent2
                child.extend(p2[10*i:10*i+10])

        # Sửa lỗi của con nếu chứa chưa đúng số lượng sản phẩm (phần này có thể tạo nên đột biến)
        child = self.repair_solution(child, products)

        return child

    def crossover_population(self, population, products, num_crossover):
        # population truyền vào có 2 cá thể tốt nhất nên ta sẽ lấy 2 cá thể đó làm parent1 và parent2 để lai ra số lượng cá thể bằng num_crossover
        parent1, parent2 = population
        new_population = []
        for _ in range(num_crossover):
            child = self.crossover(parent1, parent2, products)
            new_population.append(child)
        return new_population

    # def mutation(self, ):

    #     pass

    def get_best_solution(self, population, fitness):
        best_individual = population[np.argmin(fitness)]
        return best_individual
    
    def GA_algorithm(self, products, stocks, population_size=10, num_generations=100000):
        # Khởi tạo quần thể
        population = self.initialize_population(products, stocks, population_size)

        # Đánh giá fitness cho từng cá thể trong quần thể
        fitness = [self.fitness_function(individual) for individual in population]

        # Chọn ra cá thể tốt nhất để test
        best_individual = self.get_best_solution(population, fitness)

        for _ in range(num_generations):
            # Đánh giá fitness cho từng cá thể trong quần thể
            fitness = [self.fitness_function(individual) for individual in population]

            # Chọn lọc
            selected_population = self.selection(fitness, population)

            # Lai ghép và đột biến tạo ra quần thể mới có số cá thể bằng số cá thể quần thể cũ
            new_population = self.crossover_population(selected_population, products, len(population))

            # Tính độ fitness của từng cá thể trong new_population
            new_fitness = [self.fitness_function(individual) for individual in new_population]

            # Chọn ra cá thể tốt nhất trong new_population
            best_individual = self.get_best_solution(new_population, new_fitness)

            # Cập nhật quần thể
            population = new_population

        return best_individual
            
    # Student code here
    # You can add more functions if needed
