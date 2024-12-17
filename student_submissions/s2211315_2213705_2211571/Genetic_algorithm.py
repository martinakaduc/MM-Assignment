from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Genetic(Policy):
    def __init__(self):

        # Student code here

        """
        Khởi tạo policy.
        """
        super().__init__()
        self.sheets = []
        self.stocks = []
        self.current_generation = None

    def _can_place_(self, stock, position, prod_size):
        """
        Kiểm tra xem sản phẩm có thể được đặt tại một vị trí cụ thể trên vật liệu không.
        :param stock: Mảng numpy đại diện cho vật liệu hiện có.
        :param position: Vị trí (x, y) trên vật liệu để đặt sản phẩm.
        :param prod_size: Kích thước sản phẩm (length, width).
        :return: True nếu có thể đặt sản phẩm tại vị trí, ngược lại False.
        """
        pos_x, pos_y = position                  # `pos_x` và `pos_y` là tọa độ góc trên bên trái của sản phẩm trên vật liệu.
        prod_length, prod_width = prod_size      # `prod_length` và `prod_width` là chiều dài và chiều rộng của sản phẩm.
        stock_length, stock_width = stock.shape  # `stock_length` và `stock_width` là kích thước của vật liệu.

        if pos_x + prod_length > stock_length or pos_y + prod_width > stock_width:  # Kiểm tra nếu sản phẩm tràn ra ngoài giới hạn của vật liệu
            return False

        try:
            region = stock[pos_x:pos_x + prod_length, pos_y:pos_y + prod_width]  # Lấy vùng trên vật liệu tương ứng với vị trí sản phẩm
            return np.all(region == -1)  # Kiểm tra toàn bộ vùng có trống hay không
        except IndexError:
            return False # Trả về False nếu xảy ra lỗi IndexError (tràn ra ngoài mảng numpy)

    def get_action(self, observation, info):
        # Student code here
        """
        Tìm hành động tối ưu dựa trên thuật toán Column Generation.
        :param observation: Trạng thái hiện tại của môi trường.
        :param info: Thông tin bổ sung từ môi trường.
        :return: Hành động tối ưu.
        """
        # Lấy danh sách các stocks và products từ observation
        # Lấy danh sách các stocks và products từ observation
        if self.current_generation is None:
            sts = observation.get("stocks", None)
            products = observation.get("products", None)
            # Kiểm tra tính hợp lệ của observation
            if not sts or not products:
                return {"stock_idx": 0, "size": [0, 0], "position": [0, 0]}  # Hành động mặc định

            for product in products:
                size = product["size"]  # Lấy mảng kích thước
                roll_width, roll_height = size  # Truy cập chiều dài và chiều rộng
                quantity = product["quantity"]
                self.sheets.append(Sheet(int(roll_height), int(roll_width), int(quantity)))

            for i in range(len(sts)):
                cols, rows = np.where(sts[i] == -1) 
                length = np.ptp(rows) + 1 
                width = np.ptp(cols) + 1
                first_position = (int(rows[0]), int(cols[0]))
                lb_patterns, ub_sheet = self.calc_lb_ub(self.sheets, width, length)
                self.stocks.append(Stock(int(width), int(length), lb_patterns=lb_patterns, ub_sheet=ub_sheet, idx=i, position=first_position))

            genetic = Genetic_helper(self.stocks, self.sheets)
            self.current_generation = genetic.genetic_algorithm()

        idx, sheet = self.current_generation.choose_sheet()
        if idx is None and sheet is None: 
            return {"stock_idx": -1, "size": [0, 0], "position": [0, 0]}
        y = sheet.position[0]
        x = sheet.position[1]
        h = sheet.height
        w = sheet.width
        if sheet.rotated == True:
            return {"stock_idx": idx, "size": (np.int64(h), np.int64(w)), "position": (x,y)}
        else:
            return {"stock_idx": idx, "size": (np.int64(w), np.int64(h)), "position": (x,y)}
    
    def calc_lb_ub(self, sheets, rect_w, rect_h):
        widths = np.array([s.width for s in sheets])
        heights = np.array([s.height for s in sheets])
        # Tính tổng diện tích của các sheets
        total_area = np.sum(widths * heights)
        lb_patterns = int(np.ceil(total_area / (rect_w * rect_h))) #Giới hạn dưới của số mẫu
        
        rect_area = rect_h * rect_w

        # Tạo mảng NumPy từ danh sách sheets
        sheet_widths = np.array([sheet.width for sheet in sheets])
        sheet_heights = np.array([sheet.height for sheet in sheets])

        # Tính upper bound cho mỗi sheet
        ub_sheet = np.floor(rect_area / (sheet_widths * sheet_heights)).astype(int)  # Chuyển đổi thành kiểu int

        return lb_patterns, ub_sheet
    
    def reset(self):
        self.sheets = []
        self.stocks = []
        self.current_generation = None
        

    # Student code here
    # You can add more functions if needed

class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __repr__(self):
        return f'{self.height} X {self.width}\n'

    def __eq__(self, other):
        return isinstance(other, Rectangle) and self.width == other.width and self.height == other.height

class Sheet(Rectangle):
    def __init__(self, width, height, demand):
        super().__init__(width, height)
        self.demand = demand

    def __repr__(self):
        return f'{self.height} X {self.width}: {self.demand}\n'


'''An sheet placed on a rectangle'''
class FixedRectangle(Rectangle):
    def __init__(self, width, height, position, rotated=False):
        if rotated:
            super().__init__(height, width)
        else:
            super().__init__(width, height)

        self.rotated = rotated
        self.position = position
        h = self.height
        w = self.width

        self.up = position[0]
        self.right = position[1]  + w
        self.down = position[0] + h
        self.left = position[1]

        self.bottom_left = position[0] + h, position[1]
        self.top_left = position
        self.top_right = position[0], position[1] + w
        self.bottom_right = position[0] + h, position[1] + w

    @staticmethod
    def create_from_tuple(info):
        width, height, pos, rotated = info
        if rotated:
            return FixedRectangle(height, width, pos, rotated=True)
        return FixedRectangle(width, height, pos)

    def as_tuple(self):
        return (self.width, self.height, self.position, self.rotated)

    def __repr__(self):
        result = f'{self.position}: {self.height} X {self.width}'
        return result

    def __eq__(self, other):
        return self.as_tuple() == other.as_tuple()

    def __hash__(self):
        return hash(self.as_tuple())

class Stock(FixedRectangle):

    def __init__(self, width, height, lb_patterns, ub_sheet, idx =-1, position=None):
        super().__init__(width, height, position=position)
        self.cuts = []
        self.free_area = width * height
        self.free_rectangles = [FixedRectangle(width, height, position=position)]
        self.idx = idx
        self.lb_patterns = lb_patterns
        self.ub_sheet = ub_sheet

    def add_cut(self, sheet):
        self.cuts.append(sheet)
        self.free_area -= sheet.width * sheet.height
    
    def choose_sheet(self):
        if self.cuts:
            sheet = self.cuts[0]
            self.cuts.remove(sheet)
            return sheet
        else: return None

    def __repr__(self):
        result = f'Index stock: {self.idx} \n'
        for cut in self.cuts:
            result += str(cut) + '  '
        result += f'\n {self.height} X {self.width}  '
        result += f'\nfree space: {self.free_area}   '
        return result
    
    def __lt__(self, other): 
        if self.free_area < 0: return other.free_area
        elif other.free_area < 0: return self.free_area
        return self.free_area < other.free_area

class Solution:
    def __init__(self, stocks, sheets_per_pattern, prints_per_pattern=None, fitness=None):
        
        self.stocks = stocks
        self.sheets_per_pattern = sheets_per_pattern
        self.prints_per_pattern = prints_per_pattern
        self.fitness = fitness

    def __gt__(self, other):
        return self.fitness > other.fitness
    def __repr__(self):
        result = ''
        for idx,bin in enumerate(self.stocks):
            if bin.free_area < (bin.width * bin.height):
                result += str(bin)
                result += f'No. of prints: ' 
                if self.prints_per_pattern == None:
                    result+= f'NONE' + '\n\n'
                else:
                    result+= str(self.prints_per_pattern[f'x{idx}']) + '\n\n'
        result += f'fitness: {self.fitness}\n'
        return result
    
    def choose_sheet(self):

        for idx,bin in enumerate(self.stocks):
            if self.prints_per_pattern[f'x{idx}'] >= 1:
                sheet = bin.choose_sheet()
                if sheet is not None:
                    return idx, sheet
        return None, None



def find_best_fit(sheet, stocks):
    """
    input : 1 sheet, list stock
    renturn : stock_idx , free_idx , rotated?
    """
    best_fit = (1000000, -1, -1, False)

    for i,current_stock in enumerate(stocks):
        for j,free_rect in enumerate(current_stock.free_rectangles):
            # Neu sheet vừa với hcn trống hiện tại...
            if sheet.width <= free_rect.width and sheet.height <= free_rect.height:
                shortest_side_fit = min(free_rect.width - sheet.width, free_rect.height - sheet.height)
                if shortest_side_fit < best_fit[0]:
                    best_fit = (shortest_side_fit, i, j, False)
            # Thử lại bằng cách xoay sheet
            if sheet.width <= free_rect.height and sheet.height <= free_rect.width:
                shortest_side_fit = min(free_rect.width - sheet.height, free_rect.height - sheet.width)
                if shortest_side_fit < best_fit[0]:
                    best_fit = (shortest_side_fit, i, j, True)
        if best_fit[1] != -1:
            return tuple(best_fit[1:])

    return tuple(best_fit[1:])

def split(sheet, free_rect):
    """
    input: 1sheet, hcn free
    return: list chứa 2 hcn free
    """
    result = []
    w = sheet.width
    h = sheet.height
    if h == free_rect.height and w == free_rect.width: return result
    if free_rect.width < free_rect.height:
        # chia theo chieu ngang
        if h < free_rect.height:   
            # if sheet.bottom_left[1]          
            result.append(FixedRectangle(width=free_rect.width, height=free_rect.height - h, position=sheet.bottom_left))
        if w < free_rect.width:
            result.append(FixedRectangle(width=free_rect.width - w, height=h, position=sheet.top_right))
    else:
        # chia theo chieu doc
        if w < free_rect.width:
            result.append(FixedRectangle(width=free_rect.width - w, height=free_rect.height, position=sheet.top_right))
        if h < free_rect.height:
            result.append(FixedRectangle(width=w, height=free_rect.height - h, position=sheet.bottom_left))
    return result

def pack_rectangles(rectangles, sheets, idx=-1):
    """
    input: list stock, list sheet
    output: list stock đã cập nhập, p : p[(i,j)] = số lượng sheet j trong mẫu i
    """
    # tạo list stock ban đầu
    if idx == -1:
        stocks = [Stock(width=rect.width, height=rect.height, lb_patterns=rect.lb_patterns, ub_sheet=rect.ub_sheet, idx=rect.idx, position=rect.position) for rect in rectangles]
    elif idx == -2:
        stocks = rectangles
    else:    
        stocks = [Stock(width=rectangles[idx].width, height=rectangles[idx].height, lb_patterns=rectangles[idx].lb_patterns, ub_sheet=rectangles[idx].ub_sheet, idx=idx, position=rectangles[idx].position)]
    p = {}  # p[(i,j)] = số lượng sheet j trong mẫu i

    for i, sheet in enumerate(sheets):
        for _ in range(sheet.demand):
            # Find globally the best choice: the rectangle wich best fits on a free_rectangle of any stock
            stock_idx, fr_idx, rotate = find_best_fit(sheet, stocks)
            if stock_idx == -1:
                return [], {}

            # update the p vector
            try:
                p[stock_idx, i] += 1
            except KeyError:
                p[stock_idx, i] = 1

            current_stock = stocks[stock_idx]
            free_rectangles = current_stock.free_rectangles
            free_rect_to_split = free_rectangles[fr_idx]

            # Thêm sheet vào phía trên bên trái của Fi
            new_cut = FixedRectangle(sheet.width, sheet.height, position=(free_rect_to_split.top_left), rotated=rotate)
            current_stock.add_cut(new_cut)
            # Chia lại hcn free đã sử dụng và thêm vào list hcn free của stock
            free_rectangles.pop(fr_idx)
            free_rectangles += split(new_cut, free_rect_to_split)
            free_rectangles = {fr for fr in free_rectangles}
            
            # if np.random.rand() >= 0.75 : return stocks, p

            while True:
                changed = False
                for fi in free_rectangles:
                    for fj in free_rectangles:
                        if fi == fj: continue
                        if fi.up == fj.up and fi.down == fj.down and min(fi.right, fj.right) == max(fi.left, fj.left):
                            free_rectangles.remove(fi)
                            free_rectangles.remove(fj)
                            free_rectangles.add(FixedRectangle(width=fi.width + fj.width, height=fi.height, position=(fi.up, min(fi.left, fj.left))))
                            changed = True
                        elif fi.left == fj.left and fi.right == fj.right and min(fi.down, fj.down) == max(fi.up, fj.up):
                            free_rectangles.remove(fi)
                            free_rectangles.remove(fj)
                            free_rectangles.add(FixedRectangle(width=fi.width, height=fi.height + fj.height, position=(min(fi.up, fj.up), fi.left)))
                            changed = True
                        if changed:
                            break
                    if changed:
                        break
                if not changed:
                    break

                current_stock.free_rectangles = [fr for fr in free_rectangles]
    return stocks, p


def two_randoms(top):
    if top < 1:
        return None, None
    if top == 1:
        return 0, 0
    r1 = np.random.randint(0, top-1)
    r2 = None
    if r1 == 0:
        try:
            r2 = np.random.randint(1, top-1)
        except ValueError: r2 = 1
    elif r1 == top-1:
        r2 = np.random.randint(0, top-2)
    else:
        r2 = np.random.randint(0, top-1)
        while r2 == r1:
            r2 = np.random.randint(0, top-1)  
    return r1, r2


class Genetic_helper:
    def __init__(self, stocks, products, pop_size=60, random_walk_steps=100, hill_climbing_neighbors=25, roulette_pop = 45, no_best_solutions=10, no_generations=30, prob_crossover=0.75):
        self.stopped = False

        self.stocks = stocks       
        self.products = products
        self.total_sheets = len(products)

        self.pop_size = pop_size
        self.random_walk_steps = random_walk_steps
        self.hill_climbing_neighbors = hill_climbing_neighbors
        self.roulette_pop = roulette_pop
        self.no_best_solutions = no_best_solutions
        self.no_generations = no_generations
        self.prob_crossover=prob_crossover


    def solve_LP(self, stocks, sheets_per_pattern, sheets):
    # Lấy thông tin từ stocks và sheets
        w = [b.free_area if b.free_area < (b.width*b.height) else 100000 for b in stocks ] #if b.free_area < (b.width*b.height)

        d = [s.demand for s in sheets]
        a = [s.width * s.height for s in sheets]
        m, n = len(w), len(d)
        p = np.zeros((m, n))
        for (j, i), sheets in sheets_per_pattern.items():
            p[j][i] = sheets
        # Tạo vector hệ số cho hàm mục tiêu
        S = np.array([wj + sum(p[j][i] * ai for i, ai in enumerate(a)) for j, wj in enumerate(w)])
        c = np.array(w)
        # Tạo ma trận hệ số cho ràng buộc
        I = np.eye(m)
        A_ub = np.transpose(-p)  # Thêm ma trận đơn vị cho các ràng buộc
        # A_ub = np.vstack((A, I))
        b_ub = np.array([-di for di in d])
        

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')

        # Kiểm tra xem bài toán có giải không
        if not res.success:
            # raise ValueError("Linear programming failed: " + res.message)
            return 100000, None
        # Lấy kết quả
        x = {f"x{i}": np.ceil(x) for i, x in enumerate(res.x)}
        X = np.ceil(res.x)
        X = [x if x == 1 else x*400 for x in X]
        fitness = sum([S[i] * X[i] for i in range(len(x))]) - sum([d[i]*a[i] for i in range(n)])

        return fitness, x

    '''Thêm tấm sheet i vào mẫu j'''
    def add(self, solution):
        #lấy thứ tự của mẫu và thứ tự của tấm sheet
        # print("Add -", end=" ")
        try: p_idx = np.random.randint(0,len(solution.stocks)-1)
        except ValueError: p_idx = 0

        s_idx = np.random.randint(0, self.total_sheets - 1)
        sheets_per_pattern = dict(solution.sheets_per_pattern)
        #sheets_per_pattern[i, j] số lượng tấm sheet thứ j trong mẫu thứ i
        try:
            sheets_per_pattern[p_idx, s_idx] += 1
        except KeyError:
            sheets_per_pattern[p_idx, s_idx] = 1
        stock = solution.stocks[p_idx]
        if sheets_per_pattern[p_idx,s_idx] >= stock.ub_sheet[s_idx]:
            return None
        return sheets_per_pattern

    ''' Xóa tấm sheet i từ mẫu j'''
    def remove(self, solution):
        try: p_idx = np.random.randint(0,len(solution.stocks)-1)
        except ValueError: p_idx = 0
        s_idx = np.random.randint(0, self.total_sheets - 1)
        sheets_per_pattern = dict(solution.sheets_per_pattern)
        #sheets_per_pattern[i, j] số lượng tấm sheet thứ j trong mẫu thứ i
        try:
            sheets_per_pattern[p_idx, s_idx] -= 1
        except KeyError:
            return None
        if sheets_per_pattern[p_idx, s_idx] < 0 \
            or sum([sheets_per_pattern[_pattern, _sheet] for _pattern, _sheet in sheets_per_pattern if _sheet == s_idx]) == 0:
            return None
        return sheets_per_pattern

    '''Chuyển 1 tấm sheet vào 1 mẫu khác'''
    def move(self, solution):
        pat1_idx, pat2_idx = two_randoms(len(solution.stocks))
        s_idx = np.random.randint(0, self.total_sheets - 1)
        sheets_per_pattern = dict(solution.sheets_per_pattern)
        #sheets_per_pattern[i, j] số lượng tấm sheet thứ j trong mẫu thứ i
        try:
            sheets_per_pattern[pat1_idx, s_idx] -= 1
        except KeyError:
            return None
        try:
            sheets_per_pattern[pat2_idx, s_idx] += 1
        except KeyError:
            sheets_per_pattern[pat2_idx, s_idx] = 1

        if  sheets_per_pattern[pat1_idx, s_idx] < 0 or \
            sheets_per_pattern[pat2_idx,s_idx] >= solution.stocks[pat2_idx].ub_sheet[s_idx]:
            return None

        return sheets_per_pattern

    '''Hoán đổi 2 sheet từ 2 mẫu khác nhau'''
    def swap(self, solution):
        pat1, pat2 = two_randoms(len(solution.stocks))
        s1, s2 = two_randoms(self.total_sheets)

        sheets_per_pattern = dict(solution.sheets_per_pattern)
        try:
            sheets_per_pattern[pat1, s1] -= 1
            sheets_per_pattern[pat2, s2] -= 1
        except KeyError:
            return None

        try:
            sheets_per_pattern[pat1, s2] += 1
        except KeyError:
            sheets_per_pattern[pat1, s2] = 1

        try:
            sheets_per_pattern[pat2, s1] += 1
        except KeyError:
            sheets_per_pattern[pat2, s1] = 1

        if sheets_per_pattern[pat1, s1] < 0 or \
            sheets_per_pattern[pat2, s2] < 0 or \
            sheets_per_pattern[pat1, s2] >= solution.stocks[pat1].ub_sheet[s2] or \
            sheets_per_pattern[pat2, s1] >= solution.stocks[pat2].ub_sheet[s1]:
            return None

        return sheets_per_pattern

    def create_initial_population(self):
        initial_sheets = [Sheet(s.width, s.height, s.demand) for s in self.products]
        new_stocks, sheets_per_pattern = pack_rectangles(self.stocks, initial_sheets)
        fitness, prints_per_pattern = self.solve_LP(new_stocks, sheets_per_pattern, self.products)
        initial_solution = Solution(new_stocks, sheets_per_pattern, prints_per_pattern, fitness)
        # return initial_solution
        initial_population = []
        for _ in range(self.pop_size):
            initial_population.append(self.random_walk(initial_solution))
        return initial_population

    def random_walk(self, initial_solution):
        current_solution = initial_solution
        for _ in range(self.random_walk_steps):
            new_solution = self.choose_neighbor(current_solution)
            if new_solution != None:
                current_solution = new_solution
        # print("SpP last:", current_solution.sheets_per_pattern)
        while True:
            try:
                fitness, prints_per_pattern = self.solve_LP(current_solution.stocks, current_solution.sheets_per_pattern, self.products)
                break
            except:
                new_solution = self.choose_neighbor(current_solution)
                if new_solution != None:
                    current_solution = new_solution
        current_solution.fitness = fitness
        current_solution.prints_per_pattern = prints_per_pattern
        return current_solution

    def update_best_solution(self, population):
        best_solution = population[0]
        for solution in population:
            if solution.fitness < best_solution.fitness:
                best_solution = solution
        return best_solution

    def roulette_wheel_rank_based_selection(self, population):
        solutions = [(solution.fitness, solution) for solution in population]
        solutions.sort()
        n = len(population)
        fitness = [i for i in range(n,0,-1)]
        # Use the gauss formula to get the sum of all ranks (sum of integers 1 to N).
        total_fit = n * (n + 1) / 2
        relative_fitness = [f / total_fit for f in fitness]
        probabilities = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

        chosen = []
        for _ in range(self.roulette_pop):
            r = np.random.rand()
            for (i, pair) in enumerate(solutions):
                if r <= probabilities[i]:
                    _ , solution = pair
                    chosen.append(solution)
                    break
        return chosen

    def bests_solution_reproduction(self, population):
        solutions = [(solution.fitness, solution) for solution in population]
        solutions.sort()
        return [solution for (_, solution) in solutions[:self.no_best_solutions]]


    def choose_neighbor(self, solution):
        operators = [self.add, self.remove, self.move, self.swap] #[self.move, self.swap]
        if (len(solution.stocks) == 1):
            operators = [self.add, self.remove]
        operator = operators[np.random.randint(0, len(operators) - 1)]
        sheets_per_pattern = operator(solution)
        # Nếu toán tử không thể được áp dụng
        if sheets_per_pattern == None:
            return None
        # Check feasible of new solution
        bins = []
        for i in range(len(solution.stocks)):
            sheets = [Sheet(s.width, s.height, sheets_per_pattern[i, j]) for (j,s) in enumerate(self.products) if (i,j) in sheets_per_pattern]
            new_stocks, _ = pack_rectangles(solution.stocks, sheets, i)
            if new_stocks == []:
                return None
            bins+=new_stocks
        neighbor = Solution(bins, sheets_per_pattern)
        return neighbor
    
    def crossover(self, Population):
        def select_patterns(parent):
            patterns = [(stock.free_area, stock, stock.idx) for stock in parent.stocks]
            patterns.sort()

            # select a number between the [25%, 50%] of the patterns (patterns_len must be >= 4 since 0.25 * 4 = 1)
            if len(patterns) >= 4: count_to_select = np.random.randint(np.ceil(0.25 * len(patterns)), np.floor(0.5 * len(patterns))) 
            else: count_to_select = np.random.randint(0, len(patterns)) 
            return {frozenset((stock.free_area, r.width, r.height, r.position, r.rotated, idx) for r in stock.cuts) for _, stock, idx in patterns[:count_to_select]}

        r1, r2 = two_randoms(len(Population) - 1)
        parent1, parent2 = Population[r1], Population[r2]
        set_patterns1 = select_patterns(parent1)
        set_patterns2 = select_patterns(parent2)
        frozensets = set_patterns1 | set_patterns2

        def sort_frozenset_pairs(frozen_set): 
            sorted_list = sorted(frozen_set, key=lambda x: x[0]) # Sắp xếp theo giá trị đầu tiên trong tuple 
            return frozenset(sorted_list) # Sắp xếp các cặp trong từng frozenset 
        
        sorted_frozensets = [sort_frozenset_pairs(fs) for fs in frozensets]
        all_selected_patterns = [Stock(width=stock.width, height=stock.height, lb_patterns=stock.lb_patterns \
                                       , ub_sheet=stock.ub_sheet, idx=stock.idx, position=stock.position) for stock in self.stocks]
        can_use = [True for _ in range(len(self.stocks))]
        covered_sheets = set()
        for fz in sorted_frozensets:
            flag = -1
            for _,w, h, pos, r, idx in fz:
                if can_use[idx] == False: break
                flag = idx
                if r:
                    all_selected_patterns[idx].add_cut(FixedRectangle(h, w, pos, r))
                    covered_sheets.add((h, w))
                else:
                    all_selected_patterns[idx].add_cut(FixedRectangle(w, h, pos, r))
                    covered_sheets.add((w, h))
            if flag != -1:
                can_use[flag] = False
        sheets_to_process = {(s.width, s.height) for s in self.products} - covered_sheets
        sheets_to_process = [Sheet(w, h, 1) for (w,h) in sheets_to_process]
        placement = []
        for i in range(len(all_selected_patterns)):
            if can_use[i] == True: 
                placement.append(all_selected_patterns[i])
        newplacement, _ = pack_rectangles(placement, sheets_to_process)
        j = 0
        patterns = []
        for i in range(len(all_selected_patterns)):
            if can_use[i]: 
                patterns.append(newplacement[j])
                j +=1
            else: patterns.append(all_selected_patterns[i])
        sorted_stocks = sorted(patterns, key=lambda stock: stock.idx)
        bins = [None] * len(patterns) 
        for i, stock in enumerate(sorted_stocks): bins[i] = stock
        sheets_per_patterns = { }
        sheets_idx = { (s.width, s.height):i for i,s in enumerate(self.products) }
        j = -1
        for bin in bins:
            if bin.free_area < (bin.width * bin.height):
                j += 1
            for cut in bin.cuts:
                i = sheets_idx[cut.width, cut.height] if not cut.rotated else sheets_idx[cut.height, cut.width]
                try:
                    sheets_per_patterns[j,i] += 1
                except KeyError:
                    sheets_per_patterns[j,i] = 1
        try:
            fitness, prints_per_pattern = self.solve_LP(bins, sheets_per_patterns, self.products)
        except:
            return self.crossover(Population)
        off_spring = Solution(bins, sheets_per_patterns, prints_per_pattern, fitness)
        return self.hill_climbing(off_spring)

    def mutation(self, population):
        parent = population[np.random.randint(0, len(population)-1)]
        offspring = self.random_walk(parent)
        offspring = self.hill_climbing(offspring)
        return offspring


    def hill_climbing(self, solution):
        current_solution = solution

        while True:
            best_neighbor = current_solution
            for _ in range(self.hill_climbing_neighbors):
                new_neighbor = self.choose_neighbor(current_solution)
                if new_neighbor != None:
                    try:
                        fitness, prints_per_pattern = self.solve_LP(new_neighbor.stocks, new_neighbor.sheets_per_pattern, self.products)
                    except:
                        continue
                    new_neighbor.fitness = fitness
                    new_neighbor.prints_per_pattern = prints_per_pattern
                    if new_neighbor.fitness < best_neighbor.fitness:
                        best_neighbor = new_neighbor

            if best_neighbor.fitness < current_solution.fitness:
                current_solution = best_neighbor
            else:
                break

        return current_solution

    def genetic_algorithm(self):
        # best_known = self.create_initial_population()
        current_generation = self.create_initial_population()
        best_known = self.update_best_solution(current_generation)
        for k in range(self.no_generations):
            if self.stopped:
                return best_known

            intermediate_generation = self.roulette_wheel_rank_based_selection(current_generation)
            current_generation = self.bests_solution_reproduction(current_generation)
            for i in range(len(current_generation), self.pop_size):
                if np.random.rand() < self.prob_crossover:
                    current_generation.append(self.crossover(intermediate_generation))
                else:
                    current_generation.append(self.mutation(intermediate_generation))
            best_known = self.update_best_solution(current_generation)
        best_known = self.hill_climbing(best_known)
        # self.clean_solution(best_known)

        return best_known

    def print_population(self, population):
        result = ''
        for idx, solution in enumerate(population):
            result += f'Solution #{idx + 1}:\n{solution}\n\n'
        return result

    def clean_solution(self, solution):
        updated_prints_per_pattern = {}
        counter = 0
        updated_bins = []
        for i, bin in enumerate(solution.stocks):
            if bin.free_area < (bin.width * bin.height):
                if solution.prints_per_pattern[f'x{i}'] != 0:
                    updated_prints_per_pattern[f'x{counter}'] = solution.prints_per_pattern[f'x{i}']
                    updated_bins.append(bin)
                    counter += 1
        solution.stocks = updated_bins
        solution.prints_per_pattern = updated_prints_per_pattern


