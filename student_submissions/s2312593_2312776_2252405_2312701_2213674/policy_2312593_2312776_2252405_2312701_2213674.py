# # # # # # # # # # # # # # # # # # # # # # # #
# Nhóm 58
# Trưởng nhóm: Nguyễn Thành Phát (dev)
#
# Các thành viên:
# - Nguyễn Tấn Phước (dev)
# - Nguyễn Minh Phúc
# - Nguyễn Phước Trọng
# - Nguyễn Tuấn Kiệt
# # # # # # # # # # # # # # # # # # # # # # # #

from policy import Policy
import numpy as np
import copy as cp
import time

class Policy_2312593_2312776_2252405_2312701_2213674(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2, 3], "Policy ID must be 1, 2 or 3"

        # Student code here
        if policy_id == 1:
            self.Policy = FirstFitDecreasingHeuristic()
        elif policy_id == 2:
            self.Policy = ModifiedGreedy()
        elif policy_id == 3:
            self.Policy = FirstFitDecreasing()
            

    def get_action(self, observation, info):
        return self.Policy.get_action(observation, info)
    
    def evaluate(self):
        return self.Policy.evaluate()
    
# ! Đây không là giải thuật nhóm tập trung !
# Áp dụng giải thuật modified greedy
# Thực hiện sắp xếp và chia nhóm các vật liệu theo chiều dài, chiều rộng chẵn lẻ
# Thao tác cắt tương tự như Greedy
#
class ModifiedGreedy(Policy):
    def __init__(self):
        self.first = True

        self.m_used_surface = 0
        self.m_filled_surface = 0
        self.m_used_stock = 0

        self.m_sorted_stock_index = 0
        self.m_sorted_product_index = 0

        self.odd_odd_stock_index = []
        self.odd_even_stock_index = []
        self.even_odd_stock_index = []
        self.even_even_stock_index = []

        self.total_time = 0

        pass

    def reset(self):

        self.first = True

        self.m_used_surface = 0
        self.m_filled_surface = 0
        self.m_used_stock = 0

        self.m_sorted_product_index = []

        self.odd_odd_stock_index = []
        self.odd_even_stock_index = []
        self.even_odd_stock_index = []
        self.even_even_stock_index = []

        self.total_time = 0

        pass

    def evaluate(self):
        print("[----------==========| EVALUATE MODIFIED GREEDY |==========----------]")
        print(" - Stocks used:    ", self.m_used_stock)
        print(" - Used Surface:   ", self.m_used_surface)
        print(" - Waste Surface:  ", self.m_used_surface - self.m_filled_surface)
        print(" - Filled Surface: ", self.m_filled_surface)
        print(" - Waste Percent:  ", (1-self.m_filled_surface/self.m_used_surface)*100, "%")
        print(" - Total Time:     ", self.total_time, "s")
        print("[----------==========| EVALUATE MODIFIED GREEDY |==========----------]")
        pass

    def init_indices(self, list_stocks, list_prods):
        
        sorted_products = sorted(list_prods, key=lambda product: product['size'][0] * product['size'][1], reverse=True)
        product_indies = []
        for s_st in range(len(sorted_products)):
            for st in range(len(list_prods)):
                if (np.shape(list_prods[st]['size'])==np.shape(sorted_products[s_st]['size'])) and (np.all(list_prods[st]['size']==sorted_products[s_st]['size'])):
                    product_indies.append(st)
        self.m_sorted_product_index = product_indies

        sorted_stocks = sorted(list_stocks, key=lambda stock: np.sum(np.any(stock != -2, axis=1)) * np.sum(np.any(stock != -2, axis=0)), reverse=True)
        stock_indies = []
        for s_st in range(len(sorted_stocks)):
            for st in range(len(list_stocks)):
                if (np.shape(list_stocks[st])==np.shape(sorted_stocks[s_st])) and (np.all(list_stocks[st]==sorted_stocks[s_st])):
                    stock_indies.append(st)
        sorted_stock_index = stock_indies

        for idx in sorted_stock_index:
            stock = list_stocks[idx]
            size = self._get_stock_size_(stock)
            if (size[0]%2==0):
                if (size[1]%2==0):
                    self.even_even_stock_index.append(idx)
                else:
                    self.even_odd_stock_index.append(idx)
            else:
                if (size[1]%2==0):
                    self.odd_even_stock_index.append(idx)
                else:
                    self.odd_odd_stock_index.append(idx)

        pass

    def get_action(self, observation, info):

        start_time = time.time()
        list_prods = observation["products"]
        list_stocks = observation["stocks"]

        # Lần đầu hàm action được gọi
        if (self.first):
            self.reset()
            self.init_indices(list_stocks, list_prods)
            self.first = False

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        
        for pr_idx in self.m_sorted_product_index:
            prod = list_prods[pr_idx]
            prod_size = prod['size']
            if prod["quantity"] > 0:
                
                # tìm vật liệu để cắt tùy theo sản phẩm
                toCheck = []
                if (prod_size[0]%2==0):
                    if (prod_size[1]%2==0):
                        toCheck = self.even_even_stock_index
                    else:
                        toCheck = self.even_odd_stock_index
                else:
                    if (prod_size[1]%2==0):
                        toCheck = self.odd_even_stock_index
                    else:
                        toCheck = self.odd_odd_stock_index

                # duyệt qua các vật liệu
                for st_idx in toCheck:
                    stock = list_stocks[st_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod['size']

                    # thu thập dữ liệu để đánh giá
                    used = np.any(stock >= 0)
                    surface = stock_w * stock_h
                    filled = np.sum(stock >= 0)

                    if((stock_w < prod_w or stock_h < prod_h) and (stock_h < prod_w or stock_w < prod_h)):
                        continue
                    
                    # duyệt qua từng vị trí trong vật liệu
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                prod_size = (prod_w, prod_h)
                                stock_idx = st_idx

                                if (not used):
                                    self.m_used_surface += + surface
                                    self.m_used_stock += 1
                                
                                prod_surface = prod_w * prod_h
                                self.m_filled_surface += prod_surface
                                filled += prod_surface

                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = st_idx
                        break

                    # xử lý trường hợp xoay
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                                prod_size = (prod_h, prod_w)
                                stock_idx = st_idx

                                if (not used):
                                    self.m_used_surface += + surface
                                    self.m_used_stock += 1
                                
                                prod_surface = prod_w * prod_h
                                self.m_filled_surface += prod_surface
                                filled += prod_surface

                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = st_idx
                        break

                if pos_x is not None and pos_y is not None:
                    break
        
        # tính toán thời gian
        end_time = time.time()
        self.total_time += end_time - start_time

        # reset policy khi cần
        amount_of_products = 0
        for prod in list_prods:
            amount_of_products += prod['quantity']
        if (amount_of_products==1):
            self.first = True

        # giá trị trả về
        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}        

# ! class được nhóm đầu tư nhiều nhất là first fit decreasing heuristic !
# Thực hiện sắp xếp các khối vật liệu theo chiều giảm dần diện tích
# Cắt các sản phẩm trong môi trường ảo của policy
# Xử lý vật liệu đã cắt. Di dời các sản phẩm ảo (bao hết sản phẩm thực tế đã cắt) sang vật liệu nhỏ hơn
class FirstFitDecreasingHeuristic(Policy):
    def __init__(self):
        # data variable
        self.stocks = []
        self.products = []

        # other
        self.num_stocks = 0
        self.num_products = 0
        self.amount_of_products = 0

        # action variable
        self.action_list = [[]]
        self.first_action = True
        self.action_called = 0

        # index variable
        self.stocks_indices = []
        self.products_indices = []

        # evaluate variable
        self.cutted_stocks = []
        self.total_time = 0
        
    def reset(self):
        # data variable
        self.stocks = []
        self.products = []

        # other
        self.num_stocks = 0
        self.num_products = 0
        self.amount_of_products = 0

        # action variable
        self.action_list = [[]]
        self.first_action = True
        self.action_called = 0

        # index variable
        self.stocks_indices = []
        self.products_indices = []

        # evaluate variable
        self.cutted_stocks = []
        self.total_time = 0

    # Thực hiện cắt hay "paint" trong môi trường ảo
    def paint(self, stock_idx, prod_idx, position, custom_size):
        width, height = custom_size
        self.cutted_stocks[stock_idx] = 1
        self.products[prod_idx]["quantity"] -= 1

        x, y = position
        stock = self.stocks[stock_idx]
        stock[x : x + width, y : y + height] = prod_idx

    # hàm get_action chính
    def get_action(self, observation, info):
        # Lấy thời gian bắt đầu
        start_time = time.time()

        # Lần dầu chiếm nhiều thời gian nhất
        if self.first_action:
            self.reset()
            self.init_variable(observation["stocks"], observation["products"])
            self.first_action = False
            
            # cắt toàn bộ sản phẩm trong môi trường ảo
            for pr_idx in self.products_indices:
                prod = self.products[pr_idx]
                while prod["quantity"] > 0:
                    prod_size = prod["size"]

                    for st_idx in self.stocks_indices:
                        stock = self.stocks[st_idx]
                        stock_w, stock_h = self._get_stock_size_(stock)
                        prod_w, prod_h = prod_size

                        if((stock_w < prod_w or stock_h < prod_h) and (stock_h < prod_w or stock_w < prod_h)):
                            continue
                        
                        pos_x, pos_y = None, None
                        
                        # Dành cho không xoay
                        if prod_w <= stock_w or prod_h <= stock_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        self.paint(st_idx, pr_idx, (pos_x, pos_y), prod_size)
                                        # thêm bước thêm action vào 1 danh sách
                                        self.action_list[st_idx].append({"stock_idx": st_idx, "size": prod_size, "position": (pos_x, pos_y), "product_idx": pr_idx})
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break
                                
                            if pos_x is not None and pos_y is not None:
                                break
                        
                        # Dành cho xoay
                        if prod_h <= stock_w or prod_w <= stock_h:
                            new_size = prod_h, prod_w
                            
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock, (x, y), new_size):
                                        pos_x, pos_y = x, y
                                        self.paint(st_idx, pr_idx, (pos_x, pos_y), new_size)
                                        # thêm bước thêm action vào 1 danh sách
                                        self.action_list[st_idx].append({"stock_idx": st_idx, "size": new_size, "position": (pos_x, pos_y), "product_idx": pr_idx})
                                        break
                                if pos_x is not None and pos_y is not None:
                                    break
                                
                            if pos_x is not None and pos_y is not None:
                                break
            
            # Xử lý sau khi đã cắt xong
            sorted_list = self.sort_stock_indices_by_bounding_box()
            for ele in reversed(sorted_list):
                st_idx = ele[0]
                if (self.cutted_stocks[st_idx]==0):
                    continue
                
                stock = self.stocks[st_idx]
                temp_w, temp_h = ele[1], ele[2]
                size = self._get_stock_size_(stock)

                # cắt thử các stock nhỏ hơn
                for st_idx2 in reversed(self.stocks_indices):
                    check_stock = self.stocks[st_idx2]
                    check_size = self._get_stock_size_(check_stock)
                    
                    if (check_size[0] * check_size[1] < temp_w * temp_h):
                        break

                    if (check_size[0] * check_size[1] >= size[0] * size[1]):
                        break

                    if self._can_place_(check_stock, (0,0), (temp_w, temp_h)):
                        self.copyAtoB(st_idx, (0,0), st_idx2, (0,0), (temp_w, temp_h), False)
                        break
                    
        
        # Lấy thời gian kết thúc
        end_time = time.time()
        self.total_time += end_time - start_time

        # Lấy action
        return self.get_action_from_list()

    def calculate_bounding_box(self, stock):
        # Lấy chỉ số các phần tử không âm
        rows, cols = np.where(stock >= 0)

        if rows.size == 0 or cols.size == 0:  # Nếu không có sản phẩm nào
            return 0, 0

        # Tìm chỉ số hàng và cột nhỏ nhất, lớn nhất
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()

        # Tính kích thước bao phủ
        width = max_row - min_row + 1
        height = max_col - min_col + 1

        return width, height
    
    
    # Sắp xếp danh sách các chỉ số stocks chưa bị cắt dựa trên diện tích bounding box (sản phẩm ảo) từ bé đến lớn.
    def sort_stock_indices_by_bounding_box(self):
        stocks_with_bounding = []

        for idx, stock in enumerate(self.stocks):
            if self.cutted_stocks[idx] == 1:
                width, height = self.calculate_bounding_box(stock)
                stocks_with_bounding.append((idx, width, height))
        
        sorted_stocks = sorted(
            stocks_with_bounding, key=lambda x: x[1] * x[2]
        )

        return sorted_stocks

    # copy từ stock A sang stock B
    def copyAtoB(self, idxA, posA, idxB, posB, size, rotate):
        width, height = size
        x_A, y_A = posA
        x_B, y_B = posB
        
        for i in range(width):
            for j in range(height):
                self.stocks[idxB][x_B+i][y_B+j] = self.stocks[idxA][x_A+i][y_A+j]
                self.stocks[idxA][x_A+i][y_A+j] = -1

        
        for ac in self.action_list[idxA]:
            ac_copy = cp.copy(ac)  # Tạo bản sao của ac
            ac_copy['stock_idx'] = idxB  # Thay đổi stock_idx trong bản sao
            if rotate:
                ac_copy['size'][0], ac_copy['size'][1] = ac_copy['size'][1], ac_copy['size'][0]  # Hoán đổi kích thước nếu xoay
            self.action_list[idxB].append(ac_copy)  # Thêm bản sao vào danh sách mới
        
        self.action_list[idxA].clear()
        self.cutted_stocks[idxB] = 1
        self.cutted_stocks[idxA] = 0

    # Khởi tạo các biến thành viên
    def init_variable(self, list_stocks, list_products):
        self.stocks = cp.deepcopy(list_stocks)
        self.products = cp.deepcopy(list_products)

        for prod in self.products:
            self.amount_of_products+=prod['quantity']
        self.num_products = len(list_products)
        self.num_stocks = len(list_stocks)
        self.cutted_stocks = np.full((self.num_stocks,), fill_value=0, dtype=int)

        sorted_products = sorted(self.products, key=lambda product: product['size'][0] * product['size'][1], reverse=True)
        product_indies = []
        for s_st in range(len(sorted_products)):
            for st in range(len(self.products)):
                if (np.shape(self.products[st]['size'])==np.shape(sorted_products[s_st]['size'])) and (np.all(self.products[st]['size']==sorted_products[s_st]['size'])):
                    product_indies.append(st)
        self.products_indices = product_indies

        sorted_stocks = sorted(self.stocks, key=lambda stock: np.sum(np.any(stock != -2, axis=1)) * np.sum(np.any(stock != -2, axis=0)), reverse=True)
        stock_indies = []
        for s_st in range(len(sorted_stocks)):
            for st in range(len(self.stocks)):
                if (np.shape(self.stocks[st])==np.shape(sorted_stocks[s_st])) and (np.all(self.stocks[st]==sorted_stocks[s_st])):
                    stock_indies.append(st)
        self.stocks_indices = stock_indies
        
        self.action_list = [[] for _ in range(self.num_stocks)]
        
    # Lấy từng action từ danh sách đã lưu
    def get_action_from_list(self):
        # lấy Action
        flattened_action_list = [action for sublist in self.action_list for action in sublist]
        action = flattened_action_list[self.action_called]
        # xem đã đủ hay chưa, nếu đã lấy hết action, thì set first_action=True để reset cho dữ liệu mới
        if (self.action_called==self.amount_of_products-1):
            self.first_action = True
        else:
            self.action_called+=1
        return action
    
    # Đánh giá giải thuật
    def evaluate(self):
        # số stock sử dụng
        amount_stocks = np.sum(self.cutted_stocks)
        # tính diện tích đã dùng và đã cắt (filled)
        used = 0
        filled = 0
        for st_idx in self.stocks_indices:
            if (self.cutted_stocks[st_idx]==1):
                stock = self.stocks[st_idx]
                size = self._get_stock_size_(stock)

                filled += np.sum(stock>=0)
                used += size[0] * size[1]

        # hiển thị
        print("[----------==========| EVALUATE FIRST FIT DECREASING HEURISTIC |==========----------]")
        print(" - Stocks used:    ", amount_stocks)
        print(" - Used Surface:   ", used)
        print(" - Waste Surface:  ", used - filled)
        print(" - Filled Surface: ", filled)
        print(" - Waste Percent:  ", (1-filled/used)*100, "%")
        print(" - Total Time:     ", self.total_time, "s")
        print("[----------==========| EVALUATE FIRST FIT DECREASING HEURISTIC |==========----------]")

# ! Đây không là giải thuật nhóm tập trung !
# Thực hiện sắp xếp theo thứ tự giảm dần diện tích
# Cắt vật liệu theo Greedy
class FirstFitDecreasing(Policy):
    def __init__(self):

        self.first = True

        self.m_used_surface = 0
        self.m_filled_surface = 0
        self.m_used_stock = 0

        self.m_sorted_stock_index = 0
        self.m_sorted_product_index = 0

        self.total_time = 0

        pass

    def reset(self):

        self.first = True

        self.m_used_surface = 0
        self.m_filled_surface = 0
        self.m_used_stock = 0

        self.m_sorted_stock_index = 0
        self.m_sorted_product_index = 0

        self.total_time = 0

        pass

    def evaluate(self):
        print("[----------==========| EVALUATE BASIC FIRST FIT DECREASING |==========----------]")
        print(" - Stocks used:    ", self.m_used_stock)
        print(" - Used Surface:   ", self.m_used_surface)
        print(" - Waste Surface:  ", self.m_used_surface - self.m_filled_surface)
        print(" - Filled Surface: ", self.m_filled_surface)
        print(" - Waste Percent:  ", (1-self.m_filled_surface/self.m_used_surface)*100, "%")
        print(" - Total Time:     ", self.total_time, "s")
        print("[----------==========| EVALUATE BASIC FIRST FIT DECREASING |==========----------]")
        pass

    def init_indices(self, list_stocks, list_prods):
        
        sorted_products = sorted(list_prods, key=lambda product: product['size'][0] * product['size'][1], reverse=True)
        product_indies = []
        for s_st in range(len(sorted_products)):
            for st in range(len(list_prods)):
                if (np.shape(list_prods[st]['size'])==np.shape(sorted_products[s_st]['size'])) and (np.all(list_prods[st]['size']==sorted_products[s_st]['size'])):
                    product_indies.append(st)
        self.m_sorted_product_index = product_indies


        sorted_stocks = sorted(list_stocks, key=lambda stock: np.sum(np.any(stock != -2, axis=1)) * np.sum(np.any(stock != -2, axis=0)), reverse=True)
        stock_indies = []
        for s_st in range(len(sorted_stocks)):
            for st in range(len(list_stocks)):
                if (np.shape(list_stocks[st])==np.shape(sorted_stocks[s_st])) and (np.all(list_stocks[st]==sorted_stocks[s_st])):
                    stock_indies.append(st)
        self.m_sorted_stock_index = stock_indies

        pass

    def get_action(self, observation, info):

        start_time = time.time()
        list_prods = observation["products"]
        list_stocks = observation["stocks"]

        # Descending
        if (self.first):
            self.reset()
            self.init_indices(list_stocks, list_prods)
            self.first = False

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0
        
        for pr_idx in self.m_sorted_product_index:
            prod = list_prods[pr_idx]
            prod_size = prod['size']
            if prod["quantity"] > 0:
                # Loop through all stocks
                for st_idx in self.m_sorted_stock_index:
                    stock = list_stocks[st_idx]
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size

                    # evaluate
                    used = np.any(stock >= 0)
                    surface = stock_w * stock_h
                    filled = np.sum(stock >= 0)

                    if((stock_w < prod_w or stock_h < prod_h) and (stock_h < prod_w or stock_w < prod_h)):
                        continue
                    
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if self._can_place_(stock, (x, y), (prod_w, prod_h)):
                                stock_idx = st_idx
                                prod_size = (prod_w, prod_h)

                                if (not used):
                                    self.m_used_surface += + surface
                                    self.m_used_stock += 1
                                
                                prod_surface = prod_w * prod_h
                                self.m_filled_surface += prod_surface
                                filled += prod_surface

                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = st_idx
                        break

                    # cho xoay
                    pos_x, pos_y = None, None
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if self._can_place_(stock, (x, y), (prod_h, prod_w)):
                                stock_idx = st_idx
                                prod_size = (prod_h, prod_w)

                                if (not used):
                                    self.m_used_surface += + surface
                                    self.m_used_stock += 1
                                
                                prod_surface = prod_w * prod_h
                                self.m_filled_surface += prod_surface
                                filled += prod_surface

                                pos_x, pos_y = x, y
                                break
                        if pos_x is not None and pos_y is not None:
                            break
                    if pos_x is not None and pos_y is not None:
                        stock_idx = st_idx
                        break

                if pos_x is not None and pos_y is not None:
                    break
        
        end_time = time.time()
        self.total_time += end_time - start_time

        amount_of_products = 0
        for prod in list_prods:
            amount_of_products += prod['quantity']
        if (amount_of_products==1):
            self.first = True

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}