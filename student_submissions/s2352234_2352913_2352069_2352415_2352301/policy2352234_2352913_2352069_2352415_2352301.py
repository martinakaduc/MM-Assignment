from policy import Policy
import numpy as np

# 1: First Fit Decreasing 2: Maximize Remaining Rectangle
class Policy2352234_2352913_2352069_2352415_2352301(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id 

        self.last_prod_w = 0
        self.last_prod_h = 0
        self.last_stock_idx = 0

    def get_action(self, observation, info):
        if self.policy_id == 1:
            return self.first_fit_decreasing_action(observation, info)
        elif self.policy_id == 2:
            # First Fit Decreasing
            return self.maximize_remaining_rectangle_action(observation, info)

    ############################################################################################################
    # First Fit Decreasing    
    def first_fit_decreasing_action(self, observation, info):
        # Student code here
        list_prods = observation["products"]

        prod_size = [0, 0]
        stock_idx = -1
        pos_x, pos_y = 0, 0

        sorted_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)
        sorted_stock_incidies = self.sort_stock(observation)

        for prod in sorted_prods:
            if prod["quantity"] > 0:
                return self.get_action_for_product(prod, observation, sorted_stock_incidies)

        return {"stock_idx": stock_idx, "size": prod_size, "position": (pos_x, pos_y)}

    def area(self, stock):
        stock_w, stock_h = self._get_stock_size_(stock)

        return stock_w * stock_h

    def sort_stock(self, observation):
        sorted_stock_incidies = [i for i in range(len(observation["stocks"]))]
        sorted_stock_incidies = sorted(sorted_stock_incidies, key=lambda x: self.area(observation["stocks"][x]), reverse=True)
        return sorted_stock_incidies
    
    def get_action_for_product(self, prod, observation, stock_incidies):
        prod_size = prod["size"]
        begin_stock_idx_ = 0

        # for time improvement
        if ((self.last_prod_w == prod_size[0] 
                and self.last_prod_h == prod_size[1]) 
            or (self.last_prod_w == prod_size[1] 
                and self.last_prod_h == prod_size[0])):
            begin_stock_idx_ = self.last_stock_idx

        for i in range(begin_stock_idx_, len(stock_incidies)):
            stock = observation["stocks"][stock_incidies[i]]
            stock_w, stock_h = self._get_stock_size_(stock)
            prod_w, prod_h = prod_size

            if stock_w >= prod_w and stock_h >= prod_h:
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            self.last_prod_w = prod_size[0]
                            self.last_prod_h = prod_size[1]
                            self.last_stock_idx = i
                            return {"stock_idx": stock_incidies[i], "size": prod_size, "position": (x, y)}

            if stock_w >= prod_h and stock_h >= prod_w:
                for x in range(stock_w - prod_h + 1):
                    for y in range(stock_h - prod_w + 1):
                        if self._can_place_(stock, (x, y), prod_size[::-1]):
                            self.last_prod_w = prod_size[1]
                            self.last_prod_h = prod_size[0]
                            self.last_stock_idx = i
                            return {"stock_idx": stock_incidies[i], "size": prod_size[::-1], "position": (x, y)}
        
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    ############################################################################################################
    # Maximize Remaining Rectangle
    def maximize_remaining_rectangle_action(self, observation, info):
        list_prods = [prod for prod in observation["products"] if prod["quantity"] > 0]
        sorted_prods = sorted(list_prods, key=lambda x: x["size"][0] * x["size"][1], reverse=True)

        stock_incidies = [i for i in range(len(observation["stocks"]))]
        stock_incidies = self.sort_stock(observation) # sort by number of empty cells

        for prod in sorted_prods:
            prod_size = prod["size"]

            if prod["size"][0] > prod["size"][1]:
                prod_size = prod["size"][::-1]

            begin_stock_idx = 0
            if (prod_size[0] == self.last_prod_w and prod_size[1] == self.last_prod_h) or (prod_size[0] == self.last_prod_h and prod_size[1] == self.last_prod_w):
                begin_stock_idx = self.last_stock_idx

            for i in range(begin_stock_idx, len(stock_incidies)):
                stock = observation["stocks"][stock_incidies[i]]
                stock_w, stock_h = self._get_stock_size_(stock)
                prod_w, prod_h = prod_size

                feasible_positions = []

                if stock_w >= prod_w and stock_h >= prod_h:
                    visited_y = []
                    for x in range(stock_w - prod_w + 1):
                        for y in range(stock_h - prod_h + 1):
                            if y in visited_y:
                                break
                            if self._can_place_(stock, (x, y), prod_size):
                                visited_y.append(y)
                                feasible_positions.append({"pos" : (x, y), "size" : prod_size})
                                break
                            
                if stock_w >= prod_h and stock_h >= prod_w:
                    visited_y = []
                    for x in range(stock_w - prod_h + 1):
                        for y in range(stock_h - prod_w + 1):
                            if y in visited_y:
                                break
                            if self._can_place_(stock, (x, y), prod_size[::-1]):
                                visited_y.append(y)
                                feasible_positions.append({"pos" : (x, y), "size" : prod_size[::-1]})
                                break

                if feasible_positions:
                    best_position = self.get_best_position(stock, feasible_positions, prod_size)
                    self.last_prod_w = prod_size[0]
                    self.last_prod_h = prod_size[1]
                    self.last_stock_idx = i
                    return {"stock_idx": stock_incidies[i], "size": best_position["size"], "position": best_position["pos"]}
                            
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}
    
    def get_best_position(self, stock, feasible_positions, stock_size):
        stock_w, stock_h = stock_size
        best_position = feasible_positions[0]
        best_area = 0

        for pos in feasible_positions:
            x, y = pos["pos"]
            width, height = pos["size"]
            stock[x : x + width, y : y + height] = 0
            _, largest_area = self.largest_free_rectangles(stock)
            stock[x : x + width, y : y + height] = -1

            if largest_area > best_area:
                best_area = largest_area
                best_position = pos
        return best_position
    
    def largest_free_rectangles(self, stock):
        """
        Tính toán diện tích hình chữ nhật còn trống lớn nhất.
        
        Args:
            stock (np.ndarray): Ma trận stock.
            
        Returns:
            int: Diện tích của hình chữ nhật lớn nhất.
        """
        rows, cols = stock.shape
        visited = np.zeros_like(stock, dtype=bool)
        rectangle_count = 0
        largest_area = 0
        
        for i in range(rows):
            for j in range(cols):
                if stock[i, j] == -1 and not visited[i, j]:
                    # Tìm kích thước của hình chữ nhật bắt đầu từ (i, j)
                    width = 0
                    while j + width < cols and stock[i, j + width] == -1 and not visited[i, j + width]:
                        width += 1

                    height = 0
                    while i + height < rows:
                        # Kiểm tra dòng hiện tại có đủ chiều rộng không
                        for k in range(width):
                            if stock[i + height, j + k] >= 0 or visited[i + height, j + k]:
                                break
                        else:
                            height += 1
                            continue
                        break
                    # Đánh dấu các ô trong hình chữ nhật đã được đếm
                    for x in range(i, i + height):
                        for y in range(j, j + width):
                            visited[x, y] = True
                    rectangle_count += 1

                    area = height * width
                    if area > largest_area:
                        largest_area = area
        return rectangle_count, largest_area
