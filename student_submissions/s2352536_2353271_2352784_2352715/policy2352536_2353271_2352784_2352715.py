from policy import Policy
import numpy as np

class Policy2352536_2353271_2352784_2352715(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = TLPolicy()
        elif policy_id == 2:
            self.policy = SkylinePolicy()
    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)

class TLPolicy(Policy):
    def __init__(self):
        self.avr_pro = 0
        self.avr_sto = 0
        self.flag = True
        self.final_prod = {}
        self.final_sto = {}
        self.sorted_products = {}
        self.sorted_stocks = {}

    def get_action(self, observation, info):
        # Set flag 
        if self.flag == True:
            # Get products & stocks
            products = observation["products"]
            stocks = observation["stocks"]

            # Get all quantity of products for calculate the average area
            prod_sizes = np.array([prod["size"] for prod in products])
            prod_sizes_quantity = np.array([prod["quantity"] for prod in products])
            result = []
            for i in range(len(prod_sizes)):
                for j in range(int(prod_sizes_quantity[i])):
                    result.append(prod_sizes[i])
            
            # Sort product from small to lagre
            self.sorted_products = sorted(
                products,
                key=lambda p: p["size"][0] * p["size"][1],
                reverse=False
            )

            # Calculate the average area of products
            sum = 0
            for i in range(len(result)):
                sum = sum + result[i][0] * result[i][1]
            self.avr_pro = sum / len(result)

            # Find the products whose area value is closest to the average value of the products
            min_prod = 1000000000
            for prod in self.sorted_products:
                cal = abs(self.avr_pro - prod["size"][0] * prod["size"][1])
                if cal < min_prod:
                    min_prod = cal
            
            # Split the array into pivot, above pivot and below pivot
            pivot_prod = [
                prod["size"][0] * prod["size"][1]
                for prod in self.sorted_products
                if abs(self.avr_pro - prod["size"][0] * prod["size"][1]) == min_prod
            ]
            pre_below_prod = np.array([
                prod for prod in self.sorted_products 
                if prod["size"][0] * prod["size"][1] < pivot_prod[0]
            ])
            around_prod = np.array([
                prod for prod in self.sorted_products
                if abs(self.avr_pro - prod["size"][0] * prod["size"][1]) == min_prod
            ])
            pre_above_prod = np.array([
                prod for prod in self.sorted_products
                if prod["size"][0] * prod["size"][1] > pivot_prod[0]
            ])
            below_prod = sorted(
                pre_below_prod,
                key=lambda p: p["size"][0] * p["size"][1],
                reverse = True
            )
            above_prod = sorted(
                pre_above_prod,
                key=lambda p: p["size"][0] * p["size"][1],
                reverse = True
            )

            # Sort stocks from small to lagre
            self.sorted_stocks = sorted(
                enumerate(stocks),
                key=lambda x: np.sum(np.any(x[1] != -2, axis=1)) * np.sum(np.any(x[1] != -2, axis=0)),
                reverse=False
            )
            
            # Calculate the average area of stocks
            stock_areas = [self._get_stock_size_(s[1])[0] * self._get_stock_size_(s[1])[1] for s in self.sorted_stocks]
            sum_s_stocks = 0
            for i in range(len(stock_areas)):
                sum_s_stocks = sum_s_stocks + stock_areas[i]
            self.avr_sto = sum_s_stocks / len(stock_areas)

            # Find the stocks whose area value is closest to the average value of the stocks
            min_sto = 1000000000
            for sto in self.sorted_stocks:
                cal = abs(self.avr_sto - self._get_stock_size_(sto[1])[0] * self._get_stock_size_(sto[1])[1])
                if cal < min_sto:
                    min_sto = cal

            # Split the array into pivot, above pivot and below pivot
            pivot_sto = [
                self._get_stock_size_(sto[1])[0] * self._get_stock_size_(sto[1])[1]
                for sto in self.sorted_stocks
                if abs(self.avr_sto - self._get_stock_size_(sto[1])[0] * self._get_stock_size_(sto[1])[1]) == min_sto
            ]
            pre_below_sto = ([
                sto
                for sto in self.sorted_stocks 
                if self._get_stock_size_(sto[1])[0] * self._get_stock_size_(sto[1])[1] < pivot_sto[0]
            ])
            around_sto = ([
                sto
                for sto in self.sorted_stocks
                if abs(self.avr_sto - self._get_stock_size_(sto[1])[0] * self._get_stock_size_(sto[1])[1]) == min_sto
            ])
            pre_above_sto = ([
                sto
                for sto in self.sorted_stocks 
                if self._get_stock_size_(sto[1])[0] * self._get_stock_size_(sto[1])[1] > pivot_sto[0]
            ])
            below_sto = sorted(
                pre_below_sto,
                key=lambda x: np.sum(np.any(x[1] != -2, axis=1)) * np.sum(np.any(x[1] != -2, axis=0)),
                reverse = True
            )
            above_sto = sorted(
                pre_above_sto,
                key=lambda x: np.sum(np.any(x[1] != -2, axis=1)) * np.sum(np.any(x[1] != -2, axis=0)),
                reverse = True
            )

            # Consider the conditions for sort order
            if pivot_sto[0] < self.avr_sto:
                self.final_prod = list(around_prod) + list(below_prod) + list(above_prod)
                self.final_sto = list(around_sto) + list(below_sto) + list(above_sto)
            else:
                self.final_prod = list(around_prod) + list(above_prod) + list(below_prod)
                self.final_sto = list(around_sto) + list(above_sto) + list(below_sto)
        self.flag = False
        
        # Loop for cut the product
        visited_stocks = set()
        visited_products = set()
        while len(visited_stocks) < len(self.sorted_stocks) or len(visited_products) < len(self.sorted_products):
            for prod in self.final_prod:
                prod_size = prod["size"]
                prod_area = prod_size[0] * prod_size[1]
                

                for stock in self.final_sto:
                    stock_idx, stock_content = stock
                    stock_w, stock_h = self._get_stock_size_(stock_content)
                    stock_area = stock_w * stock_h

                    # Ignore products or stock that have already been tested
                    if prod["quantity"] == 0 or (stock_idx, tuple(prod_size)) in visited_stocks:
                        continue

                    # Check cutting conditions based on area
                    if (stock_area < self.avr_sto and prod_area < self.avr_pro) or (stock_area >= self.avr_sto and prod_area >= self.avr_pro):
                        if prod["quantity"] > 0 and stock_w >= prod_size[0] and stock_h >= prod_size[1]:
                            for x in range(stock_w - prod_size[0] + 1):
                                for y in range(stock_h - prod_size[1] + 1):
                                    if self._can_place_(stock_content, (x, y), prod_size):
                                        # Mark processed products and stock
                                        visited_stocks.add((stock_idx, tuple(prod_size)))
                                        visited_products.add(tuple(prod_size))
                                        # print(self.final_prod)
                                        return {
                                            "stock_idx": stock_idx,
                                            "size": prod_size,
                                            "position": (x, y)
                                        }
                        if prod["quantity"] > 0 and stock_w >= prod_size[1] and stock_h >= prod_size[0]:
                            for x in range(stock_w - prod_size[1] + 1):
                                for y in range(stock_h - prod_size[0] + 1):
                                    rotated_size = [prod_size[1], prod_size[0]]
                                    if self._can_place_(stock_content, (x, y), rotated_size):
                                        visited_stocks.add((stock_idx, tuple(prod_size)))
                                        visited_products.add(tuple(prod_size))
                                        return {
                                            "stock_idx": stock_idx,
                                            "size": rotated_size,
                                            "position": (x, y)
                                        }
                    # If cutting is not possible, mark the product and move to the next stock
                    visited_stocks.add((stock_idx, tuple(prod_size)))

        # If cutting is not possible, mark the product and move to the next stock
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

class SkylinePolicy(Policy):
    def __init__(self):
        super().__init__()
        self.skyline = []
        self.stock_width = 0
        self.stock_height = 0
        self.height_cache = {}
        self.lengthArr = []
        self.widthArr = []
        self.demandArr = []
        self.N = 0
        self.patterns = []
        self.current_stock_idx = 0
        self.current_pattern_idx = 0

    def generate_efficient_patterns(self, stock_length, stock_width):
        patterns = []
        for i in range(self.N):
            pattern = [0] * self.N
            pattern[i] = 1
            patterns.append(pattern)
        return patterns

    def get_action(self, observation, info):
        if not self.lengthArr:
            self._initialize_problem(observation)

        if self.N == 0:
            return self._get_empty_action()

        stocks = observation["stocks"]
        while self.current_stock_idx < len(stocks):
            stock = stocks[self.current_stock_idx]
            stock_Length, stock_Width = self._get_stock_size_(stock)
            self.stock_width, self.stock_height = stock_Width, stock_Length

            if not self.patterns:
                self.patterns = self.generate_efficient_patterns(stock_Length, stock_Width)

            while self.current_pattern_idx < len(self.patterns):
                pattern = self.patterns[self.current_pattern_idx]
                for pattern_index, count in enumerate(pattern):
                    if count > 0 and self.demandArr[pattern_index] > 0:
                        prod_size = (self.lengthArr[pattern_index], self.widthArr[pattern_index])
                        action = self._try_place_product(stock, prod_size)
                        if action:
                            self.demandArr[pattern_index] -= 1
                            return action

                self.current_pattern_idx += 1

            self.current_stock_idx += 1
            self.current_pattern_idx = 0
            self.patterns = []

        # Reset indices if we've gone through all stocks
        self.current_stock_idx = 0
        self.current_pattern_idx = 0
        return self._get_empty_action()

    def _initialize_problem(self, observation):
        list_prods = sorted(
            observation["products"],
            key=lambda x: x["size"][0] * x["size"][1],
            reverse=True,
        )
        self.lengthArr = [prod["size"][0] for prod in list_prods if prod["quantity"] > 0]
        self.widthArr = [prod["size"][1] for prod in list_prods if prod["quantity"] > 0]
        self.demandArr = [prod["quantity"] for prod in list_prods if prod["quantity"] > 0]
        self.N = len(self.lengthArr)

    def _try_place_product(self, stock, prod_size):
        for x in range(self.stock_width):
            for y in range(self.stock_height):
                if self._can_place_(stock, (x, y), prod_size):
                    return {
                        "stock_idx": self.current_stock_idx,
                        "size": prod_size,
                        "position": (x, y),
                        "rotated": False,
                    }
                elif self._can_place_(stock, (x, y), (prod_size[1], prod_size[0])):
                    return {
                        "stock_idx": self.current_stock_idx,
                        "size": (prod_size[1], prod_size[0]),
                        "position": (x, y),
                        "rotated": True,
                    }
        return None

    def _get_empty_action(self):
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}

    # Skyline-related methods (kept for potential future use)
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
        best_y = float("inf")
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


class FirstFitPolicy(Policy):
    def get_action(self, observation, info):
        # Sort products by area in descending order
        sorted_products = sorted(
            observation["products"],
            key=lambda p: p["size"][0] * p["size"][1],
            reverse=True
        )

        # Sort stocks by available area in ascending order
        sorted_stocks = sorted(
            enumerate(observation["stocks"]),
            key=lambda x: np.sum(np.any(x[1] != -2, axis=1)) * np.sum(np.any(x[1] != -2, axis=0))
        )

        # Iterate through the products and try to place them
        for prod in sorted_products:
            if prod["quantity"] > 0:
                prod_size = prod["size"]
                for stock_idx, stock in sorted_stocks:
                    stock_w, stock_h = self._get_stock_size_(stock)
                    prod_w, prod_h = prod_size
                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), prod_size):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": prod_size,
                                        "position": (x, y)
                                    }
        
        # If no placement found, return a "no-op" action
        return {"stock_idx": -1, "size": [0, 0], "position": (0, 0)}