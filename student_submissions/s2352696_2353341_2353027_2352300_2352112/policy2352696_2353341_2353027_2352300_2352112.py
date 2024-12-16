from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2352696_2353341_2353027_2352300_2352112(Policy):
    def __init__(self, policy_id = 1):
        self.current_patterns = []
        self.dual_values = []
        self.demands = None
        self.stock_size = None
        self.num_products = 0
        self.epsilon = 1e-6
        self.action_queue = []
        self.master_solution = None
        self.stock_idx = -1  # Start from 0
        self.check_100 = False
        self.last_check = False
        self.stock_placed = []
        self.iteration = 0
        self.policy_id = policy_id

    def reset(self):
        self.current_patterns = []
        self.dual_values = []
        self.demands = None
        self.stock_size = None
        self.num_products = 0
        self.epsilon = 1e-6
        self.action_queue = []
        self.master_solution = None
        self.stock_idx = -1
        self.check_100 = False
        self.last_check = False
        self.stock_placed = []
        self.iteration = 0
    def _get_stock_size_(self, stock):
        stock = np.atleast_2d(stock)
        stock_w, stock_h = stock.shape
        return stock_w, stock_h

    def _can_place_(self, stock, position, size):
        xi, yi = position
        w, h = size
        stock_w, stock_h = stock.shape
        if xi + w > stock_w or yi + h > stock_h:
            return False
        if np.any(stock[xi:xi+w, yi:yi+h] != -1):
            return False
        return True

    def _solve_master_problem(self, demand_vector):
        num_patterns = len(self.current_patterns)
        cvec = np.ones(num_patterns)

        A_eq = np.zeros((self.num_products, num_patterns))

        for j, pattern in enumerate(self.current_patterns):
            A_eq[:, j] = pattern['counts']

        b_eq = demand_vector
        result = linprog(cvec, A_eq=A_eq, b_eq=b_eq, method='highs', bounds=(0, None))

        if result.success:
            self.dual_values = result["eqlin"]["marginals"]
            self.master_solution = result["x"]
        else:
            raise ValueError("Master problem did not converge.")

    def _solve_subproblem(self):
        num_products = self.num_products
        stock_w, stock_h = self.stock_size

        dual_values = self.dual_values

        unit_values = []

        for i in range(num_products):
            product = self.demands[i]
            u_i = dual_values[i]
            w_i, h_i = product["size"]
            area_i = w_i * h_i
            unit_value = u_i / area_i if area_i > 0 else 0
            unit_values.append((unit_value, i))

        unit_values.sort(key=lambda x: x[0], reverse=True)

        stock = np.full((stock_w, stock_h), fill_value=-1, dtype=int)

        counts = np.zeros(num_products, dtype=int)

        placements = []

        for unit_value, i in unit_values:
            product = self.demands[i]
            w_i, h_i = product["size"]
            quantity_i = product["quantity"]

            orientations = [(w_i, h_i), (h_i, w_i)] if w_i != h_i else [(w_i, h_i)]

            for _ in range(quantity_i - counts[i]):
                placed = False
                for w, h in orientations:
                    for xi in range(stock_w - w + 1):
                        for yi in range(stock_h - h + 1):
                            if self._can_place_(stock, (xi, yi), (w, h)):
                                stock[xi:xi+w, yi:yi+h] = i
                                placements.append((i, xi, yi, w, h))
                                counts[i] += 1
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break
                if not placed:
                    break

        reduced_cost = 1 - np.dot(dual_values, counts)

        if reduced_cost < -self.epsilon:
            pattern = {
                'counts': counts,
                'placements': placements,
                'unit_values': unit_values
            }
            return pattern
        else:
            return None

    def _get_stock_size_(self, stock):
        # Ensure stock is at least a 2D array
        stock = np.atleast_2d(stock)
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))
        return stock_w, stock_h

    def _get_stock_area_(self, stock):
        stock_w, stock_h = self._get_stock_size_(stock)
        return stock_w * stock_h

    def get_action(self, observation, info):
        if info["trim_loss"] == 1:
            self.reset()
        if self.policy_id == 1:
            while True:
                if self.action_queue:
                    return self.action_queue.pop(0)

                self.iteration += 1
                self.stock_idx = np.random.randint(0, len(observation["stocks"]))
                if self.stock_idx in self.stock_placed:
                    self.iteration -= 1
                    continue
                # Reset current patterns
                self.current_patterns = []
                # Get demands and number of products
                stock_tuple = observation["stocks"]
                stock_list = list(enumerate(stock_tuple))
                # Sort stocks by area in descending order
                #print(self.iteration)
                if self.iteration >= 8 * len(stock_list):
                    self.check_100 = True
                    stock_list = sorted(
                        stock_list,
                        key=lambda x: self._get_stock_area_(x[1]),
                        reverse=False
                    )
                    self.stock_idx = 0
                    while True:
                        if self.stock_idx in self.stock_placed:
                            self.stock_idx += 1
                            continue
                        else:
                            break
                self.demands = observation["products"]
                self.num_products = len(self.demands)
                demands_vector = np.array([product["quantity"] for product in self.demands])

                if not self.current_patterns:
                    for i in range(self.num_products):
                        if(self.demands[i]["quantity"] == 0):
                            continue
                        counts = np.zeros(self.num_products, dtype=int)
                        counts[i] = 1
                        placements = [(i, 0, 0, self.demands[i]["size"][0], self.demands[i]["size"][1])]
                        pattern = {
                            'counts': counts,
                            'placements': placements,
                            'unit_values': [0]
                        }
                        self.current_patterns.append(pattern)

                self.stock_size = self._get_stock_size_(stock_list[self.stock_idx][1])
                stockidxreturn = stock_list[self.stock_idx][0]
                while True:
                    self._solve_master_problem(demands_vector)
                    new_pattern = self._solve_subproblem()
                    if new_pattern is None:
                        break   
                    self.current_patterns.append(new_pattern)
                x = self.master_solution
                if x is None or len(x) == 0:
                    return {'stock_idx': -1, 'size': [0, 0], 'position': [0, 0]}

                Area_array = np.zeros(len(self.current_patterns))

                for i in range(len(self.current_patterns)):
                    area_i = 0
                    for j in range(self.num_products):
                        if x[i] > 0 or i > self.num_products:
                            area_i += self.demands[j]["size"][0] * self.demands[j]["size"][1] * self.current_patterns[i]["counts"][j]
                    Area_array[i] = area_i

                pattern_idx = np.argmax(Area_array)

                a = Area_array[pattern_idx] / (self.stock_size[0] * self.stock_size[1])
                selected_pattern = self.current_patterns[pattern_idx]
                placements = selected_pattern["placements"]
                self.action_queue = []

                #print(self.stock_idx)

                length = len(stock_list)

                if placements == []:
                    continue
                if a < 0.9 and self.iteration < 2 * length:
                    continue
                if a < 0.85 and self.iteration >= 2 * length and self.iteration < 4 * length:
                    continue
                if a < 0.8 and self.iteration >= 4 * length and self.iteration < 6 * length:
                    continue
                if a < 0.75 and self.iteration >= 6 * length and self.iteration < 8 * length: 
                    continue
                self.stock_placed.append(self.stock_idx)

                for placement in placements:
                    i, xi, yi, w, h = placement
                    self.action_queue.append({
                        "stock_idx": stockidxreturn,
                        "size": [w, h],
                        "position": [xi, yi]
                    })
                if self.action_queue:
                    return self.action_queue.pop(0)
                else:
                    return {'stock_idx': -1, 'size': [0, 0], 'position': [0, 0]}
                
        elif self.policy_id == 2:
            list_prods = observation["products"]
        # Sort products by area in descending order
            list_prods = sorted(
                list_prods,
                key=lambda x: x["size"][0] * x["size"][1],
                reverse=True
            )
            # Get stocks and their indices
            stock_tuple = observation["stocks"]
            stock_list = list(enumerate(stock_tuple))
            # Sort stocks by area in descending order
            sorted_stock_list = sorted(
                stock_list,
                key=lambda x: self._get_stock_area_(x[1]),
                reverse=True
            )

            prod_size = [0, 0]
            selected_stock_idx = -1
            pos_x, pos_y = None, None

            # Pick a product that has quantity > 0
            for prod in list_prods:
                if prod["quantity"] > 0:
                    prod_size = prod["size"]
                    prod_w, prod_h = prod_size

                    # Loop through all stocks
                    for stock_idx, stock_array in sorted_stock_list:
                        stock_w, stock_h = self._get_stock_size_(stock_array)

                        # Check if the product fits without rotation
                        if stock_w >= prod_w and stock_h >= prod_h:
                            for x in range(stock_w - prod_w + 1):
                                for y in range(stock_h - prod_h + 1):
                                    if self._can_place_(stock_array, (x, y), prod_size):
                                        pos_x, pos_y = x, y
                                        selected_stock_idx = stock_idx
                                        break
                                if pos_x is not None:
                                    break
                            if pos_x is not None:
                                break

                        # Check if the product fits with rotation
                        if stock_w >= prod_h and stock_h >= prod_w:
                            rotated_size = prod_size[::-1]
                            for x in range(stock_w - prod_h + 1):
                                for y in range(stock_h - prod_w + 1):
                                    if self._can_place_(stock_array, (x, y), rotated_size):
                                        prod_size = rotated_size
                                        pos_x, pos_y = x, y
                                        selected_stock_idx = stock_idx
                                        break
                                if pos_x is not None:
                                    break
                            if pos_x is not None:
                                break

                    if pos_x is not None:
                        break

            return {
                "stock_idx": selected_stock_idx,
                "size": prod_size,
                "position": (pos_x, pos_y)
            }