# from policy import Policy
import numpy as np
from scipy.optimize import linprog

from abc import abstractmethod

import numpy as np


class Policy:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation, info):
        pass

    def _get_stock_size_(self, stock):
        stock_w = np.sum(np.any(stock != -2, axis=1))
        stock_h = np.sum(np.any(stock != -2, axis=0))

        return stock_w, stock_h

    def _can_place_(self, stock, position, prod_size):
        pos_x, pos_y = position
        prod_w, prod_h = prod_size

        return np.all(stock[pos_x : pos_x + prod_w, pos_y : pos_y + prod_h] == -1)

class Policy2311650_2311538_2311486(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"

        # Student code here
        if policy_id == 1:
            self.policy = BestFit()
        elif policy_id == 2:
            self.policy = ColumnGenerationPolicy()

    # # Student code here
    # # You can add more functions if needed
    def get_action(self, observation, info):
        # Gọi hàm get_action từ policy được khởi tạo
        return self.policy.get_action(observation, info)

########################################################

class BestFit(Policy):
    ###########################
    ##### Best Fit Policy #####
    def __init__(self):
        # Student code here
        self.total_wasted_area = 0
        self.total_area_of_all_stocks = 0
        self.total_stock_used = 0    
        self.total_products_cut = 0
        self.computation_time = 0

    def get_action(self, observation, info):
        # start_time = time.time()
        list_prods = observation["products"]
        stocks = observation["stocks"]

        best_stock_idx = -1
        best_position = (None, None)
        best_prod_size = [0, 0]
        min_wasted_space = float('inf')  # Không gian lãng phí tối thiểu

        # Lặp qua từng sản phẩm có số lượng > 0
        for prod in list_prods:
            if prod["quantity"] <= 0:
                continue

            prod_size = prod["size"]
            prod_area = prod_size[0] * prod_size[1]

            # Lặp qua từng tấm
            for stock_idx, stock in enumerate(stocks):
                stock_w, stock_h = self._get_stock_size_(stock)
                # Tìm tổng diện tích khả dụng
                stock_area = np.sum(stock == -1)

                if stock_area < prod_area:
                    continue  # Không đủ diện tích để đặt

                # Lặp qua các vị trí trên tấm
                for x in range(stock_w - prod_size[0] + 1):
                    for y in range(stock_h - prod_size[1] + 1):
                        if self._can_place_(stock, (x, y), prod_size):
                            # Tính không gian bị lãng phí
                            wasted_space = stock_area - prod_area
                            if wasted_space < min_wasted_space:
                                # Cập nhật lựa chọn tốt nhất
                                min_wasted_space = wasted_space
                                best_stock_idx = stock_idx
                                best_position = (x, y)
                                best_prod_size = prod_size
        # Tính toán các metrics
        if best_stock_idx != -1:
            stock = stocks[best_stock_idx]
            stock_w, stock_h = self._get_stock_size_(stock)
            stock_area = stock_w * stock_h
            self.total_area_of_all_stocks += stock_area

            # Cập nhật không gian lãng phí
            self.total_wasted_area += min_wasted_space

            # Cập nhật số lượng tấm được sử dụng
            self.total_stock_used += 1

            # Cập nhật số lượng sản phẩm cắt được
            self.total_products_cut += 1

        # end_time = time.time()  # Kết thúc tính thời gian thực thi
        # self.computation_time += (end_time - start_time)
        return {
            "stock_idx": best_stock_idx,
            "size": best_prod_size,
            "position": best_position,
        }
        
    # def get_metrics(self):
    #     # Tính toán Waste Ratio
    #     waste_ratio = self.total_wasted_area / self.total_area_of_all_stocks if self.total_area_of_all_stocks != 0 else 0

    #     return {
    #         "Waste Ratio": waste_ratio,
    #         "Number of Stock Used": self.total_stock_used,
    #         "Number of Products Cut": self.total_products_cut,
    #         "Computation Time (s)": self.computation_time,
    #     }
    
class ColumnGenerationPolicy(Policy):
    def __init__(self):
        self.stock_width = 10
        self.stock_height = 10
        self.starting_cut_patterns = None
        self.required_quantities = None

    def get_action(self, observation, info):
        items = observation["products"]
        stockpile = observation["stocks"]

        self._DATAPREPARE_(items)

        # Solve with column generation
        cut_solution = self._CG_PROCESSOR_(stockpile)

        # Choose an action based on the solution
        move = self._HELPER_ACCTIONI_(stockpile, cut_solution)
        return move

    def _DATAPREPARE_(self, products):
        self.required_quantities = np.array([product["quantity"] for product in products])
        self.product_dimensions = np.array([product["size"] for product in products])

        # Initialize with basic cutting patterns
        num_products = len(self.product_dimensions)
        self.initial_cut_patterns = np.eye(num_products, dtype=int)

    def _CG_PROCESSOR_(self, stockpile):
        is_new_pattern_found = True
        new_cut_pattern = None
        current_patterns = self.initial_cut_patterns

        while is_new_pattern_found:
            if new_cut_pattern is not None:
                current_patterns = np.column_stack((current_patterns, new_cut_pattern))

            dual_values = self._lp_relaxation_solution(current_patterns)
            is_new_pattern_found, new_cut_pattern = self._find_other_pattern_(dual_values, stockpile)

        optimal_stock_count, pattern_allocations = self._master_solution_(current_patterns)
        return {
            "cut_patterns": current_patterns,
            "minimal_stock": optimal_stock_count,
            "pattern_allocations": pattern_allocations,
        }

    def _solve_linear_problem(self, objective_coeffs, constraint_matrix, constraint_bounds, variable_bounds):
        res = linprog(
            objective_coeffs,
            A_ub=constraint_matrix,
            b_ub=constraint_bounds,
            bounds=variable_bounds,
            method="highs",
        )
        if not res.success:
            raise ValueError("Linear programming problem could not be solved.")
        return res

    def _lp_relaxation_solution(self, cut_patterns):
        num_patterns = cut_patterns.shape[1]

        objective_coeffs = np.ones(num_patterns)
        constraint_matrix = -cut_patterns
        constraint_bounds = -self.required_quantities
        variable_bounds = [(0, None) for _ in range(num_patterns)]

        result = self._solve_linear_problem(objective_coeffs, constraint_matrix, constraint_bounds, variable_bounds)
        return result.slack

    def _master_solution_(self, cut_patterns):
        num_patterns = cut_patterns.shape[1]

        objective_coeffs = np.ones(num_patterns)
        constraint_matrix = -cut_patterns
        constraint_bounds = -self.required_quantities
        variable_bounds = [(0, None) for _ in range(num_patterns)]

        result = self._solve_linear_problem(objective_coeffs, constraint_matrix, constraint_bounds, variable_bounds)

        # Convert LP solution to integer solution
        allocation_variables = np.round(result.x).astype(int)
        optimal_stock_count = allocation_variables.sum()
        return optimal_stock_count, allocation_variables

    def _solve_pattern_finding_problem(self, obj_coeffs, constraint_matrix, constraint_bounds, var_bounds):
        res = linprog(obj_coeffs, A_ub=constraint_matrix, b_ub=constraint_bounds, bounds=var_bounds, method="highs")
        if not res.success:
            raise ValueError("Pattern finding problem could not be solved.")
        return res

    def _find_other_pattern_(self, dual_values, stockpile):
        num_items = len(self.product_dimensions)

        obj_coeffs = dual_values - 1
        constraint_matrix = [
            self.product_dimensions[:, 0],  # Widths of items
            self.product_dimensions[:, 1]   # Heights of items
        ]
        constraint_bounds = [self.stock_width, self.stock_height]  # Stock dimensions
        var_bounds = [(0, None) for _ in range(num_items)]

        result = self._solve_pattern_finding_problem(obj_coeffs, constraint_matrix, constraint_bounds, var_bounds)

        cutting_pattern = np.round(result.x).astype(int)
        reduced_cost = 1 - dual_values @ cutting_pattern

        if reduced_cost < 0:
            return True, cutting_pattern
        else:
            return False, None

    def _find_placement_in_stock(self, stock, stock_size, product_size):
        """
        Tìm vị trí hợp lệ để đặt sản phẩm trong tấm nguyên liệu.

        Parameters:
            stock (array-like): Tấm nguyên liệu hiện tại.
            stock_size (tuple): Kích thước của tấm nguyên liệu (rộng, cao).
            product_size (tuple): Kích thước của sản phẩm (rộng, cao).

        Returns:
            tuple: (bool, tuple) 
                - True và vị trí (x, y) nếu tìm thấy vị trí hợp lệ.
                - False và (0, 0) nếu không tìm thấy vị trí hợp lệ.
        """
        stock_w, stock_h = stock_size
        prod_w, prod_h = product_size

        # Đảm bảo sản phẩm có thể vừa trong tấm nguyên liệu
        if stock_w < prod_w or stock_h < prod_h:
            return False, (0, 0)

        # Thử tìm vị trí đặt hợp lệ
        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                # Kiểm tra liệu sản phẩm có thể đặt tại vị trí (x, y)
                if self._can_place_(stock, (x, y), product_size):
                    return True, (x, y)

        # Không tìm thấy vị trí hợp lệ
        return False, (0, 0)

    def _HELPER_ACCTIONI_(self, stockpile, cut_solution):
        for stock_idx, stock in enumerate(stockpile):
            stock_size = self._get_stock_size_(stock)

            for pattern, qty in zip(cut_solution["cut_patterns"].T, cut_solution["pattern_allocations"]):
                if qty > 0:
                    for prod_idx, prod_count in enumerate(pattern):
                        if prod_count > 0:
                            product_size = self.product_dimensions[prod_idx]
                            can_place, position = self._find_placement_in_stock(stock, stock_size, product_size)
                            if can_place:
                                return {"stock_idx": stock_idx, "size": product_size, "position": position}

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}