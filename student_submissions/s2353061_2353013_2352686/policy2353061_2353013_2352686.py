import numpy as np
from scipy.optimize import linprog
from policy import Policy

class Policy2353061(Policy):  # Best-fit approach
    def __init__(self, policy_id=1):
        super().__init__()
        self.policy_id = policy_id
        self.current_stock_idx = 0
        self.sorted_stocks = []
        self.products = []
        self.remaining_product_area = 0
        self.total_product_area = 0
        self.allow_rotation = True
        self.dynamic_mode = True

    def _calculate_stock_area(self, stock):
        width, height = self._get_stock_size_(stock)
        return width * height

    def _current_stock_size(self, stock):
        return (np.max(np.sum(np.any(stock > -1, axis=1))),
                np.max(np.sum(np.any(stock > -1, axis=0))))

    def _find_best_position(self, stock, product, orientations):
        stock_width, stock_height = self._get_stock_size_(stock)
        current_width, current_height = self._current_stock_size(stock)
        best_fit_score = float('inf')
        chosen_position = None
        chosen_orientation = None

        for product_width, product_height in orientations:
            if stock_width < product_width or stock_height < product_height:
                continue

            for x in range(stock_width - product_width + 1):
                if x > current_width:
                    continue

                for y in range(stock_height - product_height + 1):
                    if y > current_height:
                        continue

                    if self._can_place_(stock, (x, y), (product_width, product_height)):
                        proximity_to_edges = x + y
                        new_area = max(current_height, y + product_height) * max(current_width, x + product_width)
                        area_increase = (new_area - current_width * current_height) / (product_width * product_height)
                        fit_score = area_increase + proximity_to_edges / 10

                        if fit_score < best_fit_score:
                            best_fit_score = fit_score
                            chosen_position = (x, y)
                            chosen_orientation = (product_width, product_height)

        return chosen_position, chosen_orientation

    def _generate_orientations(self, product):
        width, height = product["size"]
        orientations = [(width, height)]
        if self.allow_rotation and width != height:
            orientations.append((height, width))
        return orientations

    def get_action(self, observation, info):
        if info["filled_ratio"] == 0:
            self.__init__(policy_id=self.policy_id)
            self.total_product_area = sum(np.prod(prod["size"]) * prod["quantity"] for prod in observation["products"])
            self.remaining_product_area = self.total_product_area

            self.sorted_stocks = sorted(
                enumerate(observation["stocks"]),
                key=lambda x: self._calculate_stock_area(x[1]),
                reverse=True
            )

            self.products = sorted(
                observation["products"],
                key=lambda x: np.prod(x["size"]),
                reverse=True
            )

        while self.current_stock_idx < len(self.sorted_stocks):
            stock_index, _ = self.sorted_stocks[self.current_stock_idx]
            stock = observation["stocks"][stock_index]

            for product in self.products:
                if product["quantity"] <= 0:
                    continue

                orientations = self._generate_orientations(product)
                chosen_position, chosen_orientation = self._find_best_position(stock, product, orientations)

                if chosen_position:
                    product["quantity"] -= 1
                    self.remaining_product_area -= np.prod(chosen_orientation)
                    return {
                        "stock_idx": stock_index,
                        "size": chosen_orientation,
                        "position": chosen_position
                    }

            self.current_stock_idx += 1

        return {"stock_idx": -1, "size": [0, 0], "position": (-1, -1)}


class Policy2352686(Policy):  # Column generation approach
    def __init__(self, policy_id=2):
        super().__init__()
        self.policy_id = policy_id
        self.stock_size = None
        self.products = None
        self.patterns = []
        self.dual_values = None

    def _initialize_patterns(self):
        patterns = []
        for i, product in enumerate(self.products):
            pattern = np.zeros(len(self.products))
            pattern[i] = 1
            patterns.append(pattern)
        return patterns

    def _solve_master_problem(self):
        if not self.patterns:
            return None

        num_patterns = len(self.patterns)
        num_products = len(self.products)
        A = np.zeros((num_products, num_patterns))
        for j, pattern in enumerate(self.patterns):
            A[:, j] = pattern

        c = np.ones(num_patterns)
        b = np.array([product["quantity"] for product in self.products])

        res = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if res.success:
            self.dual_values = res.get("y", np.zeros(num_products))
        else:
            print("Master problem solution failed.")
            self.dual_values = None

    def _solve_subproblem(self):
        num_products = len(self.products)
        stock_width, stock_height = self.stock_size

        dual_values = self.dual_values if self.dual_values is not None else np.zeros(num_products)
        profit = dual_values
        new_pattern = np.zeros(num_products)
        remaining_area = stock_width * stock_height

        for i, product in sorted(enumerate(self.products), key=lambda x: profit[x[0]], reverse=True):
            size = product["size"]
            area = size[0] * size[1]
            max_quantity = min(product["quantity"], remaining_area // area)
            new_pattern[i] = max_quantity
            remaining_area -= max_quantity * area

        return new_pattern

    def get_action(self, observation, info):
        if self.stock_size is None:
            self.stock_size = observation["stocks"][0].shape
            self.products = observation["products"]
            self.patterns = self._initialize_patterns()

        self._solve_master_problem()

        new_pattern = self._solve_subproblem()
        if np.sum(new_pattern) > 0:
            self.patterns.append(new_pattern)

        for stock_idx, stock in enumerate(observation["stocks"]):
            stock_w, stock_h = self._get_stock_size_(stock)
            for i, product in enumerate(self.products):
                if product["quantity"] > 0:
                    size = product["size"]
                    prod_w, prod_h = size

                    if stock_w >= prod_w and stock_h >= prod_h:
                        for x in range(stock_w - prod_w + 1):
                            for y in range(stock_h - prod_h + 1):
                                if self._can_place_(stock, (x, y), size):
                                    return {
                                        "stock_idx": stock_idx,
                                        "size": size,
                                        "position": (x, y),
                                    }

        return {"stock_idx": -1, "size": (0, 0), "position": (0, 0)}

    def reset(self):
        self.stock_size = None
        self.products = None
        self.patterns = []
        self.dual_values = None


class Policy2353013_2352686_2353061__Group(Policy):
    def __init__(self, policy_id):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        if policy_id == 1:
            self.policy = Policy2353061(policy_id)
        elif policy_id == 2:
            self.policy = Policy2352686(policy_id)

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)
