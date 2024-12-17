from policy import Policy
import numpy as np
from scipy.optimize import linprog

class Policy2352964_2353087_2352373_2350006_2353103(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in [1, 2], "Policy ID must be 1 or 2"
        self.policy_id = policy_id
        # Student code here
        if policy_id == 1:
            self.policy = VeryGreedy()  # Khởi tạo BFPolicy
        elif policy_id == 2:
            self.policy = ColumnGeneration()

    def get_action(self, observation, info):
        # Student code here
        if self.policy_id == 1 and self.policy:
            return self.policy.get_action(observation, info)

        elif self.policy_id == 2:
           return self.policy.get_action(observation, info)
        return None


    # Student code here
    # You can add more functions if needed
class VeryGreedy(Policy):
    def __init__(self):
        self.sorted_stocks_index = np.array([])
        self.sorted_products = []
        self.counter = 0

    def sort_stock_product(self, stock_array, prod_array):
        stock_areas = [self._get_stock_size_(stock)[0] * self._get_stock_size_(stock)[1] for stock in stock_array]
        self.sorted_stocks_index = np.argsort(stock_areas)[::-1]
        self.sorted_products = sorted(prod_array, key=lambda prod: prod['size'][0] * prod['size'][1], reverse=True)

    def get_action(self, observation, info):
        if self.counter == 0:
            self.sort_stock_product(observation['stocks'], observation['products'])
        self.counter += 1

        if self.sorted_products[-1]['quantity'] == 1:
            self.counter = 0

        selected_product_size = [0, 0]
        selected_stock_idx = -1
        position_x, position_y = 0, 0

        for product in self.sorted_products:
            if product["quantity"] > 0:
                selected_product_size = product["size"]

                for stock_idx in self.sorted_stocks_index:
                    stock_width, stock_height = self._get_stock_size_(observation['stocks'][stock_idx])
                    product_width, product_height = selected_product_size

                    if stock_width < product_width or stock_height < product_height:
                        continue

                    position_x, position_y = None, None
                    for x in range(stock_width - product_width + 1):
                        for y in range(stock_height - product_height + 1):
                            if self._can_place_(observation['stocks'][stock_idx], (x, y), selected_product_size):
                                position_x, position_y = x, y
                                break

                        if position_x is not None and position_y is not None:
                            break

                    if position_x is not None and position_y is not None:
                        selected_stock_idx = stock_idx
                        break

                if position_x is not None and position_y is not None:
                    break

        return {
            "stock_idx": selected_stock_idx,
            "stock_size": self._get_stock_size_(observation['stocks'][selected_stock_idx]),
            "size": selected_product_size,
            "position": (position_x, position_y)
        }

class ColumnGeneration(Policy):
    def __init__(self):
        super().__init__()
        self.pattern_catalog = []
        self.processed_patterns = set()
        self.current_stock_idx = 0
        self.occupied_positions = {}
        self.game_started = False

    def reset_planner(self):
        self.pattern_catalog.clear()
        self.processed_patterns.clear()
        self.current_stock_idx = 0
        self.occupied_positions.clear()
        self.game_started = False

    def create_patterns(self, products, stock_size):
        """Generate possible cutting patterns based on available products and stock size."""
        sorted_products = sorted(
            enumerate(products),
            key=lambda x: x[1]['size'][0] * x[1]['size'][1],
            reverse=True
        )
        pattern_list = []

        for idx, product in sorted_products:
            if product['quantity'] > 0:
                stock_w, stock_h = stock_size
                prod_w, prod_h = product['size']

                if prod_w > stock_w or prod_h > stock_h:
                    continue

                pattern = [0] * len(products)
                max_pieces_w = stock_w // prod_w
                max_pieces_h = stock_h // prod_h
                total_pieces = min(product['quantity'], max_pieces_w * max_pieces_h)

                if total_pieces > 0:
                    pattern[idx] = total_pieces
                    pattern_list.append(pattern)
        return pattern_list

    def get_action(self, observation, info):
        """Determine the next action based on the current observation."""
        if not self.game_started:
            self.reset_planner()
            self.game_started = True

        stocks = observation["stocks"]
        products = observation["products"]

        remaining_items = sum(product["quantity"] for product in products)
        if remaining_items == 0:
            return {
                "stock_idx": -1,
                "size": [0, 0],
                "position": [0, 0]
            }

        sorted_stocks = sorted(
            enumerate(stocks),
            key=lambda x: self.calculate_stock_area(x[1]),
            reverse=True
        )

        sorted_products = sorted(
            enumerate(products),
            key=lambda x: x[1]['size'][0] * x[1]['size'][1],
            reverse=True
        )

        while self.current_stock_idx < len(sorted_stocks):
            stock_idx, _ = sorted_stocks[self.current_stock_idx]
            if stock_idx not in self.occupied_positions:
                self.occupied_positions[stock_idx] = set()

            stock_size = self.extract_stock_dimensions(stocks[stock_idx])

            if not self.pattern_catalog:
                self.pattern_catalog = self.create_patterns(products, stock_size)
                self.processed_patterns.clear()

            pattern_found = False
            for pattern_idx, pattern in enumerate(self.pattern_catalog):
                if pattern_idx not in self.processed_patterns:
                    for prod_idx, count in enumerate(pattern):
                        if count > 0 and products[prod_idx]["quantity"] > 0:
                            valid_positions = self.identify_valid_positions(
                                stocks[stock_idx],
                                products[prod_idx]["size"]
                            )
                            if valid_positions:
                                for pos in valid_positions:
                                    position_key = (*pos, *products[prod_idx]["size"])
                                    if position_key not in self.occupied_positions[stock_idx]:
                                        self.occupied_positions[stock_idx].add(position_key)
                                        # Decrease product quantity
                                        products[prod_idx]["quantity"] -= 1
                                        self.processed_patterns.add(pattern_idx)
                                        return {
                                            "stock_idx": stock_idx,
                                            "size": products[prod_idx]["size"],
                                            "position": pos
                                        }
                    self.processed_patterns.add(pattern_idx)
            if not pattern_found:
                self.current_stock_idx += 1
                self.pattern_catalog = []
        # If no valid actions are found, return no-op action
        return {
            "stock_idx": -1,
            "size": [0, 0],
            "position": [0, 0]
        }

    def identify_valid_positions(self, stock, product_size):
        """Find all valid positions where a product can be placed on a stock."""
        stock_w, stock_h = self.extract_stock_dimensions(stock)
        prod_w, prod_h = product_size
        positions = []

        for x in range(stock_w - prod_w + 1):
            for y in range(stock_h - prod_h + 1):
                if self.is_placeable(stock, (x, y), product_size):
                    positions.append((x, y))
        return positions

    def is_placeable(self, stock, position, product_size):
        """Check if a product can be placed at the given position on the stock."""
        x_start, y_start = position
        w, h = product_size
        stock_w, stock_h = stock.shape

        if x_start + w > stock_w or y_start + h > stock_h:
            return False
        return np.all(stock[x_start:x_start+w, y_start:y_start+h] == -1)

    def extract_stock_dimensions(self, stock):
        """Extract the usable width and height of the stock."""
        stock_array = np.atleast_2d(stock)
        width = stock_array.shape[0]
        height = stock_array.shape[1]
        return width, height

    def calculate_stock_area(self, stock):
        """Calculate the usable area of the stock."""
        width, height = self.extract_stock_dimensions(stock)
        return width * height






